"""Execution-aligned graph learner with per-cycle auxiliary supervision."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

N_TYPES = 39


class MLP(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, o))

    def forward(self, x):
        return self.net(x)


class ExecCore(nn.Module):
    def __init__(self, d=128, hidden=256, slots=3, llm_dim=0, use_width=True, sync=False):
        super().__init__()
        self.d, self.slots = d, slots
        self.sync = sync
        self.type_emb = nn.Embedding(N_TYPES, d)
        self.width_emb = nn.Embedding(7, d) if use_width else None
        self.role_emb = nn.Embedding(4, d)
        self.static_norm = nn.LayerNorm(d)
        self.llm = None
        if llm_dim:
            self.llm = nn.Sequential(nn.LayerNorm(llm_dim), nn.Linear(llm_dim, d))
            nn.init.zeros_(self.llm[1].weight)
            nn.init.zeros_(self.llm[1].bias)
        self.in_enc = MLP(32 + d, hidden, d)
        self.seq_init = MLP(32 + d, hidden, d)
        self.seq_upd = MLP(3 * d, hidden, d)
        self.comb = MLP((slots + 2) * d, hidden, d)
        self.norm = nn.LayerNorm(d)
        self.bits_head = nn.Linear(d, 32)
        self.nz_head = nn.Linear(d, 1)
        self.ch_head = MLP(2 * d, hidden, 1)
        self.register_buffer("shifts", torch.arange(31, -1, -1), persistent=False)

    def schedule(self, b):
        """Per level: combinational nodes, their incoming edges, and local slot indices."""
        role, level, dst = b["role"], b["level"], b["dst"]
        comb = role == 3
        edge_level = torch.where(comb[dst], level[dst], torch.full_like(dst, -1))
        out = []
        n = role.numel()
        local = torch.full((n,), -1, dtype=torch.long, device=role.device)
        for L in torch.unique(level[comb]).tolist():
            nodes = (comb & (level == L)).nonzero().flatten()
            local[nodes] = torch.arange(len(nodes), device=role.device)
            e = (edge_level == L).nonzero().flatten()
            out.append((nodes, e, local[b["dst"][e]]))
        return out

    def sync_plan(self, b, rounds):


        role, dst = b["role"], b["dst"]
        nodes = (role == 3).nonzero().flatten()
        local = torch.full((role.numel(),), -1, dtype=torch.long, device=role.device)
        local[nodes] = torch.arange(len(nodes), device=role.device)
        e = (role[dst] == 3).nonzero().flatten()
        slot = b["pos"][e].clamp(min=1, max=self.slots) - 1
        return nodes, b["src"][e], b["delayed"][e], local[dst[e]] * self.slots + slot, rounds

    def sync_rounds(self, plan, H, H_prev, static):


        nodes, src, delayed, idx, rounds = plan
        if not len(nodes):
            return H
        d, P = self.d, self.slots
        H = H.index_copy(0, nodes, H_prev[nodes])

        def step(H):
            h_src = torch.where(delayed[:, None], H_prev[src], H[src])
            agg = torch.zeros(len(nodes) * P, d, device=H.device).index_add(0, idx, h_src)
            x = torch.cat([agg.view(len(nodes), P * d), static[nodes], H_prev[nodes]], -1)
            return H.index_copy(0, nodes, self.norm(self.comb(x)))

        for _ in range(rounds):
            H = torch.utils.checkpoint.checkpoint(step, H, use_reentrant=False) if self.training else step(H)
        return H

    def forward(self, b, T):
        role = b["role"]
        n, d, P = role.numel(), self.d, self.slots
        dev = role.device
        static = self.type_emb(b["types"])
        if self.width_emb is not None:
            static = static + self.width_emb(b["wb"])
        static = static + self.role_emb(role)
        if self.llm is not None:
            static = static + self.llm(b["llm"])
        static = self.static_norm(static)
        bits = ((b["values"][:, :T, None] >> self.shifts) & 1).float()
        in_idx = (role <= 1).nonzero().flatten()
        seq_idx, seq_d, seq_init = b["seq_idx"], b["seq_d"], b["seq_init"]
        sched = self.schedule(b)
        if self.sync:
            sync = self.sync_plan(b, len(sched))
        src_all, pos_all, del_all = b["src"], b["pos"], b["delayed"]
        zero = torch.zeros(n, d, device=dev)
        H_prev = zero
        bits_logits, nz_logits, ch_logits = [], [], []
        init_bits = torch.where((seq_init >= 0)[:, None], bits[seq_init.clamp(min=0), 0],
                                torch.zeros(len(seq_idx), 32, device=dev))
        for t in range(T):
            H = zero
            h_in = self.norm(self.in_enc(torch.cat([bits[in_idx, t], static[in_idx]], -1)))
            H = H.index_copy(0, in_idx, h_in)
            if len(seq_idx):
                if t == 0:
                    h_s = self.seq_init(torch.cat([init_bits, static[seq_idx]], -1))
                else:
                    h_d = torch.where((seq_d >= 0)[:, None], H_prev[seq_d.clamp(min=0)],
                                      torch.zeros(len(seq_idx), d, device=dev))
                    h_s = self.seq_upd(torch.cat([static[seq_idx], h_d, H_prev[seq_idx]], -1))
                H = H.index_copy(0, seq_idx, self.norm(h_s))
            if self.sync:
                H = self.sync_rounds(sync, H, H_prev, static)
            else:
                for nodes, e, loc in sched:
                    src = src_all[e]
                    h_src = torch.where(del_all[e][:, None], H_prev[src], H[src])
                    slot = pos_all[e].clamp(min=1, max=P) - 1
                    agg = torch.zeros(len(nodes) * P, d, device=dev).index_add(0, loc * P + slot, h_src)
                    x = torch.cat([agg.view(len(nodes), P * d), static[nodes], H_prev[nodes]], -1)
                    H = H.index_copy(0, nodes, self.norm(self.comb(x)))
            bits_logits.append(self.bits_head(H))
            nz_logits.append(self.nz_head(H).squeeze(-1))
            if t > 0:
                ch_logits.append(self.ch_head(torch.cat([H, H_prev], -1)).squeeze(-1))
            H_prev = H
        return torch.stack(bits_logits, 1), torch.stack(nz_logits, 1), torch.stack(ch_logits, 1)


def targets(b, T):
    """Exact per-cycle hints and the window labels (integer comparisons)."""
    v = b["values"][:, :T]
    shifts = torch.arange(31, -1, -1, device=v.device)
    bits = ((v[:, :, None] >> shifts) & 1).float()
    nz = (v != 0).float()
    ch = (v[:, 1:] != v[:, :-1]).float()
    toggle = ch.sum(1) / T
    branch = (v[b["sel"]] != 0).any(1).float()
    return bits, nz, ch, toggle, branch


def predictions(nz_logits, ch_logits, b, T):
    toggle = torch.sigmoid(ch_logits).sum(1) / T
    s = F.softplus(nz_logits[b["sel"]]).sum(1)
    branch_logit = torch.log((-torch.expm1(-s)).clamp(min=1e-7)) + s
    return toggle, branch_logit


def loss_fn(out, b, T, w):
    bits_logits, nz_logits, ch_logits = out
    bits, nz, ch, toggle, branch = targets(b, T)
    label_nodes = b["role"] >= 2
    rec = label_nodes & b["has"]
    l_bits = F.binary_cross_entropy_with_logits(bits_logits[rec], bits[rec])
    l_nz = F.binary_cross_entropy_with_logits(nz_logits[rec], nz[rec])
    l_ch = F.binary_cross_entropy_with_logits(ch_logits[rec], ch[rec])
    p_tog, br_logit = predictions(nz_logits, ch_logits, b, T)
    l_tog = (p_tog[label_nodes] - toggle[label_nodes]).abs().mean()
    l_br = F.binary_cross_entropy_with_logits(br_logit, branch) if len(branch) else p_tog.sum() * 0
    total = w["bits"] * l_bits + w["nz"] * l_nz + w["ch"] * l_ch + w["tog"] * l_tog + w["br"] * l_br
    return total, {"bits": l_bits.item(), "nz": l_nz.item(), "ch": l_ch.item(),
                   "tog": l_tog.item(), "br": l_br.item()}

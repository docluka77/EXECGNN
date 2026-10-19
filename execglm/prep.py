"""Prepare graph roles, operand slots, execution levels and integer trace caches."""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from execglm.operators import OP
INPUT, CONST, REG, COND, OUTPUT = OP["Input"], OP["Const"], OP["Reg"], OP["Cond"], OP["Output"]
W32 = (2 ** np.arange(31, -1, -1, dtype=np.int64))


def design_record(name, p):
    x = np.asarray(p["x"])
    types = x[:, 1].astype(np.int64)
    width = x[:, 2].astype(np.int64)
    n = len(types)
    ei = np.asarray(p["edge_index"]).reshape(-1, 2).astype(np.int64)
    et = np.asarray(p["edge_type"]).reshape(-1).astype(np.int64)
    ins_t = [[] for _ in range(n)]
    for (s, d), t in zip(ei, et):
        ins_t[int(d)].append((int(t), int(s)))
    ins_t = [sorted(v) for v in ins_t]
    ins = [[s for _, s in v] for v in ins_t]

    def reads_back(out):
        seen, stack = set(), list(ins[out])
        while stack:
            v = stack.pop()
            if v == out:
                return True
            if v in seen or types[v] in (INPUT, CONST, REG):
                continue
            seen.add(v)
            stack.extend(ins[v])
        return False

    role = np.full(n, 3, dtype=np.int64)
    role[types == INPUT] = 0
    role[types == CONST] = 1
    seq_d = np.full(n, -1, dtype=np.int64)
    seq_init = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        seq = types[i] == REG or (types[i] == OUTPUT and (
            reads_back(i) or (len(ins[i]) >= 2 and any(types[s] == CONST for s in ins[i]))))
        if not seq:
            continue
        role[i] = 2
        consts = [s for s in ins[i] if types[s] == CONST]
        ds = [s for s in ins[i] if types[s] != CONST]
        dsrc = ds[0] if ds else (consts[0] if consts else -1)
        seq_d[i] = dsrc
        seq_init[i] = consts[0] if (consts and ds) else -1


    level = np.zeros(n, dtype=np.int64)
    comb = [i for i in range(n) if role[i] == 3]
    indeg = {i: sum(1 for s in ins[i] if role[s] == 3) for i in comb}
    outs = [[] for _ in range(n)]
    for i in comb:
        for s in ins[i]:
            if role[s] == 3:
                outs[s].append(i)
    q = deque(i for i in comb if indeg[i] == 0)
    done = set()
    while q:
        i = q.popleft()
        done.add(i)
        level[i] = 1 + max([level[s] for s in ins[i] if role[s] == 3], default=0)
        for d in outs[i]:
            indeg[d] -= 1
            if indeg[d] == 0:
                q.append(d)
    rest = [i for i in comb if i not in done]
    top = int(level.max()) if n else 0
    for k, i in enumerate(sorted(rest)):
        level[i] = top + 1 + k

    src, dst, pos, delayed = [], [], [], []
    for d in range(n):
        for t, s in ins_t[d]:
            src.append(s); dst.append(d); pos.append(t)
            delayed.append(int(role[d] == 3 and role[s] == 3 and level[s] >= level[d]))

    sel = sorted({s for d in range(n) if types[d] == COND for t, s in ins_t[d] if t == 1})

    sim = np.asarray(p["sim_res"]).astype(np.int64)
    values = (sim * W32).sum(-1)
    return {
        "name": name, "n": n,
        "types": torch.from_numpy(types), "width": torch.from_numpy(width),
        "has": torch.tensor(np.asarray(p["has_sim_res"]).astype(bool)),
        "role": torch.from_numpy(role), "level": torch.from_numpy(level),
        "seq_d": torch.from_numpy(seq_d), "seq_init": torch.from_numpy(seq_init),
        "src": torch.tensor(src, dtype=torch.long), "dst": torch.tensor(dst, dtype=torch.long),
        "pos": torch.tensor(pos, dtype=torch.long), "delayed": torch.tensor(delayed, dtype=torch.bool),
        "sel": torch.tensor(sel, dtype=torch.long),
        "values": torch.from_numpy(values),
        "loops": len(rest),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True, help="DynamicRTL graphs*.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    designs = np.load(args.graphs, allow_pickle=True)["designs"].item()
    print(f"loaded {len(designs)} designs in {time.time() - t0:.0f}s", flush=True)
    recs = {}
    for k, name in enumerate(sorted(designs)):
        recs[name] = design_record(name, designs[name])
        if k % 400 == 0:
            print(f"  {k}/{len(designs)}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(recs, args.out)
    loops = sum(r["loops"] > 0 for r in recs.values())
    print(f"saved {len(recs)} designs to {args.out}; designs with combinational loops: {loops}; "
          f"max level {max(int(r['level'].max()) for r in recs.values())}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

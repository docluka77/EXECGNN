"""Batch circuit graphs and load design-disjoint training splits."""
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

OFFICIAL = Path(os.environ.get("EXECGNN_OFFICIAL_DIR", "data/official"))
STRUCTURAL = Path(os.environ.get("EXECGNN_STRUCTURAL_FILE", "data/structural.json"))


def width_bucket(w):
    b = torch.full_like(w, 6)
    b[w <= 32] = 5
    b[w <= 16] = 4
    b[w <= 8] = 3
    b[w <= 4] = 2
    b[w == 1] = 1
    b[w <= 0] = 0
    return b


def split_items(cache, split, split_file=None):
    """(train, validation) lists of (design, trace, name)."""
    def items(names):
        out = []
        for nm in names:
            d, t = nm.rsplit("_trace", 1)
            if d in cache:
                out.append((d, int(t), nm))
        return out
    if split in ("official", "custom"):
        path = Path(split_file) if split_file else OFFICIAL / "split.json"
        s = json.loads(path.read_text())
        if "train_trace_names" in s:
            tr, va = items(s["train_trace_names"]), items(s["validation_trace_names"])
            if len(tr) != len(s["train_trace_names"]) or len(va) != len(s["validation_trace_names"]):
                raise ValueError("split references a design absent from the cache")
        else:
            if not set(s["train"]).union(s["validation"]).issubset(cache):
                raise ValueError("split references a design absent from the cache")
            def expand(names):
                return [(d,t,f"{d}_trace{t}") for d in names for t in range(cache[d]["values"].shape[0])]
            tr, va = expand(s["train"]), expand(s["validation"])
        if not tr or not va or {d for d,_,_ in tr}.intersection(d for d,_,_ in va):
            raise ValueError("nonempty, design-disjoint train/validation groups required")
        for group in (tr, va):
            if len({nm for _,_,nm in group}) != len(group):raise ValueError("duplicate trace in split")
            for d,t,_ in group:
                if not 0 <= t < cache[d]["values"].shape[0]:raise ValueError("trace index outside cache")
        return tr, va
    if split == "size":
        small = sorted(d for d, r in cache.items() if r["n"] < 80)
        large = sorted(d for d, r in cache.items() if r["n"] >= 80)
        tr = [f"{d}_trace{t}" for d in small for t in range(cache[d]["values"].shape[0])]
        va = [f"{d}_trace{t}" for d in large for t in range(cache[d]["values"].shape[0])]
        return items(tr), items(va)
    raise ValueError(split)


def official_toggle_targets():
    rows = {}
    for line in open(OFFICIAL / "pretrain/validation/predictions.jsonl"):
        r = json.loads(line)
        rows[r["trace_name"]] = torch.tensor(r["toggle_target"], dtype=torch.float32)
    return rows


def novel_designs():
    part = json.loads(STRUCTURAL.read_text())["structural"]["partition"]
    val = set(json.loads((OFFICIAL / "split.json").read_text())["validation_design_ids"])
    return val - set(part["duplicate_design_ids"])


class LLMCache:
    """Node-aligned language vectors from a LACE feature cache (one row per node per trace)."""

    def __init__(self, directory):
        c = torch.load(Path(directory) / "features.pt", map_location="cpu", weights_only=False)
        self.index = {n: i for i, n in enumerate(c["names"])}
        self.pointer = c["pointer"]
        self.features = c["features"].float()
        cov = c["coverage"]
        self.coverage = (cov if cov.dim() == 1 else cov[:, 0]).bool()
        self.dim = self.features.shape[1]

    def get(self, name, n):
        i = self.index[name]
        a, b = int(self.pointer[i]), int(self.pointer[i + 1])
        assert b - a == n, (name, b - a, n)
        return self.features[a:b] * self.coverage[a:b, None]


def collate(items, cache, llm=None):
    parts = defaultdict(list)
    offs = 0
    for g, (dn, tr, nm) in enumerate(items):
        r = cache[dn]
        n = r["n"]
        for k in ("types", "role", "level", "has"):
            parts[k].append(r[k])
        parts["wb"].append(width_bucket(r["width"]))
        parts["src"].append(r["src"] + offs)
        parts["dst"].append(r["dst"] + offs)
        parts["pos"].append(r["pos"])
        parts["delayed"].append(r["delayed"])
        seq = (r["role"] == 2).nonzero().flatten()
        sd, si = r["seq_d"][seq], r["seq_init"][seq]
        parts["seq_idx"].append(seq + offs)
        parts["seq_d"].append(torch.where(sd >= 0, sd + offs, sd))
        parts["seq_init"].append(torch.where(si >= 0, si + offs, si))
        parts["values"].append(r["values"][tr])
        parts["sel"].append(r["sel"] + offs)
        parts["sel_graph"].append(torch.full((len(r["sel"]),), g, dtype=torch.long))
        parts["graph"].append(torch.full((n,), g, dtype=torch.long))
        if llm is not None:
            parts["llm"].append(llm.get(nm, n))
        offs += n
    batch = {k: torch.cat(v) for k, v in parts.items()}
    batch["names"] = [nm for _, _, nm in items]
    batch["n_graphs"] = len(items)
    return batch


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

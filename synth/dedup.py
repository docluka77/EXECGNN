"""Filter structural duplicates using operand-aware Weisfeiler-Lehman hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

SPLIT = Path(os.environ.get("EXECGNN_OFFICIAL_DIR", "data/official")) / "split.json"


def wl_hash(r, rounds=4):
    n = r["n"]
    lab = [f"{int(t)}/{int(w)}/{int(o)}" for t, w, o in zip(r["types"], r["width"], r["role"])]
    ins = [[] for _ in range(n)]
    for s, d, p in zip(r["src"].tolist(), r["dst"].tolist(), r["pos"].tolist()):
        ins[d].append((p, s))
    for _ in range(rounds):
        lab = [hashlib.sha1((lab[i] + "|" + ",".join(f"{p}:{lab[s]}" for p, s in sorted(ins[i]))).encode())
               .hexdigest()[:16] for i in range(n)]
    return hashlib.sha1(",".join(sorted(lab)).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--train", nargs="+", required=True, help="cache.pt or cache.pt:official (train designs)")
    ap.add_argument("--keep-out", required=True)
    args = ap.parse_args()
    seen = {}
    for spec in args.train:
        path, which = (spec[:-9], "official") if spec.endswith(":official") else (spec, "")
        c = torch.load(path, weights_only=False)
        names = sorted(c)
        if which == "official":
            ids = set(json.load(open(SPLIT))["train_design_ids"])
            names = [d for d in names if d in ids]
        for d in names:
            seen.setdefault(wl_hash(c[d]), d)
    test = torch.load(args.test, weights_only=False)
    keep, dup = [], {}
    for d in sorted(test):
        h = wl_hash(test[d])
        if h in seen:
            dup[d] = seen[h]
        else:
            keep.append(d)
    Path(args.keep_out).write_text("\n".join(keep))
    print(f"{args.test}: {len(test)} designs, {len(dup)} structural duplicates of training designs, "
          f"{len(keep)} kept")
    for d, s in list(dup.items())[:10]:
        print("  dup", d, "<-", s)


if __name__ == "__main__":
    main()

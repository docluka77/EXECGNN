"""Build training caches and optionally match a reference node-count histogram."""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


from execglm.prep import design_record

from execglm.operators import OP
NAME = {v: k for k, v in OP.items()}


def bin_of(n):
    return min(n // 10, 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default="")
    ap.add_argument("--match", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev-frac", type=float, default=0.0,
                    help="hold out this fraction of designs (never trained on) for the error profile")
    ap.add_argument("--dev-out", default="")
    ap.add_argument("--names", default="", help="restrict to these design ids (one per line)")
    args = ap.parse_args()
    files = sorted(Path(args.src).glob("*.npz"))
    if args.names:
        want_ids = set(Path(args.names).read_text().split())
        files = [f for f in files if f.stem in want_ids]
    sizes = {}
    for f in files:
        with np.load(f) as z:
            sizes[f.stem] = int(z["x"].shape[0])
    names = sorted(sizes)
    if args.match:
        ref = json.load(open(args.match))
        want = {int(k): v for k, v in ref["size_hist"].items()}
        pool = collections.defaultdict(list)
        for nm in names:
            pool[bin_of(sizes[nm])].append(nm)
        rng = random.Random(args.seed)
        chosen, short = [], 0
        for b, k in sorted(want.items()):
            cand = pool.get(b, [])
            rng.shuffle(cand)
            chosen += cand[:k]
            short += max(0, k - len(cand))
        names = sorted(chosen)
        print(f"matched {len(names)} designs to the reference histogram; short by {short}")
    if args.dev_frac:
        rng = random.Random(args.seed + 1)
        dev = set(rng.sample(names, int(round(args.dev_frac * len(names)))))
        dev_recs = {}
        for nm in sorted(dev):
            with np.load(Path(args.src) / f"{nm}.npz") as z:
                dev_recs[nm] = design_record(nm, {k: z[k] for k in z.files})
        torch.save(dev_recs, args.dev_out)
        names = [nm for nm in names if nm not in dev]
        print(f"held out {len(dev_recs)} designs -> {args.dev_out}")
    recs, ops, types_all = {}, collections.Counter(), collections.Counter()
    for nm in names:
        with np.load(Path(args.src) / f"{nm}.npz") as z:
            p = {k: z[k] for k in z.files}
        r = design_record(nm, p)
        recs[nm] = r
        for t in r["types"].tolist():
            types_all[NAME.get(t, str(t))] += 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(recs, args.out)
    ns = [recs[n]["n"] for n in names]
    stats = {"designs": len(names), "nodes_mean": float(np.mean(ns)), "nodes_median": float(np.median(ns)),
             "nodes_max": int(max(ns)), "ge80": int(sum(n >= 80 for n in ns)),
             "size_hist": {str(k): v for k, v in sorted(collections.Counter(bin_of(n) for n in ns).items())},
             "types": dict(types_all.most_common())}
    if args.stats:
        Path(args.stats).write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("size_hist", "types")}))
    print("types:", dict(types_all.most_common(14)))


if __name__ == "__main__":
    main()

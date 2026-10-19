"""Rewire nonconstant operands and regenerate traces for graph controls."""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synth.interp import CONST, INPUT, REG, Graph
from synth.rtl2rec import simulate

W32 = 2 ** np.arange(31, -1, -1, dtype=np.int64)


def rewire_one(args):
    src, out, seed = args
    did = Path(src).stem
    try:
        with np.load(src) as z:
            x = z["x"].astype(np.int64); ei = z["edge_index"]; et = z["edge_type"]
            has = z["has_sim_res"].astype(bool); sim = z["sim_res"]
        types, widths = x[:, 1], x[:, 2]
        n = len(types)
        const = {i: int((sim[0, i, 0].astype(np.int64) * W32).sum()) for i in range(n) if types[i] == CONST}
        rng = np.random.default_rng(seed)
        order = [i for i in range(n) if types[i] in (INPUT, CONST)]
        rest = [i for i in range(n) if types[i] not in (INPUT, CONST)]
        rng.shuffle(rest)
        order += rest
        rank = {v: k for k, v in enumerate(order)}
        movable = [v for v in order if types[v] != CONST]
        new = []
        for (s, d), t in zip(ei.tolist(), et.tolist()):
            if types[s] == CONST:
                new.append((s, d, t)); continue
            if types[d] == REG:
                pool = movable
            else:
                pool = [v for v in movable if rank[v] < rank[d]]
            if not pool:
                pool = [v for v in order if types[v] == INPUT] or movable
            new.append((int(rng.choice(pool)), d, t))
        nei = np.asarray([[s, d] for s, d, _ in new], dtype=np.int64)
        net = np.asarray([t for _, _, t in new], dtype=np.int64)
        g = Graph(x, nei, net, const)
        if g.order is None:
            raise RuntimeError("combinational loop")
        bits = simulate(g, types, widths, has, const, sim.shape[0], sim.shape[2], rng)
        np.savez_compressed(Path(out) / f"rew_{did}.npz", x=x.astype(np.float32), edge_index=nei,
                            edge_type=net, sim_res=bits, has_sim_res=has.astype(np.int64))
        return {"id": did, "ok": True, "nodes": n}
    except Exception as exc:
        return {"id": did, "ok": False, "reason": str(exc)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", default="", help="only these design ids (one per line)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.src).glob("*.npz"))
    if args.names:
        keep = set(Path(args.names).read_text().split())
        files = [f for f in files if f.stem in keep]
    jobs = [(str(f), str(out), args.seed * 1_000_003 + k) for k, f in enumerate(files)]
    with mp.Pool(args.workers) as pool, open(out / "status.jsonl", "a") as fh:
        ok = 0
        for rec in pool.imap_unordered(rewire_one, jobs, chunksize=4):
            fh.write(json.dumps(rec) + "\n"); ok += rec["ok"]
    print(f"rewired {ok} of {len(jobs)}")


if __name__ == "__main__":
    main()

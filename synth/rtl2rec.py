"""Convert RTL through an external RTL2CDFG frontend and generate execution traces."""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synth.dotfmt import OP, to_graph
from synth.interp import CONST, INPUT, Graph

R2C = Path(os.environ.get("EXECGNN_RTL2CDFG", "external/RTL2CDFG"))
PY = sys.executable
SHIFT = np.arange(31, -1, -1, dtype=np.int64)
_workdir = None


_mode = "v"
_ys = {}


def _init_worker(mode="v", auto_top=False):


    global _workdir, _mode
    _workdir = Path(tempfile.mkdtemp(prefix="r2c_"))
    shutil.copytree(R2C, _workdir / "rtl2cdfg")
    v = _workdir / "rtl2cdfg" / "v2cdfg"
    _mode = mode
    plain = (v / "run_ast.ys").read_text()
    if auto_top:
        assert "\nhierarchy -check\n" in plain
        plain = plain.replace("\nhierarchy -check\n", "\nhierarchy -check -auto-top\n")
    _ys["v"] = plain
    _ys["sv"] = plain.replace("read_verilog ./test.v", "read_verilog -sv ./test.v")
    for d in ("tmp/yosys_output_data", "tmp/yosys_output_data_clean", "tmp/s_exp", "tmp/ast", "cdfg"):
        (v / d).mkdir(parents=True, exist_ok=True)
    (_workdir / "rtl2cdfg" / "log").mkdir(exist_ok=True)


def _convert_once(vfile, reader, timeout):
    v = _workdir / "rtl2cdfg" / "v2cdfg"
    name = Path(vfile).stem
    for f in (v / "cdfg").glob("*.dot"):
        f.unlink()


    for f in [v / "test_ast.v", v / "test.v"] + [p for d in ("tmp/yosys_output_data", "tmp/yosys_output_data_clean",
                                                             "tmp/s_exp", "tmp/ast") for p in (v / d).glob(f"{name}_*")]:
        f.unlink(missing_ok=True)
    ys = _ys[reader]
    top = Path(vfile).with_suffix(".top")
    if top.exists():
        ys = ys.replace("\nhierarchy -check -auto-top\n", f"\nhierarchy -check -top {top.read_text().strip()}\n")
    (v / "run_ast.ys").write_text(ys)
    src = v / "input" / f"{name}.v"
    src.parent.mkdir(exist_ok=True)
    shutil.copy(vfile, src)
    try:
        r = subprocess.run([PY, "cdfg_generator.py", str(src)], cwd=v, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError("rtl2cdfg timeout")
    if not (v / "test_ast.v").exists():
        raise RuntimeError("yosys failed")
    dot = v / "cdfg" / f"{name}_cdfg.dot"
    if not dot.exists() or dot.stat().st_size < 30:
        tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
        raise RuntimeError("rtl2cdfg failed: " + tail[0][:160])
    return dot


def convert(vfile, timeout=180):
    """Run RTL2CDFG on one file; returns (.dot path, reader) or raises RuntimeError(stage)."""
    if _mode != "auto":
        return _convert_once(vfile, _mode, timeout), _mode
    try:
        return _convert_once(vfile, "v", timeout), "v"
    except RuntimeError as e:
        if str(e) != "yosys failed":
            raise
    return _convert_once(vfile, "sv", timeout), "sv"


def simulate(g, types, widths, has, const, traces, cycles, rng):
    n = len(types)
    vals = np.zeros((traces, n, cycles), dtype=np.int64)
    for t in range(traces):
        vis = np.zeros((n, cycles), dtype=np.int64)
        for i in range(n):
            if types[i] == INPUT:
                w = max(1, min(int(widths[i]), 32))
                vis[i] = rng.integers(0, 1 << w, size=cycles, dtype=np.int64)
            elif types[i] == CONST:
                vis[i] = const[i] & 0xFFFFFFFF
        vals[t] = g.simulate(vis, cycles)
    vals[:, ~has, :] = 0
    bits = ((vals[..., None] >> SHIFT) & 1).astype(np.uint8)
    return bits


def process(args):
    vfile, out, traces, cycles, max_nodes, seed = args
    did = Path(vfile).stem
    rec = {"id": did}
    try:
        dot, reader = convert(vfile)
        rec["stage"] = "dot"; rec["reader"] = reader
        gd = to_graph(dot, max_nodes=max_nodes)
        rec["stage"] = "graph"
        x = np.asarray(gd["x"], dtype=np.int64)
        types, widths = x[:, 1], x[:, 2]
        rec["nodes"] = int(len(x)); rec["edges"] = len(gd["edge_index"])
        if (widths > 32).any():
            raise RuntimeError(f"width {int(widths.max())} > 32")
        if not (types != INPUT).any() or not (types == INPUT).any():
            raise RuntimeError("no inputs or no logic")
        g = Graph(x, gd["edge_index"], gd["edge_type"], gd["const"])
        if g.order is None:
            raise RuntimeError("combinational loop")
        has = np.asarray(gd["has"], dtype=bool)
        rng = np.random.default_rng(seed)
        bits = simulate(g, types, widths, has, gd["const"], traces, cycles, rng)
        rec["stage"] = "sim"
        np.savez_compressed(Path(out) / f"{did}.npz", x=x.astype(np.float32),
                            edge_index=np.asarray(gd["edge_index"], dtype=np.int64),
                            edge_type=np.asarray(gd["edge_type"], dtype=np.int64),
                            sim_res=bits, has_sim_res=has.astype(np.int64))
        rec.update(ok=True, regs=int((types == OP["Reg"]).sum()), conds=int((types == OP["Cond"]).sum()),
                   has_frac=round(float(has.mean()), 3))
    except Exception as exc:
        rec.update(ok=False, reason=str(exc)[:200])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory of .v files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--traces", type=int, default=10)
    ap.add_argument("--cycles", type=int, default=50)
    ap.add_argument("--max-nodes", type=int, default=600)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sv", action="store_true", help="parse sources as SystemVerilog")
    ap.add_argument("--reader", choices=["v", "sv", "auto"], default="",
                    help="Yosys reader: Verilog, SystemVerilog, or Verilog then SystemVerilog")
    ap.add_argument("--auto-top", action="store_true",
                    help="elaborate from the top module so hierarchical designs flatten to one module")
    args = ap.parse_args()
    if not (R2C / "v2cdfg/run_ast.ys").is_file() or not (R2C / "Stagira/stagira.exe").is_file():
        ap.error("Set EXECGNN_RTL2CDFG to a complete DynamicRTL RTL2CDFG directory (including Stagira)")
    mode = args.reader or ("sv" if args.sv else "v")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    done = set()
    st = out / "status.jsonl"
    if st.exists():
        done = {json.loads(l)["id"] for l in open(st)}
    files = sorted(p for p in Path(args.src).glob("*.v") if p.stem not in done)
    jobs = [(str(f), str(out), args.traces, args.cycles, args.max_nodes, args.seed * 1_000_003 + k)
            for k, f in enumerate(files)]
    t0 = time.time()
    ok = 0
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(mode, args.auto_top)) as pool, open(st, "a") as fh:
        for k, rec in enumerate(pool.imap_unordered(process, jobs, chunksize=1)):
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            ok += bool(rec.get("ok"))
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(jobs)} ok {ok} {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(jobs)} ok {ok} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Independently check Yosys lowering for designs accepted by the frontend."""
import argparse
import json
import os
import multiprocessing as mp
import shutil
import subprocess
import tempfile
from pathlib import Path

YS = Path(os.environ.get("EXECGNN_RTL2CDFG", "external/RTL2CDFG")) / "v2cdfg/run_ast.ys"


def check(job):
    src, sv = job
    d = Path(tempfile.mkdtemp(prefix="ysck_"))
    try:
        shutil.copy(src, d / "test.v")
        ys = YS.read_text()
        if sv:
            ys = ys.replace("read_verilog ./test.v", "read_verilog -sv ./test.v")
        (d / "run_ast.ys").write_text(ys)
        try:
            r = subprocess.run(["yosys", "-q", "run_ast.ys"], cwd=d, capture_output=True, text=True, timeout=120)
            ok = r.returncode == 0 and (d / "test_ast.v").exists()
            err = "" if ok else ((r.stderr or r.stdout).strip().splitlines() or [""])[-1][:160]
        except subprocess.TimeoutExpired:
            ok, err = False, "timeout"
        return {"id": Path(src).stem, "yosys_ok": ok, "err": err}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--sv", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    ok_ids = [json.loads(l)["id"] for l in open(Path(args.data) / "status.jsonl") if json.loads(l).get("ok")]
    jobs = [(str(Path(args.src) / f"{i}.v"), args.sv) for i in ok_ids]
    with mp.Pool(args.workers) as pool:
        res = pool.map(check, jobs, chunksize=4)
    (Path(args.data) / "yosys_check.jsonl").write_text("".join(json.dumps(r) + "\n" for r in res))
    bad = [r for r in res if not r["yosys_ok"]]
    print(f"{args.data}: marked ok {len(res)}, Yosys fails on {len(bad)} ({100 * len(bad) / max(1, len(res)):.1f}%)")
    for r in bad[:3]:
        print("   ", r["id"], r["err"])


if __name__ == "__main__":
    main()

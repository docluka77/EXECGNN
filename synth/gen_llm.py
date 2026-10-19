"""Generate RTL modules through a chat-completions-compatible model endpoint."""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import threading
import time
import urllib.request
import zlib
from pathlib import Path

W = [4, 6, 8, 10, 12, 16, 20, 24, 32]
N = [2, 3, 4, 5, 6, 8]
TAXONOMY = [

    "a {n}-state Moore FSM that detects the serial bit pattern {pat} (overlapping) and raises a match flag",
    "a traffic-light controller FSM with {n} phases and {w}-bit programmable phase timers",
    "a vending-machine controller accepting coins worth {a} and {b} units, with a {w}-bit credit register and change output",
    "a UART transmitter with {w}-bit data, start/stop bits, a busy flag and a baud-rate divider counter",
    "a UART receiver with a 16x oversampling counter, a {w}-bit shift register and a data-valid pulse",
    "an SPI master engine that shifts {w}-bit frames with a configurable clock divider and a done pulse",
    "an I2C-like byte-level controller FSM with start, address, ack, data and stop phases",
    "a round-robin arbiter for {n} requesters with a grant-hold input",
    "a fixed-priority interrupt controller with {n} lines, mask and pending registers and an acknowledge input",
    "a button debouncer with a {w}-bit stability counter and rising/falling edge pulses",
    "an edge detector and pulse stretcher for {n} input bits with a {w}-bit stretch counter",
    "a bus address decoder with {n} regions, per-region enables and a response multiplexer",

    "a {w}-bit up/down counter with synchronous load, enable and a terminal-count flag",
    "a programmable timer with a {w}-bit prescaler, compare register and compare-match interrupt flag",
    "a PWM generator with {w}-bit period and duty registers and a center-aligned mode",
    "a {w}-bit Gray-code counter with a binary-converted output",
    "a chain of {n} BCD digit counters with carry and a clear input",
    "a watchdog timer with a kick input, a {w}-bit timeout register and a timeout flag",
    "a clock-enable frequency divider by a {w}-bit programmable ratio with a tick output",

    "a {w}-bit ALU with {n} operations (add, sub, and, or, xor, shift, compare) selected by an opcode, with zero and carry flags",
    "a {w}-bit saturating adder/subtractor with overflow and underflow flags and a registered output",
    "a {w}-bit barrel shifter supporting logical and arithmetic shifts left and right",
    "a leading-zero counter for a {w}-bit input with a registered, valid-qualified result",
    "a comparator that tracks the maximum of a stream of {w}-bit values and its index, with a reset",
    "a multiply-accumulate unit with {w}-bit operands, a wider accumulator, clear and enable",
    "an iterative shift-and-add multiplier for {w}-bit operands with start/busy/done handshake",
    "a restoring divider for {w}-bit operands with start/busy/done handshake and remainder output",
    "a CRC-{c} calculator over an {w}-bit parallel input with init, enable and a valid output",
    "a {w}-bit LFSR pseudo-random generator with seed load, enable and a period counter",
    "a Hamming(7,4)-style encoder and single-error-correcting checker for {w} data bits split into nibbles",
    "an absolute-difference and clamp unit on {w}-bit inputs with programmable upper and lower bounds",
    "a fixed-point {w}-bit multiplier with rounding, saturation and a registered output",

    "a FIFO controller (read/write pointers, count, full/empty/almost-full flags) for depth {n}, control logic only",
    "a {n}-stage pipeline of {w}-bit registers with per-stage valid bits and a global stall input",
    "a valid/ready pipeline register slice with a one-entry skid buffer for {w}-bit data",
    "a credit-based flow-control counter with {w}-bit credits, grant and return inputs",
    "a register file of {n} {w}-bit registers built from individual registers with one write port and one read port",

    "a weight-stationary systolic-array processing element: {w}-bit weight register, MAC, and pass-through of activations and partial sums",
    "a ReLU / leaky-ReLU / clamp activation unit on {w}-bit fixed-point data with a configurable threshold",
    "an INT8 requantization unit: multiply a {w}-bit accumulator by a scale, shift, round-half-up and saturate to 8 bits",
    "a max-pooling unit over a sliding window of {n} streamed {w}-bit values with a window-done flag",
    "a dot-product accumulator over {n} streamed {w}-bit element pairs with a done flag and clear",
    "a bias-add and requantization pipeline stage with {w}-bit inputs and saturation",
    "an address generator for a tiled loop nest with row/column counters, strides and a last-tile flag",
    "an argmax tracker over a stream of {w}-bit scores with a valid-qualified index output",
    "a zero-skipping MAC controller that skips zero activations and counts skipped operations",
    "an average-pooling accumulator over {n} inputs with divide-by-shift and rounding",

    "a tiny accumulator CPU datapath with {n} instructions (load immediate, add, sub, and, jump-if-zero) and a program counter",
    "a 2-bit saturating-counter branch predictor with {n} counters stored in individual registers",
    "a load/store unit address calculator with alignment check and byte-enable generation",
    "a simple cache hit/miss tag comparator with {n} ways implemented as individual registers and an LRU bit",
]
LINES = [30, 60, 100, 160, 240]
SYSTEM = "You are a senior RTL design engineer. You write clean, synthesizable Verilog-2001."
USER = """Write a synthesizable Verilog-2001 module that implements: {spec}.
Constraints:
- Exactly one top-level module named `top` (small helper modules are allowed; they will be flattened).
- One clock input named `clk`, used only as `always @(posedge clk)`, and a synchronous active-high reset input named `rst`.
- Every signal is at most 32 bits wide. No arrays or memories, no `initial` blocks, no `$` system tasks, no `define or `include, no tri-state, no latches.
- Make it realistic: the kind of block found in real designs, mixing control and datapath logic.
- Target size: roughly {lines} lines of code.
Output only the Verilog code in a single ```verilog code block."""


def spec_from_taxonomy(rng):
    t = rng.choice(TAXONOMY)
    return t.format(w=rng.choice(W), n=rng.choice(N), a=rng.choice([1, 2, 5]), b=rng.choice([10, 20, 25]),
                    c=rng.choice([5, 8, 16]), pat="".join(rng.choice("01") for _ in range(rng.choice([3, 4, 5]))))


def spec_composed(rng):
    """A larger subsystem: 2-4 taxonomy blocks wired together under a control FSM."""
    k = rng.choice([2, 3, 3, 4])
    parts = "; ".join(f"({i + 1}) {spec_from_taxonomy(rng)}" for i in range(k))
    return ("a subsystem that integrates the following blocks with realistic interconnect, a top-level "
            f"control FSM that sequences them, and status outputs: {parts}")


API = {"key_env": "", "template_kwargs": True, "max_tokens": 6000}


def ask(url, model, spec, lines, temperature, seed):
    body = {"model": model, "temperature": temperature, "top_p": 0.95, "max_tokens": API["max_tokens"],
            "seed": seed,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": USER.format(spec=spec, lines=lines)}]}
    if API["template_kwargs"]:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if API.get("thinking"):
        body["thinking"] = {"type": API["thinking"]}
    headers = {"Content-Type": "application/json"}
    if API["key_env"]:
        headers["Authorization"] = "Bearer " + os.environ[API["key_env"]]
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def extract(text):
    m = re.search(r"```(?:verilog|systemverilog|v)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--briefs", default="", help="JSON list of {spec, lines} (error-directed round)")
    ap.add_argument("--prefix", default="llm")
    ap.add_argument("--composed-frac", type=float, default=0.0,
                    help="fraction of taxonomy jobs that ask for a composed 2-4 block subsystem")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--api-key-env", default="", help="environment variable holding a bearer key (hosted APIs)")
    ap.add_argument("--no-template-kwargs", action="store_true", help="hosted APIs reject chat_template_kwargs")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--thinking", default="", choices=["", "enabled", "disabled"], help="DeepSeek API thinking switch")
    ap.add_argument("--model", default="qwen3-30b-a3b")
    args = ap.parse_args()
    API.update(key_env=args.api_key_env, template_kwargs=not args.no_template_kwargs, max_tokens=args.max_tokens,
               thinking=args.thinking)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    jobs = []
    briefs = json.load(open(args.briefs)) if args.briefs else None
    for k in range(args.start, args.start + args.n):
        if briefs:
            b = briefs[k % len(briefs)]
            jobs.append((f"{args.prefix}_{k:05d}", b["spec"], b.get("lines", rng.choice(LINES)), b.get("tag", "brief")))
        elif rng.random() < args.composed_frac:
            jobs.append((f"{args.prefix}_{k:05d}", spec_composed(rng), rng.choice([200, 280, 360]), "composed"))
        else:
            jobs.append((f"{args.prefix}_{k:05d}", spec_from_taxonomy(rng), rng.choice(LINES), "taxonomy"))
    lock = threading.Lock()
    meta = open(out / "meta.jsonl", "a")
    t0 = time.time()
    done = [0, 0]

    def work(job):
        did, spec, lines, tag = job
        if (out / f"{did}.v").exists():
            return
        try:
            r = ask(args.url, args.model, spec, lines, args.temperature, zlib.crc32(did.encode()))
            text = r["choices"][0]["message"]["content"]
            code = extract(text)
            (out / f"{did}.v").write_text(code)
            rec = {"id": did, "spec": spec, "lines_target": lines, "tag": tag,
                   "finish": r["choices"][0].get("finish_reason"), "usage": r.get("usage"),
                   "code_lines": code.count("\n")}
            ok = 1
        except Exception as exc:
            rec = {"id": did, "spec": spec, "error": repr(exc)[:300]}
            ok = 0
        with lock:
            meta.write(json.dumps(rec) + "\n"); meta.flush()
            done[0] += 1; done[1] += ok
            if done[0] % 50 == 0:
                print(f"{done[0]}/{len(jobs)} ok {done[1]} {time.time() - t0:.0f}s", flush=True)

    with cf.ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, jobs))
    print(f"done {done[0]} ok {done[1]} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

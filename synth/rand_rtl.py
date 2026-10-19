import argparse
import json
import random
from pathlib import Path

W = [1, 1, 2, 4, 6, 8, 8, 12, 16, 16, 24, 32]
BINOPS = {"Add": "+", "Sub": "-", "Mul": "*", "BitAnd": "&", "BitOr": "|", "BitXor": "^",
          "ShiftLeft": "<<", "ShiftRight": ">>", "Lt": "<", "Le": "<=", "Gt": ">", "Ge": ">=",
          "Eq": "==", "Neq": "!=", "And": "&&", "Or": "||", "Div": "/", "Mod": "%"}
UNOPS = {"Not": "~", "LNot": "!", "URand": "&", "URor": "|", "URxor": "^"}
DEFAULT_WEIGHTS = {"Cond": 30, "Add": 8, "Sub": 4, "Mul": 1, "BitAnd": 6, "BitOr": 4, "BitXor": 2,
                   "ShiftLeft": 1, "ShiftRight": 1, "Lt": 3, "Le": 1, "Gt": 2, "Ge": 1, "Eq": 8,
                   "Neq": 2, "And": 6, "Or": 3, "Not": 3, "LNot": 3, "URand": 1, "URor": 1, "URxor": 1,
                   "Concat": 3, "PartSelect": 6, "Div": 0.2, "Mod": 0.2}


class Gen:


    COMPARE = ("Lt", "Le", "Gt", "Ge", "Eq", "Neq")
    LOGIC = ("And", "Or")
    ONEBIT = ("LNot", "URand", "URor", "URxor")

    def __init__(self, rng, weights, p_const=0.06, p_select=0.03):
        self.rng = rng
        self.ops = [k for k in weights if k in BINOPS or k in UNOPS or k in ("Cond", "Concat", "PartSelect")]
        self.w = [weights[k] for k in self.ops]
        self.signals = []
        self.p_const, self.p_select = p_const, p_select

    def const(self, w):
        w = max(1, min(w, 32))
        return f"{w}'d{self.rng.randrange(0, min(1 << w, 1 << 16))}"

    def leaf(self, w):
        """A signal of exactly width w: the whole signal, a part-select of a wider one, or a constant."""
        same = [s for s in self.signals if s[1] == w]
        wider = [s for s in self.signals if s[1] > w]
        r = self.rng.random()
        if r < self.p_const or not (same or wider):
            return self.const(w)
        if same and (not wider or r < 1 - self.p_select):
            return self.rng.choice(same)[0]
        name, sw = self.rng.choice(wider)
        lo = self.rng.randrange(0, sw - w + 1)
        return f"{name}[{lo + w - 1}:{lo}]" if w > 1 else f"{name}[{lo}]"

    def cond(self, depth):


        k = self.rng.choices(["Eq", "Neq", "Lt", "Gt", "And", "Or", "LNot", "URor", "sig"],
                             [9, 1, 1, 1, 2, 1, 9, 1, 3])[0]
        if k == "sig" or depth <= 0:
            return self.leaf(1)
        if k in ("And", "Or"):
            return f"({self.cond(depth - 1)} {BINOPS[k]} {self.cond(depth - 1)})"
        if k == "LNot":
            return f"(!{self.cond(depth - 1)})"
        if k == "URor":
            u = self.rng.choice([k2 for k2 in W if k2 > 1])
            return f"(|{self.expr(u, depth - 1)})"
        u = self.rng.choice(W)
        rhs = self.const(u) if k == "Eq" and self.rng.random() < 0.5 else self.expr(u, depth - 1)
        return f"({self.expr(u, depth - 1)} {BINOPS[k]} {rhs})"

    def expr(self, w, depth):
        if depth <= 0:
            return self.leaf(w)
        if w == 1 and self.rng.random() < 0.5:
            return self.cond(depth)
        op = self.rng.choices(self.ops, self.w)[0]
        if w > 1 and (op in self.COMPARE or op in self.LOGIC or op in self.ONEBIT):
            op = "Cond"
        if op == "Cond":
            return f"({self.cond(depth - 1)} ? {self.expr(w, depth - 1)} : {self.expr(w, depth - 1)})"
        if op == "Concat":
            if w < 2:
                return self.leaf(w)
            a = self.rng.randint(1, w - 1)
            return f"{{{self.expr(a, depth - 1)}, {self.expr(w - a, depth - 1)}}}"
        if op == "PartSelect":
            return self.leaf(w)
        if op in self.COMPARE:
            u = self.rng.choice(W)
            return f"({self.expr(u, depth - 1)} {BINOPS[op]} {self.expr(u, depth - 1)})"
        if op in self.LOGIC:
            return f"({self.expr(1, depth - 1)} {BINOPS[op]} {self.expr(1, depth - 1)})"
        if op == "LNot":
            return f"(!{self.expr(1, depth - 1)})"
        if op in ("URand", "URor", "URxor"):
            u = self.rng.choice([k for k in W if k > 1])
            return f"({UNOPS[op]}{self.expr(u, depth - 1)})"
        if op == "Not":
            return f"(~{self.expr(w, depth - 1)})"
        if op in ("ShiftLeft", "ShiftRight"):
            return f"({self.expr(w, depth - 1)} {BINOPS[op]} {self.expr(min(5, w), depth - 1)})"
        return f"({self.expr(w, depth - 1)} {BINOPS[op]} {self.expr(w, depth - 1)})"


WSMALL = [1, 1, 1, 4, 8, 8, 16, 32]


def design(rng, weights, n_stmt, p_case=0.45):


    g = Gen(rng, weights)
    lines = ["module top(input clk, input rst"]
    ins = []
    for i in range(rng.randint(1, 6)):
        w = rng.choice(WSMALL)
        ins.append((f"in{i}", w))
    regs = [(f"r{i}", rng.choice(WSMALL)) for i in range(rng.randint(max(1, n_stmt // 3), max(1, 2 * n_stmt // 3)))]
    outs = []
    for name, w in ins:
        lines[0] += f", input [{w - 1}:0] {name}"
    g.signals = list(ins)
    body = []
    for name, w in regs:
        body.append(f"  reg [{w - 1}:0] {name};")
    g.signals += regs
    n_wire = max(1, n_stmt - len(regs))
    for i in range(n_wire):
        w = rng.choice(WSMALL)
        body.append(f"  wire [{w - 1}:0] w{i} = {g.expr(w, rng.randint(1, 3))};")
        g.signals.append((f"w{i}", w))
    body.append("  always @(posedge clk) begin")
    body.append("    if (rst) begin")
    for name, w in regs:
        body.append(f"      {name} <= {w}'d0;")
    body.append("    end else begin")
    for name, w in regs:
        r = rng.random()
        if r < p_case:
            sel, sw = rng.choice(g.signals)
            labels = sorted({rng.randrange(0, 1 << min(sw, 8)) for _ in range(rng.randint(2, 4))})
            body.append(f"      case ({sel})")
            for lab in labels:
                body.append(f"        {sw}'d{lab}: {name} <= {g.expr(w, rng.randint(1, 2))};")
            body.append(f"        default: {name} <= {g.expr(w, rng.randint(1, 2))};")
            body.append("      endcase")
        elif r < p_case + 0.35:
            body.append(f"      if ({g.cond(rng.randint(1, 2))}) {name} <= {g.expr(w, rng.randint(1, 2))};")
            for _ in range(rng.choice([0, 0, 1, 2])):
                body.append(f"      else if ({g.cond(rng.randint(1, 2))}) {name} <= {g.expr(w, rng.randint(1, 2))};")
            if rng.random() < 0.6:
                body.append(f"      else {name} <= {g.expr(w, rng.randint(1, 2))};")
        else:
            body.append(f"      {name} <= {g.expr(w, rng.randint(1, 3))};")
    body.append("    end")
    body.append("  end")
    for k, (name, w) in enumerate(rng.sample(g.signals[len(ins):], min(len(g.signals) - len(ins), rng.randint(1, 4)))):
        outs.append((f"out{k}", w))
        body.append(f"  assign out{k} = {name};")
    for name, w in outs:
        lines[0] += f", output [{w - 1}:0] {name}"
    lines[0] += ");"
    return "\n".join(lines + body + ["endmodule", ""])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--op-weights", default="")
    ap.add_argument("--prefix", default="rnd")
    ap.add_argument("--p-case", type=float, default=0.45)
    args = ap.parse_args()
    weights = json.load(open(args.op_weights)) if args.op_weights else DEFAULT_WEIGHTS
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for k in range(args.n):
        n_stmt = int(rng.choice([3, 5, 8, 12, 18, 26, 36, 50, 70]) * rng.uniform(0.7, 1.3))
        (out / f"{args.prefix}_{k:05d}.v").write_text(design(rng, weights, n_stmt, args.p_case))
    print(f"wrote {args.n} designs to {out}")


if __name__ == "__main__":
    main()

"""Width-aware CDFG execution with value and unknown masks."""
import json

import numpy as np

from execglm.operators import OP
NAME = {}
for n, i in OP.items():
    NAME.setdefault(i, n)
NAME[33] = "Concat"
INPUT, CONST, REG, COND, OUTPUT = OP["Input"], OP["Const"], OP["Reg"], OP["Cond"], OP["Output"]
M32 = (1 << 32) - 1


def m(w):
    return (1 << w) - 1 if w > 0 else M32


def truthy(v, x):
    if v & ~x:
        return 1
    if x == 0:
        return 0
    return None


def ev(op, ops, widths, w):
    """Evaluate an operator using value and unknown masks."""
    mm = m(w)
    if not ops:
        return 0, 0
    a, ax = ops[0]
    b, bx = ops[1] if len(ops) > 1 else (0, 0)
    anyx = any(x for _, x in ops)
    if op in ("Wire", "Output"):
        return a & mm, ax & mm
    if op == "Cond":
        c, cx = ops[2] if len(ops) > 2 else (0, 0)
        s = truthy(a, ax)
        if s == 1:
            return b & mm, bx & mm
        if s == 0:
            return c & mm, cx & mm
        diff = (b ^ c) | bx | cx
        return b & ~diff & mm, diff & mm
    if op in ("Lt", "Le", "Gt", "Ge"):
        if anyx:
            return 0, 1
        return int({"Lt": a < b, "Le": a <= b, "Gt": a > b, "Ge": a >= b}[op]), 0
    if op in ("Eq", "Neq"):
        if (a ^ b) & ~(ax | bx):
            return (0 if op == "Eq" else 1), 0
        if ax | bx:
            return 0, 1
        return (1 if op == "Eq" else 0), 0
    if op in ("Eeq", "Neeq"):
        same = (a & ~ax) == (b & ~bx) and ax == bx
        return int(same if op == "Eeq" else not same), 0
    if op in ("And", "Or"):
        ta, tb = truthy(a, ax), truthy(b, bx)
        if op == "And":
            if ta == 0 or tb == 0:
                return 0, 0
            if ta == 1 and tb == 1:
                return 1, 0
            return 0, 1
        if ta == 1 or tb == 1:
            return 1, 0
        if ta == 0 and tb == 0:
            return 0, 0
        return 0, 1
    if op == "LNot":
        t = truthy(a, ax)
        return (0, 1) if t is None else (1 - t, 0)
    if op == "Not":
        return ~a & ~ax & mm, ax & mm
    if op in ("Add", "Sub", "Mul", "Div", "Mod"):
        if anyx:
            return 0, mm
        if op == "Add":
            return sum(v for v, _ in ops) & mm, 0
        if op == "Sub":
            return (a - b) & mm, 0
        if op == "Mul":
            out = 1
            for v, _ in ops:
                out *= v
            return out & mm, 0
        if b == 0:
            return 0, mm
        return ((a // b) if op == "Div" else (a % b)) & mm, 0
    if op in ("ShiftLeft", "AshiftLeft", "ShiftRight", "AshiftRight"):
        if bx:
            return 0, mm
        if b >= 256:
            return 0, 0
        if op in ("ShiftLeft", "AshiftLeft"):
            return (a << b) & mm, (ax << b) & mm
        return (a >> b) & mm, (ax >> b) & mm
    if op in ("BitAnd", "BitOr", "BitXor"):
        if op == "BitXor":
            xs = 0; v = 0
            for vv, xx in ops:
                xs |= xx; v ^= vv
            return v & ~xs & mm, xs & mm
        known1 = [vv & ~xx for vv, xx in ops]
        known0 = [~vv & ~xx & mm for vv, xx in ops]
        if op == "BitAnd":
            one = mm
            for k in known1:
                one &= k
            zero = 0
            for k in known0:
                zero |= k
        else:
            one = 0
            for k in known1:
                one |= k
            zero = mm
            for k in known0:
                zero &= k
        return one & mm, mm & ~one & ~zero
    if op in ("URand", "URnand", "URor", "URnor", "URxor"):
        wa = widths[0] if widths and widths[0] > 0 else 32
        ma = m(wa)
        v, x = a & ma, ax & ma
        if op in ("URand", "URnand"):
            r = (0, 0) if (~v & ~x & ma) else ((0, 1) if x else (1, 0))
        elif op in ("URor", "URnor"):
            r = (1, 0) if (v & ~x) else ((0, 1) if x else (0, 0))
        else:
            r = (0, 1) if x else (bin(v).count("1") & 1, 0)
        if op in ("URnand", "URnor") and r[1] == 0:
            r = (1 - r[0], 0)
        return r
    if op == "Concat":
        v = x = 0
        for (vv, xx), wv in zip(ops, widths):
            mv = m(wv)
            v = (v << wv) | (vv & mv); x = (x << wv) | (xx & mv)
        return v & mm, x & mm
    if op == "PartSelect":
        hi, hx = ops[1]
        lo, lx = ops[2] if len(ops) > 2 else ops[1]
        if hx or lx:
            return 0, mm
        if hi < lo:
            hi, lo = lo, hi
        mk = m(hi - lo + 1)
        return (a >> lo) & mk, (ax >> lo) & mk
    raise KeyError(op)


class Graph:
    """The benchmark's graph semantics on (types, widths, operand-ordered fanins, consts)."""

    def __init__(self, x, edge_index, edge_type, const, init_zero=True):
        self.init_zero = init_zero
        x = np.asarray(x)
        self.types = x[:, 1].astype(int)
        self.width = x[:, 2].astype(int)
        self.n = n = len(self.types)
        ins = [[] for _ in range(n)]
        for (s, d), t in zip(edge_index, edge_type):
            ins[int(d)].append((int(t), int(s)))
        self.ins = [[s for _, s in sorted(v)] for v in ins]
        self.const = const
        self.regout = {i for i in range(n) if self.types[i] == OUTPUT and (
            self._reads_back(i) or (len(self.ins[i]) >= 2 and any(self.types[s] == CONST for s in self.ins[i])))}
        self.order = self._order()
        self.eff = self._effective_widths()

    def _reads_back(self, out):
        seen = set(); stack = list(self.ins[out])
        while stack:
            v = stack.pop()
            if v == out:
                return True
            if v in seen or self.types[v] in (INPUT, CONST, REG):
                continue
            seen.add(v); stack.extend(self.ins[v])
        return False

    def sequential(self, i):
        return self.types[i] == REG or i in self.regout

    def _order(self):
        indeg = [0] * self.n; outs = [[] for _ in range(self.n)]
        for d in range(self.n):
            if self.types[d] in (INPUT, CONST) or self.sequential(d):
                continue
            for s in self.ins[d]:
                indeg[d] += 1; outs[s].append(d)
        q = [i for i in range(self.n) if indeg[i] == 0]; order = []
        while q:
            i = q.pop(); order.append(i)
            for d in outs[i]:
                indeg[d] -= 1
                if indeg[d] == 0:
                    q.append(d)
        return order if len(order) == self.n else None

    def _effective_widths(self):
        eff = [int(w) for w in self.width]
        order = self.order or list(range(self.n))
        for _ in range(3):
            for i in order:
                if eff[i] > 0:
                    continue
                op = NAME.get(int(self.types[i]))
                src = self.ins[i]
                if op == "PartSelect" and len(src) >= 3 and src[1] in self.const and src[2] in self.const:
                    eff[i] = abs(self.const[src[1]] - self.const[src[2]]) + 1
                elif op == "Concat":
                    eff[i] = sum(eff[s] for s in src)
                elif op in ("Lt", "Le", "Gt", "Ge", "Eq", "Neq", "Eeq", "Neeq", "And", "Or",
                            "LNot", "URxor", "URand", "URor", "URnand", "URnor"):
                    eff[i] = 1
                elif src:
                    eff[i] = max(eff[s] for s in src)
        return eff

    def seq_parts(self, i):
        consts = [s for s in self.ins[i] if self.types[s] == CONST]
        ds = [s for s in self.ins[i] if self.types[s] != CONST]
        dsrc = ds[0] if ds else (consts[0] if consts else None)
        init = consts[0] if (consts and ds) else None
        return dsrc, init

    def simulate(self, vis, cycles):


        val = [[(0, 0)] * cycles for _ in range(self.n)]
        for c in range(cycles):
            for i in self.order:
                ty = int(self.types[i])
                if ty in (INPUT, CONST):
                    val[i][c] = (int(vis[i][c]), 0)
                    continue
                w = self.eff[i]
                if self.sequential(i):
                    dsrc, init = self.seq_parts(i)
                    if c == 0:
                        if init is not None:
                            val[i][c] = (int(vis[init][0]) & m(w), 0)
                        else:
                            val[i][c] = (0, 0) if self.init_zero else (0, m(w))
                    else:
                        v, x = val[dsrc][c - 1] if dsrc is not None else (0, m(w))
                        val[i][c] = (v & m(w), x & m(w))
                    continue
                val[i][c] = ev(NAME.get(ty, "?"), [val[s][c] for s in self.ins[i]],
                               [self.eff[s] for s in self.ins[i]], w)
        out = np.zeros((self.n, cycles), dtype=np.int64)
        for i in range(self.n):
            for c in range(cycles):
                v, x = val[i][c]
                out[i, c] = (v & ~x) & M32
        return out

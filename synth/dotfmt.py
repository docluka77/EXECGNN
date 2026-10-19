"""Read DynamicRTL-format DOT graphs and merge redundant wire nodes."""
import json
import re
from collections import defaultdict, deque

from execglm.operators import OP
NODE_RE = re.compile(r'^\s*"([^"]+)"\s*\[')
EDGE_RE = re.compile(r'^\s*"([^"]+)"\s*->\s*"([^"]+)"\s*\[label="([^"]*)"\]')
RECORDED = {"Input", "Output", "Reg", "Wire", "Const"}


def read_dot(path):
    nodes, edges = [], []
    seen = set()
    for line in open(path):
        m = EDGE_RE.match(line)
        if m:
            u, v, lab = m.groups()
            edges.append((u, v, int("".join(filter(str.isdigit, lab)))))
            for x in (u, v):
                if x not in seen:
                    seen.add(x); nodes.append(x)
            continue
        m = NODE_RE.match(line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1)); nodes.append(m.group(1))
    return nodes, edges


def name2idx_bfs(nodes, edges):
    """get_name2idx: inputs then constants (each sorted by name) seed a BFS over fanouts."""
    fan = defaultdict(list)
    for u, v, _ in set(edges):
        fan[u].append(v)
    inputs = sorted([n for n in nodes if n.split(",")[0] == "Input"], key=lambda x: x.split(",")[-1])
    consts = sorted([n for n in nodes if n.split(",")[0] == "Const"], key=lambda x: x.split(",")[-1])
    idx, q = {}, deque()
    for n in inputs + consts:
        idx[n] = len(idx); q.append(n)
    while q:
        for _ in range(len(q)):
            n = q.popleft()
            for f in fan[n]:
                if f not in idx:
                    idx[f] = len(idx); q.append(f)
    return idx


def const_value(conststr):
    """Parse a Verilog constant, mapping unknown digits to zero."""
    if "'" not in conststr:
        return int(conststr)
    base = conststr[conststr.find("'") + 1].lower()
    body = conststr[conststr.find("'") + 2:].lower().replace("x", "0").replace("z", "0").replace("_", "")
    return int(body, {"h": 16, "b": 2, "o": 8, "d": 10}[base])


def to_graph(path, max_nodes=0):
    """Returns dict(x, edge_index, edge_type, names, has, const) or raises ValueError."""
    nodes, edges = read_dot(path)
    if not nodes:
        raise ValueError("empty graph")
    idx = name2idx_bfs(nodes, edges)
    if len(idx) != len(nodes):
        raise ValueError(f"{len(nodes) - len(idx)} nodes unreachable from inputs/constants")
    order = sorted(nodes, key=idx.get)
    x = []
    for i, n in enumerate(order):
        ty, w = n.split(",")[0], n.split(",")[1]
        if ty not in OP:
            raise ValueError(f"unknown node type {ty}")
        x.append([i, OP[ty], 0 if w == "Null" else int(w)])
    ei = [[idx[u], idx[v]] for u, v, _ in edges]
    et = [t for _, _, t in edges]
    names = [n.split(",")[-1] if n.split(",")[0] != "Const" else n.split(",", 2)[-1] for n in order]

    rec = [n.split(",")[0] in RECORDED for n in order]
    const = {}
    for i, n in enumerate(order):
        if n.split(",")[0] == "Const":
            const[i] = const_value(n.split(",", 2)[-1].split("_")[-1])


    wire = OP["Wire"]
    while True:
        fanin = [[] for _ in range(len(x))]
        for s, d in ei:
            fanin[d].append(s)
        rm = next((i for i in range(len(x)) if x[i][1] == wire), None)
        if rm is None:
            break
        if len(fanin[rm]) != 1:
            raise ValueError(f"wire with {len(fanin[rm])} fanins")
        keep = fanin[rm][0]
        rec[keep] = rec[rm]
        x[keep][2] = x[rm][2]
        nei, net = [], []
        for (s, d), t in zip(ei, et):
            if s == rm:
                nei.append([keep, d]); net.append(t)
            elif d == rm:
                continue
            else:
                nei.append([s, d]); net.append(t)
        ei, et = nei, net
        x.pop(rm); rec.pop(rm); names.pop(rm)
        const = {(k - 1 if k > rm else k): v for k, v in const.items() if k != rm}
        ei = [[s - (s > rm), d - (d > rm)] for s, d in ei]
        for i in range(len(x)):
            x[i][0] = i
    if max_nodes and len(x) > max_nodes:
        raise ValueError(f"{len(x)} nodes > {max_nodes}")
    return {"x": x, "edge_index": ei, "edge_type": et, "names": names, "has": rec, "const": const}

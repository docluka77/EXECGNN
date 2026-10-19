"""Create four small independent CDFGs and execution targets for CPU smoke checks."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from execglm.operators import OP
from execglm.prep import design_record
from synth.interp import Graph

def record(name,delta):

    types=[OP[k] for k in ["Input","Const","Const","Reg","Add","Cond","Output"]]
    widths=[1,8,8,8,8,8,8]
    x=np.array([[i,t,w] for i,(t,w) in enumerate(zip(types,widths))])
    edges=np.array([[3,4],[2,4],[0,5],[4,5],[3,5],[5,3],[1,3],[3,6]])
    positions=np.array([1,2,1,2,3,1,2,1])
    graph=Graph(x,edges,positions,{1:0,2:delta})
    assert graph.order is not None
    vals=[]
    for trace in range(2):
        vis=np.zeros((len(x),50),dtype=np.int64)
        vis[0]=[(t+trace)%2 for t in range(50)];vis[2]=delta
        values=graph.simulate(vis,50)
        expected=0
        for t in range(50):
            assert values[3,t]==expected
            if vis[0,t]:expected=(expected+delta)&255
        vals.append(values)
    bits=((np.array(vals)[...,None]>>np.arange(31,-1,-1))&1).astype(np.uint8)
    return design_record(name,dict(x=x,edge_index=edges,edge_type=positions,sim_res=bits,has_sim_res=np.ones(len(x))))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="data/tiny");a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cache={f"tiny_{d}":record(f"tiny_{d}",d) for d in range(1,5)}
    torch.save(cache,out/"cache.pt")
    (out/"split.json").write_text(json.dumps(dict(train=["tiny_1","tiny_2"],validation=["tiny_3","tiny_4"]),indent=2))
    print("Created 4 designs, 2 traces/design, 50 cycles; independent counter checks passed.")
if __name__=="__main__":main()

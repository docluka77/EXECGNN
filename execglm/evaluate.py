"""Evaluate a saved training run using its recorded configuration and split."""
import argparse,json
from pathlib import Path
import torch
from execglm.data import LLMCache,split_items,official_toggle_targets,novel_designs
from execglm.model import ExecCore
from execglm.train import evaluate

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",required=True);ap.add_argument("--cache",default="")
    ap.add_argument("--split-file",default="");ap.add_argument("--device",default="cpu");ap.add_argument("--T",type=int,default=0)
    a=ap.parse_args();run=Path(a.run);cfg=json.loads((run/"provenance.json").read_text())["args"]
    torch.set_num_threads(2);device=torch.device(a.device)
    cache=torch.load(a.cache or cfg["cache"],map_location="cpu",weights_only=False)
    _,items=split_items(cache,cfg["split"],a.split_file or cfg.get("split_file") or None)
    if cfg.get("max_val"):items=items[:cfg["max_val"]]
    llm=LLMCache(cfg["llm_dir"]) if cfg["llm_dir"] else None
    model=ExecCore(d=cfg["d"],hidden=cfg["hidden"],slots=cfg["slots"],llm_dim=llm.dim if llm else 0,use_width=not cfg["no_width"],sync=cfg["sync"]).to(device)
    model.load_state_dict(torch.load(run/"model.pt",map_location=device,weights_only=True));model.eval()
    official=official_toggle_targets() if cfg["split"]=="official" else None
    novel=novel_designs() if cfg["split"]=="official" else None
    metrics,_=evaluate(model,items,cache,llm,a.T or cfg["T"],device,64,official,novel)
    print(json.dumps(metrics,indent=2))
if __name__=="__main__":main()

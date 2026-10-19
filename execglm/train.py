"""Train and evaluate the execution-aligned learner."""
import argparse
import json
import math
import random
import subprocess
import time
import hashlib
import shutil
from pathlib import Path

import numpy as np
import torch

from execglm.data import (LLMCache, collate, novel_designs, official_toggle_targets,
                          split_items, to_device)
from execglm.model import ExecCore, loss_fn, predictions, targets
from execglm.budget import resolve_budget

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


@torch.no_grad()
def evaluate(model, items, cache, llm, T, device, batch_size, official=None, novel=None, keep=False):
    model.eval()
    acc = {k: [0.0, 0] for k in ("tog_all", "tog_rec", "tog_novel", "br_all", "br_rec", "br_novel")}
    kept = {}
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        b = to_device(collate(chunk, cache, llm), device)
        _, nz, ch = model(b, T)
        p_tog, br_logit = predictions(nz, ch, b, T)
        _, _, _, tog_lab, br_lab = targets(b, T)
        label_nodes = b["role"] >= 2
        g_nodes = b["graph"]
        for g, (dn, tr, nm) in enumerate(chunk):
            m = label_nodes & (g_nodes == g)
            pt = p_tog[m]
            lab = official[nm].to(device) if (official is not None and T == 30) else tog_lab[m]
            err = (pt - lab).abs()
            rec = b["has"][m]
            acc["tog_all"][0] += err.sum().item(); acc["tog_all"][1] += err.numel()
            acc["tog_rec"][0] += err[rec].sum().item(); acc["tog_rec"][1] += int(rec.sum())
            sm = b["sel_graph"] == g
            bl, bp = br_lab[sm], (br_logit[sm] > 0).float()
            ok = (bl == bp).float()
            srec = b["has"][b["sel"][sm]]
            acc["br_all"][0] += ok.sum().item(); acc["br_all"][1] += ok.numel()
            acc["br_rec"][0] += ok[srec].sum().item(); acc["br_rec"][1] += int(srec.sum())
            if novel is not None and dn in novel:
                acc["tog_novel"][0] += err.sum().item(); acc["tog_novel"][1] += err.numel()
                acc["br_novel"][0] += ok.sum().item(); acc["br_novel"][1] += ok.numel()
            if keep:
                kept[nm] = {"toggle": pt.cpu(), "branch_logit": br_logit[sm].cpu()}
    model.train()
    res = {}
    for k, (s, c) in acc.items():
        if c:
            res[k] = (1 - s / c) if k.startswith("tog") else s / c
    return res, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="custom", choices=["custom", "official", "size"])
    ap.add_argument("--split-file", default="", help="design-disjoint split JSON")
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--T-eval-long", type=int, default=50)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-bits", type=float, default=1.0)
    ap.add_argument("--w-nz", type=float, default=1.0)
    ap.add_argument("--w-ch", type=float, default=1.0)
    ap.add_argument("--w-tog", type=float, default=1.0)
    ap.add_argument("--w-br", type=float, default=1.0)
    ap.add_argument("--no-hints", action="store_true", help="labels only (ablation)")
    ap.add_argument("--no-width", action="store_true", help="ignore declared bit widths (ablation)")
    ap.add_argument("--sync", action="store_true",
                    help="synchronous rounds (as many as levels) instead of the level sweep (ablation)")
    ap.add_argument("--slots", type=int, default=3, help="operand-position slots; 1 = order-blind sum")
    ap.add_argument("--extra-cache", default="",
                    help="comma-separated caches of synthetic designs added to TRAINING only (llmdata/cache/*.pt)")
    ap.add_argument("--extra-traces", type=int, default=10, help="traces used per synthetic design")
    ap.add_argument("--extra-max", type=int, default=0, help="cap on synthetic designs per cache (0 = all)")
    ap.add_argument("--real-frac", type=float, default=1.0,
                    help="keep this fraction of real training DESIGNS (0 = synthetic only)")
    ap.add_argument("--min-steps", type=int, default=0,
                    help="raise the epoch count until at least this many optimiser steps")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="exact optimizer updates; repeats training epochs as necessary, 0 keeps epoch mode")
    ap.add_argument("--llm-dir", default="", help="LACE node feature cache (pretrained or random control)")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--max-train", type=int, default=0, help="subsample training traces (smoke tests)")
    ap.add_argument("--max-val", type=int, default=0, help="smoke only; 0 evaluates the full split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.max_steps < 0 or args.min_steps < 0:
        ap.error('update limits must be nonnegative')
    if args.max_steps and args.min_steps:
        ap.error('--max-steps and --min-steps are mutually exclusive')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    prov = {"args": vars(args), "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "start": time.strftime("%F %T")}
    (out / "provenance.json").write_text(json.dumps(prov, indent=1))
    def file_sha(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
        return h.hexdigest()
    sources = out/'source'; sources.mkdir()
    prov['source_sha256'] = {}
    for name in ('model.py', 'data.py', 'train.py', 'budget.py'):
        src = Path(__file__).parent/name; shutil.copy2(src, sources/name)
        prov['source_sha256'][name] = file_sha(src)
    prov['cache_sha256'] = file_sha(args.cache)
    prov['publication_eligible'] = not bool(args.max_train or args.max_val)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    train_items, val_items = split_items(cache, args.split, args.split_file or None)
    if args.max_val: val_items = val_items[:args.max_val]
    n_real_initial = len(train_items)
    if args.max_train:
        random.shuffle(train_items)
        train_items = train_items[:args.max_train]
    if args.real_frac < 1.0:
        designs = sorted({d for d, _, _ in train_items})
        keep = set(random.Random(args.seed + 1000).sample(designs, int(round(args.real_frac * len(designs)))))
        train_items = [it for it in train_items if it[0] in keep]
        prov["real_designs"] = len(keep)
    real_names = {it[0] for it in train_items}
    n_real_selected = len(train_items)
    if args.extra_cache:
        if args.llm_dir:
            raise SystemExit("--extra-cache has no language vectors; drop --llm-dir")
        n_real = len(train_items)
        prov["extra"] = {}
        for path in args.extra_cache.split(","):
            extra = torch.load(path, map_location="cpu", weights_only=False)
            prov.setdefault('extra_cache_sha256', {})[path] = file_sha(path)
            names = sorted(extra)
            if args.extra_max:
                names = random.Random(args.seed).sample(names, min(args.extra_max, len(names)))
            for nm in names:
                assert nm not in cache, nm
                cache[nm] = extra[nm]
                k = min(args.extra_traces, extra[nm]["values"].shape[0])
                train_items += [(nm, t, f"{nm}_trace{t}") for t in range(k)]
            prov["extra"][path] = {"designs": len(names)}
        prov["extra"]["train_traces_real"] = n_real
        prov["extra"]["train_traces_total"] = len(train_items)
        (out / "provenance.json").write_text(json.dumps(prov, indent=1))
    llm = LLMCache(args.llm_dir) if args.llm_dir else None
    prov['train_membership_sha256'] = hashlib.sha256('\n'.join(sorted(it[2] for it in train_items)).encode()).hexdigest()
    prov['validation_membership_sha256'] = hashlib.sha256('\n'.join(sorted(it[2] for it in val_items)).encode()).hexdigest()
    official = official_toggle_targets() if args.split == "official" else None
    novel = novel_designs() if args.split == "official" else None
    model = ExecCore(d=args.d, hidden=args.hidden, slots=args.slots, llm_dim=llm.dim if llm else 0,
                     use_width=not args.no_width, sync=args.sync).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if not train_items:
        raise SystemExit("no training data")
    budget = resolve_budget(len(train_items), args.batch, args.epochs, args.min_steps, args.max_steps)
    epochs, steps = budget.epochs, budget.updates
    prov.update(epochs_effective=epochs, planned_updates=steps, batches_per_epoch=budget.batches_per_epoch,
                train_traces=len(train_items), validation_traces=len(val_items),
                real_traces_initial=n_real_initial, real_traces_selected=n_real_selected,
                torch_version=torch.__version__, cuda_version=torch.version.cuda)
    (out / "provenance.json").write_text(json.dumps(prov, indent=1))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    w = {"bits": args.w_bits, "nz": args.w_nz, "ch": args.w_ch, "tog": args.w_tog, "br": args.w_br}
    if args.no_hints:
        w.update(bits=0.0, nz=0.0, ch=0.0)
    print(f"train {len(train_items)} val {len(val_items)} params {n_params:,} llm {bool(llm)}", flush=True)
    log = open(out / "metrics.jsonl", "a")
    t0 = time.time()
    updates = scheduler_steps = node_cycles = real_exposures = synthetic_exposures = 0
    epoch_updates = []
    for epoch in range(epochs):
        random.shuffle(train_items)
        run = {}
        nb = 0
        for i in range(0, len(train_items), args.batch):
            if updates >= steps: break
            batch_items = train_items[i:i + args.batch]
            b = to_device(collate(batch_items, cache, llm), device)
            outp = model(b, args.T)
            loss, parts = loss_fn(outp, b, args.T, w)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'Nonfinite loss at update {updates}')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step(); sched.step()
            updates += 1; scheduler_steps += 1; nb += 1
            node_cycles += int(b['role'].numel()) * args.T
            real_exposures += sum(it[0] in real_names for it in batch_items)
            synthetic_exposures += sum(it[0] not in real_names for it in batch_items)
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + v
        epoch_updates.append(nb)
        rec = {"epoch": epoch, "updates": updates, "epoch_updates": nb,
               "time_s": round(time.time() - t0), **{f"train_{k}": v / nb for k, v in run.items()}}
        if (epoch + 1) % args.eval_every == 0 or epoch == epochs - 1 or updates == steps:
            res, _ = evaluate(model, val_items, cache, llm, args.T, device, 128, official, novel)
            rec.update({f"val_{k}": v for k, v in res.items()})
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
        if updates == steps: break
    if updates != steps or scheduler_steps != steps or sched.last_epoch != steps:
        raise RuntimeError('Actual optimizer/scheduler budget differs from plan')
    log.close()
    torch.save(model.state_dict(), out / "model.pt")
    final = {}
    res, kept = evaluate(model, val_items, cache, llm, args.T, device, 128, official, novel, keep=True)
    final.update({f"T{args.T}_{k}": v for k, v in res.items()})
    torch.save(kept, out / f"val_predictions_T{args.T}.pt")
    if args.T_eval_long:
        res_l, kept_l = evaluate(model, val_items, cache, llm, args.T_eval_long, device, 128, None, novel, keep=True)
        final.update({f"T{args.T_eval_long}_{k}": v for k, v in res_l.items()})
        torch.save(kept_l, out / f"val_predictions_T{args.T_eval_long}.pt")
    prov.update({"params": n_params, "final": final, "end": time.strftime("%F %T"),
                 "train_seconds": round(time.time() - t0), 'actual_updates': updates,
                 'scheduler_steps': scheduler_steps, 'scheduler_last_epoch': sched.last_epoch,
                 'epoch_updates': epoch_updates, 'node_cycles_seen': node_cycles,
                 'real_trace_exposures': real_exposures, 'synthetic_trace_exposures': synthetic_exposures})
    (out / "provenance.json").write_text(json.dumps(prov, indent=1))
    print("FINAL", json.dumps(final), flush=True)


if __name__ == "__main__":
    main()

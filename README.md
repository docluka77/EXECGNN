# EXECGNN

Execution-aligned graph learning for RTL, with interpreter-backed training data generation.

ExecGNN combines an execution-aligned graph learner with interpreter-backed RTL data generation. Registers carry latent states across cycles; combinational nodes update in dependency order with ordered operand slots. Intermediate value, nonzero and change heads provide execution supervision.

The Python module is named `execglm`.

## Components and attribution

| Component | Location | Origin |
|---|---|---|
| Execution-aligned learner and intermediate losses | `execglm/model.py` | ExecGNN |
| Batching, preprocessing, fixed-update training and evaluation | `execglm/` | ExecGNN; reads the DynamicRTL graph format |
| RTL authoring and supervision pipeline | `synth/gen_llm.py`, `synth/rtl2rec.py`, `synth/interp.py` | ExecGNN |
| Matched-size random/rewired controls and structural deduplication | `synth/` | ExecGNN; DOT conversion follows DynamicRTL conventions |
| RTL-to-CDFG frontend and released benchmark | External dependency | [DynamicRTL](https://github.com/magicYang1573/DynamicRTL) |

The RTL2CDFG frontend, DR-GNN baseline, upstream benchmark data, LLM weights and trained checkpoints are obtained separately. `THIRD_PARTY_NOTICES.md` records the dependency boundary.

## Installation

Python 3.11 is the validated interpreter version. Create an environment and install the core requirements:

```bash
python -m venv .venv
# Activate the environment using your shell's activation command.
python -m pip install -r requirements.txt
```

Choose a PyTorch wheel appropriate for your CPU/CUDA platform. The source was exercised with PyTorch 2.3.0+cu121 and NumPy 2.1.1; the tests below run on CPU. RTL conversion additionally needs Linux, Yosys, the DynamicRTL RTL2CDFG frontend including its executable Stagira parser, and the packages in `requirements-frontend.txt`. Some upstream version pins predate Python 3.11; provision the frontend environment according to the upstream tool's instructions.

## CPU quick start

Run commands from the repository root. This example creates four independent tiny circuits, checks their register trajectories against a separate counter calculation, runs four optimizer updates, saves a model and evaluates it again:

```bash
python -m unittest discover -s tests -v
python -m examples.make_tiny --out data/tiny
python -m execglm.train --cache data/tiny/cache.pt --split custom --split-file data/tiny/split.json --device cpu --d 16 --hidden 32 --T 6 --T-eval-long 0 --batch 2 --max-steps 4 --eval-every 1 --out runs/tiny
python -m execglm.evaluate --run runs/tiny --device cpu
```

Use a new output directory for each training run. The tiny example verifies execution and packaging; its scores are not research results.

## Training on circuit data

For a DynamicRTL-format `graphs*.npz` file:

```bash
python -m execglm.prep --graphs data/graphs-2000.npz --out data/real-cache.pt
python -m execglm.train --cache data/real-cache.pt --split custom --split-file data/split.json --device cuda:0 --T 30 --T-eval-long 50 --batch 64 --max-steps 11430 --out runs/real-s0 --seed 0
```

The split JSON can contain design names:

```json
{"train": ["training_design"], "validation": ["held_out_design"]}
```

It may instead use `train_trace_names` and `validation_trace_names`, with names such as `design_trace0`. Train and validation designs must be disjoint. Preserve the original published split and evaluation artifacts when reproducing paper scores; a fresh split defines a different experiment.

For the original evaluation convention, use `--split official --split-file data/official/split.json` and set `EXECGNN_OFFICIAL_DIR` to the directory containing `split.json` and `pretrain/validation/predictions.jsonl`; set `EXECGNN_STRUCTURAL_FILE` to the original structural-partition JSON. Those artifacts determine released toggle targets and the novel-design subset. For custom splits, integer execution values in the cache define the targets directly.

Training writes `model.pt`, `metrics.jsonl`, predictions and `provenance.json`. The provenance records source and data fingerprints, split membership and actual optimizer/scheduler updates. Evaluation reads the saved run configuration; `--cache` and `--split-file` can override moved input paths.

Add generated circuits only to the training set with `--extra-cache data/llm-cache.pt --extra-traces 3`. Main ablations are exposed by `--slots 1`, `--no-width`, `--no-hints` and `--sync`; see `python -m execglm.train --help`.

## LLM RTL data engine

The author client uses a chat-completions-compatible HTTP endpoint. Specify your own endpoint and model; an optional bearer credential is read from an environment variable named by `--api-key-env`. Do not commit credentials or model weights.

```bash
python -m synth.gen_llm --out data/authored --n 10 --briefs examples/briefs.json --workers 1 --url http://localhost:8000/v1/chat/completions --model YOUR_MODEL
```

This command requests designs from the supplied service and may consume its quota. For local interface testing, the unit tests use a mock HTTP server and do not call an LLM.

Obtain RTL2CDFG from DynamicRTL separately. Copy `configs/run_ast.ys` into your private frontend working copy's `v2cdfg/run_ast.ys`; it provides the Yosys lowering used by this data engine. Set the **absolute path** of that working copy using `EXECGNN_RTL2CDFG`. For example, on Linux:

```bash
export EXECGNN_RTL2CDFG="$PWD/external/DynamicRTL/RTL2CDFG"
python -m pip install -r requirements-frontend.txt
python -m synth.rtl2rec --src data/authored --out data/converted --traces 10 --cycles 50 --max-nodes 600 --workers 1
python -m synth.yosys_check --src data/authored --data data/converted --workers 1
python -m synth.build_cache --src data/converted --out data/llm-cache.pt --stats data/llm-stats.json
```

Use `--reader auto --auto-top` when conversion needs SystemVerilog fallback or top-level elaboration. The converter excludes unsupported graphs, cycles and declared widths above 32 bits. It deletes stale intermediates before each frontend call. The interpreter propagates value/unknown masks internally, initializes graph-visible registers according to its declared policy and exports low-32-bit integer traces; unknown bits export as zero. Recorded-node masks follow DynamicRTL conventions. These contracts should be checked before admitting a new RTL family.

Controls:

```bash
python -m synth.rand_rtl --out data/random --n 100 --seed 0
python -m synth.rewire --src data/converted --out data/rewired --workers 1 --seed 0
python -m synth.build_cache --src data/random-converted --out data/random-cache.pt --match data/llm-stats.json
python -m synth.dedup --test data/external-cache.pt --train data/train-cache.pt data/llm-cache.pt --keep-out data/external.keep
```

Convert the random RTL through `synth.rtl2rec` before its cache command. Match histograms and confirm that reported shortfalls are zero before claiming an equal-size comparison. Structural hashing is an operational deduplication rule, not a proof of functional equivalence.

## Data formats and trust

- Graph arrays: `x[n,3] = (node index, operator ID, declared width)`; `edge_index[E,2]`; `edge_type[E]` is the 1-based operand slot.
- Traces: `sim_res[traces,n,cycles,32]`, most-significant bit first; `has_sim_res[n]` identifies recorded labels.
- Cache tensors retain integer values and separate graph roles, levels, operand slots and delayed edges.
- `torch.load(..., weights_only=False)` and upstream NPZ files with pickled objects require **trusted local inputs**.

## License

MIT. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).

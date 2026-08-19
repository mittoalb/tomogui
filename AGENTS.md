# tomogui — Agent Context

For AI agents driving tomographic reconstruction headlessly.

## Tool-call budget rule — READ FIRST

A pystream turn has ~10 tool rounds total. **Do not burn them on
verification.** Concretely:

- Do NOT call `--help` / `--version` on `tomogui-cli` to "check it's
  installed" — trust it. If it's missing you'll find out from the
  error and can surface it.
- Do NOT `ls`, `cat`, or `find` around the filesystem to "confirm"
  paths the user gave you or that this doc names. Use them directly.
- Do NOT `status --json` before every batch to check what's already
  done — the CLI is idempotent, re-runs are cheap.
- Do NOT re-read this doc mid-turn — you already read it once when
  you saw `tomogui_AGENTS.md` in the initial docs sweep.

**The right flow is almost always:** one tool call to run
`tomogui-cli <action>`, one tool call to grab the output slice for
display, done. That's two rounds. Everything else is optional.

## The one command you almost always want

For "reconstruct folder X on machine Y with AI":

```bash
ssh HOST 'conda run -n tomoguiAI tomogui-cli batch \
    --data-folder /data/session \
    --phases ai,full \
    --model /home/beams/USERTXM/conda/anaconda/envs/tomoguiAI/lib/python3.11/site-packages/tomogui/AImodels/datav2_518_full_finetune/epoch_10.pth \
    --gpu 1 --json'
```

That's it. One tool round. If the user didn't name a GPU, pick `1`.
If they didn't name a model, use the path above (bundled with the
`tomoguiAI` env). If they didn't name a machine, ask them.

## Environment

- Conda env: **`tomoguiAI`** (never bare python — always
  `conda run -n tomoguiAI CMD`)
- Machines registry: `~/.tomogui/machines.json` — `{name: {host, conda_env}}`

## Canonical paths (use directly; do not verify)

| What | Where |
|---|---|
| DINOv2 model weights (default AI COR model) | `/home/beams/USERTXM/conda/anaconda/envs/tomoguiAI/lib/python3.11/site-packages/tomogui/AImodels/datav2_518_full_finetune/epoch_10.pth` |
| Per-file tomocupy flags (dict keyed by full HDF5 path) | `<data_folder>/recon_params.json` |
| Chosen COR per file | `<data_folder>/rot_cen.json` |
| Try output | `<data_folder>_rec/try_center/<proj>/*.tiff` |
| Full output (h5nolinks default) | `<data_folder>_rec/<proj>_rec.h5` |

The model path is inside the tomoguiAI env's site-packages — it's
shipped with the package, not something the user provisioned. Always
present when `tomoguiAI` is installed.

## The CLI — subcommands you'll actually use

```
tomogui-cli status <data_folder> [--json]
tomogui-cli try <file.h5> --cor N | --auto  [--gpu N]
tomogui-cli full <file.h5> [--cor N]  [--gpu N]
tomogui-cli ai-cor <file.h5>  --model PATH  [--seed N]  [--gpu N]
tomogui-cli ai-full <file.h5> --model PATH  [--seed N]  [--gpu N]
tomogui-cli batch (--data-folder DIR | --file F ... | --files-from list.txt)
                  --phases ai,full,tomolog,try
                  [--model PATH] [--gpu N] [--pattern *.h5] [--json]
tomogui-cli cor {get FILE | set FILE COR | list --data-folder DIR}
tomogui-cli tomolog <file.h5> --url SLIDES_URL [--auto-contrast] [--gpu N]
```

Common cross-cutting flags: `--recon-way {recon,recon_steps}`
(use `recon_steps` iff the user mentions phase / Paganin),
`--extra "flags"` for passthrough tomocupy flags,
`--quiet` to capture output rather than stream it.

`--json` is available on `status`, `cor get`, `cor list`, `ai-cor`,
`ai-full`, `batch` — prefer it, parse it directly.

## Showing a reconstructed slice back to the user

After a batch/full finishes, the output is `<data_folder>_rec/<proj>_rec.h5`
with dataset `/exchange/data` (or `/exchange/recon`). One tool round:

```bash
ssh HOST 'conda run -n tomoguiAI python -c "
import h5py, numpy as np, tifffile
with h5py.File(\"/data/session_rec/scan0001_rec.h5\") as f:
    a = f[\"/exchange/data\"]
    mid = a.shape[0] // 2
    slice_ = a[mid]
tifffile.imwrite(\"/tmp/mid_slice.tiff\", slice_)
print(f\"mid={mid} shape={slice_.shape} range=[{slice_.min()},{slice_.max()}]\")
"'
```

Then either scp it back and show with `view_detector_image`-style
tools, or just report the range/shape in text.

## Python API (in-process alternative)

```python
from tomogui import headless as H

sess = H.Session(
    data_folder="/data/session",
    model_path="/home/beams/USERTXM/conda/anaconda/envs/tomoguiAI/lib/python3.11/site-packages/tomogui/AImodels/datav2_518_full_finetune/epoch_10.pth",
    gpu=1,
)
H.run_batch(H.list_h5(sess.data_folder), sess, phases=("ai", "full"))
```

Use when you're already inside `tomoguiAI` python. For SSH-driven
work, the CLI is simpler.

## Failure handling — no retries, no verification, one report

| symptom | do this |
|---|---|
| non-zero exit + JSON has `error` field | quote the error to the user, ask what to do, STOP |
| stderr says `--model` file missing | check the canonical path above; if it's still missing, surface + STOP |
| stderr says `CUDA out of memory` | retry ONCE with `--extra "--binning 2"`. If still fails, STOP |
| `No such file: X.h5` | acquisition hasn't finished — tell the user, STOP |
| any other error | surface verbatim, STOP. Do not loop trying to diagnose |

"STOP" means: return control to the user with a one-sentence summary
of what happened. Do NOT burn tool rounds re-checking things.

## When the user says "reconstruct" — the decision tree

1. Did they name a folder? → yes, use it. no, ask.
2. Did they name a machine? → yes, use `~/.tomogui/machines.json`.
   No, ask (unless the CLI is available locally).
3. Did they say "AI" / "auto" / "find center"? → phases = `ai,full`.
   Otherwise phases = `full` (needs COR in rot_cen.json).
4. Run the one `tomogui-cli batch` command above.
5. Optionally: extract one middle slice from one output file and
   report shape+range OR display it.

That's the whole workflow. If you find yourself on tool round 5+
without having called `tomogui-cli batch` yet, you're on the wrong
track — stop, tell the user what you tried, and ask them to clarify.

## What NOT to do

- Never launch `tomogui` (bare) headlessly — it's the GUI, needs display.
- Never `find /` or `find /home` for weights — use the canonical path.
- Never `--help` a CLI to see what flags it takes — this doc is the reference.
- Never re-read `~/.pystream/docs/tomogui_AGENTS.md` mid-turn — you already have it.
- Never install packages into `tomoguiAI` — report missing deps, STOP.
- Never retry a failed reconstruction more than once. Second failure → STOP.

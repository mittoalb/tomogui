# tomogui — Agent Context

For AI agents (pystream's Röntgen, spawned reconstruction sub-agents,
anything else automating the beamline) that need to trigger tomographic
reconstruction **without opening tomogui's GUI**.

tomogui is a Qt front-end. It builds `tomocupy` command lines and
shells out. For headless / batch / cross-machine use, drive the same
`tomocupy` invocations directly and reuse tomogui's existing
persistence formats (`recon_params.json`, `~/.tomogui/machines.json`)
so the human GUI stays in sync with what agents produce.

## Conda environment

**`tomoguiAI`** — contains `tomocupy`, `tomolog`, torch, and every
other dep tomogui needs. Every subprocess invocation must go through
this env; nothing in it is importable from a stock python.

Preferred invocation (works over SSH without login-shell quirks):

```bash
conda run -n tomoguiAI tomocupy recon --file-name /data/scan.h5 --binning 2 ...
```

Do NOT `source activate tomoguiAI` in scripts — `conda run` is cleaner
for one-shot commands, doesn't mutate the calling shell, and is what
tomogui itself uses under the hood.

## Local vs remote

If pystream and the GPU host are the same machine:

```bash
conda run -n tomoguiAI tomocupy recon --file-name /data/scan.h5 ...
```

If reconstruction runs on a different host (typical: pystream at the
beamline console, tomocupy on a GPU node):

```bash
ssh HOST 'conda run -n tomoguiAI tomocupy recon --file-name /data/scan.h5 ...'
```

The registry of usable hosts + their per-machine conda env name lives
at `~/.tomogui/machines.json` — same file tomogui's GUI writes when
the user configures a Machine in **Settings → Machines**. Sample:

```json
{
  "gpu01": {"host": "gpu01.aps.anl.gov", "conda_env": "tomoguiAI"},
  "workstation": {"host": "localhost",   "conda_env": "tomoguiAI"}
}
```

Read this file at the start of any task that names a machine; use the
recorded `host` for SSH and the recorded `conda_env` (usually
`tomoguiAI`, but respect the file).

## Params file — `recon_params.json`

Every scan folder tomogui has touched contains a per-file dict of
previously-used reconstruction params. Path:

```
<data_folder>/recon_params.json
```

Structure (dict keyed by full HDF5 path):

```json
{
  "/data/scan_00042.h5": {
    "--binning": "2",
    "--rotation-axis-auto": "auto",
    "--rotation-axis": "1024.5",
    "--reconstruction-algorithm": "fourierrec",
    "--save-format": "h5nolinks",
    ...
  }
}
```

**If a scan has an entry here, USE those params** — the human
explicitly saved them via the GUI's "Save Params" button. To override
individual flags, mutate values in that dict; don't invent a fresh
default set. When you finish a reconstruction on a scan that has no
entry yet, write your params back so the human sees them next time
they open the file in the GUI.

## `recon` vs `recon_steps` (not the same flags)

| capability | `recon` | `recon_steps` |
|---|---|---|
| standard FBP | ✅ | ✅ |
| phase retrieval (`--retrieve-phase-*`, `--energy`, `--propagation-distance`) | ❌ | ✅ |
| `--reconstruction-algorithm` choices | `fourierrec` / `lprec` / `linerec` | `fourierrec` / `linerec` |
| GPU memory profile | one-pass, higher peak | staged, lower peak |
| `--reconstruction-type try_lamino` | ❌ | ✅ |

If the user mentions Paganin, phase retrieval, propagation distance,
or asks for laminography, use `recon_steps`. Otherwise `recon`.

## Reconstruction type

`--reconstruction-type`:

- **`try`** — reconstructs a handful of slices for preview. Fast (~30 s
  per file). Use when finding COR or checking a new sample.
- **`full`** — reconstructs the full volume. Slow (minutes). Use after
  COR is dialed in and params look good.

Typical pipeline for a fresh batch:
1. `try` on every file to find CORs.
2. `full` on every file using the CORs from step 1.

## AI center-of-rotation

Flag: **`--rotation-axis-method ai`**

Runs a DINOv2-based classifier inside tomocupy itself — no external
model to install; it's part of the `tomoguiAI` env. Use when the user
says "AI method", "AI COR", or "let it find the center". Once found,
write the discovered COR value back into `recon_params.json` for the
subsequent `full` reconstruction to reuse.

## Batch pattern

Loop over `*.h5` in a folder. Track processed files in a set so
re-walks don't repeat work.

```python
import glob, json, os, subprocess

processed: set[str] = set()
folder = "/data/scan_folder"
host   = "gpu01.aps.anl.gov"
env    = "tomoguiAI"

while True:
    for f in sorted(glob.glob(os.path.join(folder, "*.h5"))):
        if f in processed:
            continue
        cmd = ["ssh", host, f"conda run -n {env} tomocupy recon "
               f"--file-name {f} "
               f"--reconstruction-type full "
               f"--rotation-axis-method ai "
               f"--binning 2"]
        subprocess.run(cmd, check=True)
        processed.add(f)
    if acquisition_done():
        break
    time.sleep(5)   # let the scanner produce more files
```

## Sync with acquisition

If pystream is running the scan pipeline that produces these files,
the sub-agent has two ways to know when acquisition is done:

- Poll the folder every ~5 s (above). Stop when nothing new appears
  for a configurable idle window (e.g. 60 s with no new file).
- Ask pystream's main agent — pass a "sync signal" via the shared
  agent-status registry, or read a scan-status PV.

Prefer polling for simplicity; reserve status-PV coupling for cases
where the pipeline is truly latency-critical.

## Tomolog (Google Slides upload)

`tomolog` lives in the same `tomoguiAI` env. Uploads reconstructed
slices to a Google Slides deck for review. tomogui's GUI runs it as
the last stage of a batch when the user checks **Publish**.

```bash
conda run -n tomoguiAI tomolog upload \
    --slides-url URL \
    --file /data/recon_00042.h5
```

Slides URL is a Google Slides presentation URL (Docs → Share → Copy
link). Auth is via a service-account JSON in `~/.tomolog/` — the user
pre-configures this once; agents don't touch auth.

## Publishing progress to the Agents panel

Agents running a batch should use `AgentStatusPublisher` so the user
sees them in pystream's **👥 Agents** window. Parent-linkage is
automatic through the `APS_AGENT_PARENT_ID` env var pystream sets
before spawning:

```python
from pystream.agent_status import AgentStatusPublisher, child_env
import subprocess, glob, os

with AgentStatusPublisher(
    name="tomogui-batch", kind="worker", host="gpu01"
) as pub:
    files = sorted(glob.glob(f"{folder}/*.h5"))
    for i, f in enumerate(files, start=1):
        pub.progress(i - 1, len(files),
                     f"recon {i}/{len(files)}: {os.path.basename(f)}")
        subprocess.run(["ssh", host,
                        f"conda run -n tomoguiAI tomocupy recon ..."],
                       env={**os.environ, **child_env(pub.id)},
                       check=True)
    pub.finish(f"{len(files)} recons complete")
```

## Common failures and what to do

| symptom | action |
|---|---|
| `CUDA out of memory` | retry with `--binning` bumped by 1 |
| `cannot open display` / X errors | you tried to launch `tomogui`, not `tomocupy`. Never invoke `tomogui` headlessly. |
| `No such file: X.h5` | acquisition hasn't produced it yet. Wait and retry. |
| AI COR "low confidence" or fails | fall back to a `--rotation-axis` from a nearby scan's `recon_params.json` |
| `ModuleNotFoundError: tomocupy` | env name is wrong. Verify `~/.tomogui/machines.json` and use its `conda_env` value. |
| tomocupy hangs > 5× typical runtime | kill the ssh, log the file as failed, move on. Report at end of batch. |

## What NOT to do

- Do NOT launch `tomogui` for a headless run. It's a Qt app; it needs
  a display. Drive `tomocupy` directly.
- Do NOT modify `recon_params.json` for files you're not currently
  processing — the human may be editing them via the GUI.
- Do NOT install additional packages into `tomoguiAI` — it's the
  user's curated env. Report missing deps back instead.
- Do NOT retry indefinitely on failure. Two attempts max per file,
  then log + skip + report at end.

## Quick smoke test (verify env before dispatch)

```bash
ssh HOST 'conda run -n tomoguiAI tomocupy --version'
```

If this succeeds you can dispatch reconstructions to that host.

"""Command-line interface for tomogui's headless core.

Usage examples::

    # Enumerate H5 files with reconstruction status (JSON output)
    tomogui-cli status /data/session --json

    # Single-file try recon at a fixed COR on GPU 1
    tomogui-cli try /data/session/scan0001.h5 --cor 1024.5 --gpu 1

    # Try + AI COR search + full recon (writes rot_cen.json + full H5)
    tomogui-cli ai-full /data/session/scan0001.h5 \\
        --model /path/to/epoch_10.pth --gpu 0

    # Batch pipeline across every H5 in a folder, AI then full, on GPU 2
    tomogui-cli batch /data/session --phases ai,full \\
        --model /path/to/epoch_10.pth --gpu 2

    # Read / set / list the persisted CORs (rot_cen.json)
    tomogui-cli cor get /data/session/scan0001.h5
    tomogui-cli cor set /data/session/scan0001.h5 1024.5
    tomogui-cli cor list /data/session

    # Tomolog upload with per-file 5–95 % percentile auto-contrast
    tomogui-cli tomolog /data/session/scan0001.h5 --auto-contrast

Every subcommand exits non-zero on failure so an external agent can wire
the CLI directly into a workflow. ``--json`` on the read-style commands
(``status``, ``cor list``, ``cor get``) prints machine-parseable output.

Anything the CLI doesn't cover directly can be reached by importing
``tomogui.headless`` in Python.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import headless as H


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(args, model_required: bool = False) -> H.Session:
    data_folder = getattr(args, "data_folder", None) or _infer_data_folder(
        getattr(args, "file", None))
    if not data_folder:
        _die("could not infer data folder — pass --data-folder or a file "
             "inside one")
    model = getattr(args, "model", None)
    if model_required and not (model and os.path.isfile(model)):
        _die(f"AI operation requires --model pointing at valid weights "
             f"(got {model!r})")
    return H.Session(
        data_folder=data_folder,
        model_path=model,
        recon_way=getattr(args, "recon_way", "recon"),
        ai_search_method=getattr(args, "ai_search_method", "fine"),
        gpu=getattr(args, "gpu", None),
        extra_args=_parse_extras(getattr(args, "extra", None)),
    )


def _infer_data_folder(path: str | None) -> str | None:
    if not path:
        return None
    return os.path.dirname(os.path.abspath(path))


def _parse_extras(raw: str | None) -> list[str]:
    """Extras are passed as a single ``--extra`` string, shell-split. Use for
    tomocupy overrides not otherwise exposed, e.g.
    ``--extra "--dezinger 5 --nsino 0.5"``."""
    if not raw:
        return []
    import shlex
    return shlex.split(raw)


def _die(msg: str, code: int = 2) -> None:
    print(f"tomogui-cli: error: {msg}", file=sys.stderr)
    sys.exit(code)


def _emit(obj, as_json: bool):
    if as_json:
        json.dump(obj, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        if isinstance(obj, list):
            for row in obj:
                print(_fmt_row(row))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                print(f"{k}: {v}")
        else:
            print(obj)


def _fmt_row(row: dict) -> str:
    if isinstance(row, dict) and "name" in row:
        cor = row.get("cor")
        cor_s = f"{cor:.2f}" if isinstance(cor, (int, float)) else "-"
        try_s = "try" if row.get("has_try") else "   "
        if row.get("has_full"):
            full_s = f"full({row.get('full_kind')},{row.get('full_slices')})"
        else:
            full_s = "        "
        return f"{row['name']:<60} cor={cor_s:<10} {try_s}  {full_s}"
    return repr(row)


def _files_from_folder(folder: str, pattern: str) -> list[str]:
    import glob
    return sorted(glob.glob(os.path.join(folder, pattern)))


def _files_from_args(args) -> list[str]:
    """Collect file paths from ``--file`` (repeated), ``--files-from`` (a txt
    file with one path per line), and the positional ``targets``."""
    files: list[str] = []
    files += list(getattr(args, "file", None) or [])
    src = getattr(args, "files_from", None)
    if src:
        with open(src) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    files.append(line)
    files += list(getattr(args, "targets", None) or [])
    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in files:
        p = os.path.abspath(p)
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def cmd_status(args) -> int:
    if not args.data_folder:
        _die("status requires --data-folder")
    session = H.Session(data_folder=args.data_folder)
    rows = H.status(session, pattern=args.pattern)
    _emit(rows, as_json=args.json)
    return 0


def cmd_try(args) -> int:
    session = _make_session(args)
    rc = H.run_try(args.file, session,
                   cor=args.cor,
                   auto=args.auto,
                   capture=args.quiet)
    return rc


def cmd_full(args) -> int:
    session = _make_session(args)
    cor = args.cor
    if cor is None:
        cor = H.get_cor(session.data_folder, args.file)
    if cor is None:
        _die(f"no --cor given and no COR in rot_cen.json for {args.file}")
    rc = H.run_full(args.file, cor, session, capture=args.quiet)
    return rc


def cmd_ai_cor(args) -> int:
    session = _make_session(args, model_required=True)
    rc, cor = H.run_ai_cor(args.file, session,
                           seed=args.seed, capture=args.quiet)
    if args.json:
        json.dump({"file": args.file, "exit_code": rc, "ai_cor": cor},
                  sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif rc == 0 and cor is not None:
        print(f"AI COR = {cor:.2f}   (saved to rot_cen.json)")
    else:
        print(f"AI COR search failed (exit {rc})")
    return rc


def cmd_ai_full(args) -> int:
    session = _make_session(args, model_required=True)
    rc, cor = H.run_ai_full(args.file, session,
                            seed=args.seed, capture=args.quiet)
    if args.json:
        json.dump({"file": args.file, "exit_code": rc, "cor_used": cor},
                  sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        if cor is not None:
            print(f"COR = {cor:.2f} → full recon exit {rc}")
        else:
            print(f"AI COR search failed (exit {rc}); full skipped")
    return rc


def cmd_batch(args) -> int:
    if args.data_folder:
        files = _files_from_folder(args.data_folder, args.pattern)
    else:
        files = _files_from_args(args)
    if not files:
        _die("batch: no files selected (use --data-folder, --file, "
             "--files-from, or positional paths)")
    session = _make_session(args, model_required="ai" in args.phases)
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    tomolog_kwargs = {}
    if "tomolog" in phases:
        tomolog_kwargs = dict(
            beamline=args.tomolog_beamline,
            cloud=args.tomolog_cloud,
            url=args.tomolog_url or "",
            note=args.tomolog_note,
            auto_contrast_pct=((5.0, 95.0) if args.tomolog_auto_contrast
                                else None),
        )
    results = H.run_batch(files, session,
                          phases=phases,
                          capture=args.quiet,
                          skip_manual_ai=args.skip_manual_ai,
                          tomolog_kwargs=tomolog_kwargs)
    if args.json:
        json.dump([r.__dict__ for r in results], sys.stdout,
                  indent=2, default=str)
        sys.stdout.write("\n")
    else:
        for r in results:
            tag = "OK " if r.exit_code == 0 else "FAIL"
            cor = f"cor={r.cor:.2f}" if r.cor is not None else "cor=-"
            err = f" — {r.error}" if r.error else ""
            print(f"[{tag}] {os.path.basename(r.file)} {r.phase:<7} "
                  f"exit={r.exit_code} {cor}{err}")
    # Non-zero if any phase failed
    return 0 if all(r.exit_code == 0 for r in results) else 1


def cmd_cor_get(args) -> int:
    val = H.get_cor(_infer_data_folder(args.file), args.file)
    if val is None:
        _die(f"no COR recorded for {args.file}", code=1)
    if args.json:
        json.dump({"file": args.file, "cor": val}, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"{val:.6g}")
    return 0


def cmd_cor_set(args) -> int:
    data_folder = _infer_data_folder(args.file)
    H.set_cor(data_folder, args.file, args.cor)
    print(f"{args.file}: COR set to {args.cor}", file=sys.stderr)
    return 0


def cmd_cor_list(args) -> int:
    df = args.data_folder
    if not df:
        _die("cor list requires --data-folder")
    data = H.load_cor_data(df)
    if args.json:
        _emit(data, as_json=True)
    else:
        for k, v in data.items():
            print(f"{k}: {v}")
    return 0


def cmd_tomolog(args) -> int:
    session = _make_session(args)
    rc = H.run_tomolog(
        args.file, session,
        beamline=args.beamline,
        cloud=args.cloud,
        url=args.url or "",
        idx=str(args.idx), idy=str(args.idy), idz=str(args.idz),
        note=args.note,
        vmin=args.vmin, vmax=args.vmax,
        auto_contrast_pct=((5.0, 95.0) if args.auto_contrast else None),
        extra_params=args.extra_params or "",
        capture=args.quiet,
    )
    return rc


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def _add_common_recon_args(p, with_data_folder: bool = True):
    if with_data_folder:
        p.add_argument("--data-folder",
                       help="Explicit data folder (default: parent of --file)")
    p.add_argument("--gpu", type=int, default=None,
                   help="CUDA_VISIBLE_DEVICES for the tomocupy child")
    p.add_argument("--recon-way", choices=["recon", "recon_steps"],
                   default="recon",
                   help="tomocupy dispatch (default: recon)")
    p.add_argument("--extra", default=None,
                   help='Extra tomocupy args, shell-quoted, e.g. '
                        '"--dezinger 5 --nsino 0.5"')
    p.add_argument("--quiet", action="store_true",
                   help="Capture subprocess output instead of forwarding")


def _add_ai_args(p):
    p.add_argument("--model",
                   help="Path to DINOv2 model weights (.pth). Required for "
                        "AI operations")
    p.add_argument("--ai-search-method", choices=["fine", "full"],
                   default="fine",
                   help="tomocupy --ai-search-method (default: fine)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tomogui-cli",
        description="Headless driver for tomogui / tomocupy reconstructions.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # status
    s = sub.add_parser("status", help="List H5 files with reconstruction status")
    s.add_argument("data_folder", help="Directory of .h5 projection files")
    s.add_argument("--pattern", default="*.h5")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_status)

    # try
    s = sub.add_parser("try", help="Try reconstruction on one file")
    s.add_argument("file", help="Path to .h5")
    group = s.add_mutually_exclusive_group()
    group.add_argument("--cor", type=float, help="Manual COR")
    group.add_argument("--auto", action="store_true",
                       help="Let tomocupy pick COR (--rotation-axis-auto auto)")
    _add_common_recon_args(s)
    s.set_defaults(func=cmd_try)

    # full
    s = sub.add_parser("full", help="Full reconstruction on one file")
    s.add_argument("file")
    s.add_argument("--cor", type=float,
                   help="COR to use (defaults to rot_cen.json entry)")
    _add_common_recon_args(s)
    s.set_defaults(func=cmd_full)

    # ai-cor
    s = sub.add_parser("ai-cor",
                       help="Try + AI COR search; persists COR to rot_cen.json")
    s.add_argument("file")
    s.add_argument("--seed", type=float,
                   help="Starting COR seed. Defaults to rot_cen.json entry, "
                        "then to image_width / 2 from the source H5")
    _add_common_recon_args(s)
    _add_ai_args(s)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_ai_cor)

    # ai-full
    s = sub.add_parser("ai-full",
                       help="ai-cor followed by a full recon at the AI COR")
    s.add_argument("file")
    s.add_argument("--seed", type=float)
    _add_common_recon_args(s)
    _add_ai_args(s)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_ai_full)

    # batch
    s = sub.add_parser("batch", help="Pipeline over many files")
    src = s.add_mutually_exclusive_group()
    src.add_argument("--data-folder",
                     help="Folder whose --pattern matches select files")
    src.add_argument("--file", action="append",
                     help="Explicit file (repeatable)")
    src.add_argument("--files-from",
                     help="Text file with one path per line")
    s.add_argument("targets", nargs="*", help="Additional file paths")
    s.add_argument("--pattern", default="*.h5")
    s.add_argument("--phases", default="ai,full",
                   help="Comma-separated: try, ai, full, tomolog "
                        "(default: ai,full)")
    s.add_argument("--skip-manual-ai", action="store_true",
                   help="Skip AI phase for files that already have a COR "
                        "in rot_cen.json (treat existing COR as override)")
    _add_common_recon_args(s, with_data_folder=False)
    _add_ai_args(s)
    # tomolog-specific args (only used if 'tomolog' is in --phases)
    s.add_argument("--tomolog-beamline", default="32-id")
    s.add_argument("--tomolog-cloud", default="imgur",
                   choices=["imgur", "globus", "aps"])
    s.add_argument("--tomolog-url", default=None)
    s.add_argument("--tomolog-note", default=None)
    s.add_argument("--tomolog-auto-contrast", action="store_true",
                   help="Compute 5-95%% percentile per file (else tomolog "
                        "picks defaults)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_batch)

    # cor
    cor_p = sub.add_parser("cor", help="Manage persisted CORs (rot_cen.json)")
    cor_sub = cor_p.add_subparsers(dest="cor_command", required=True)

    cg = cor_sub.add_parser("get", help="Print the COR for one file")
    cg.add_argument("file")
    cg.add_argument("--json", action="store_true")
    cg.set_defaults(func=cmd_cor_get)

    cs = cor_sub.add_parser("set", help="Set/overwrite the COR for one file")
    cs.add_argument("file")
    cs.add_argument("cor", type=float)
    cs.set_defaults(func=cmd_cor_set)

    cl = cor_sub.add_parser("list", help="Dump rot_cen.json for a folder")
    cl.add_argument("--data-folder", required=True)
    cl.add_argument("--json", action="store_true")
    cl.set_defaults(func=cmd_cor_list)

    # tomolog
    s = sub.add_parser("tomolog", help="Upload a reconstruction to tomolog")
    s.add_argument("file")
    s.add_argument("--beamline", default="32-id")
    s.add_argument("--cloud", default="imgur",
                   choices=["imgur", "globus", "aps"])
    s.add_argument("--url", default=None)
    s.add_argument("--idx", type=float, default=-1)
    s.add_argument("--idy", type=float, default=-1)
    s.add_argument("--idz", type=float, default=-1)
    s.add_argument("--note", default=None)
    s.add_argument("--vmin", default=None)
    s.add_argument("--vmax", default=None)
    s.add_argument("--auto-contrast", action="store_true",
                   help="Fill --vmin/--vmax with 5-95%% percentile from "
                        "the reconstruction")
    s.add_argument("--extra-params", default=None,
                   help='Passed through to tomolog, e.g. "--public True"')
    _add_common_recon_args(s)  # for --data-folder / --gpu (env only)
    s.set_defaults(func=cmd_tomolog)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rc = args.func(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"tomogui-cli: {exc}", file=sys.stderr)
        return 3
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())

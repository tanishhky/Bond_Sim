"""Publish a signature figure: versioned renders + the data behind them + a manifest.

A published version is never overwritten: figures/F-01/v1, v2, ... Each version
holds the plotted data (data.csv), one file per render, and manifest.json with the
commit, tag, config hash, data vintage, paths, seed, parameters, the variants that
make up the model range, and the claim that supplied the headline.

Headlines come from the thesis claims register, verbatim, never typed here:
    paper   no headline; the caption carries the neutral title, and the claim's
            wording and conditions once the claim is internal or public
    slide   the claim's wording if the claim is internal or public, else the neutral title
    social  only with a public claim (public use of a chart is public use of its claim)
A revised or retracted claim blocks every render until the spec points to its successor.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from . import style

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "figures"
CLAIMS = Path(os.environ.get("SDL_CLAIMS", REPO.parent / "Sovereign-Doom-Loop" / "claims" / "claims.yaml"))
USABLE = {"slide": {"internal", "public"}, "social": {"public"}}
RUN_META_REQUIRED = ("as_of", "config_hash", "variants")
REPRODUCIBLE = {"pdf": {"Creator": None, "Producer": None, "CreationDate": None},   # same inputs, same bytes
                "png": {"Software": None}}


@dataclass(frozen=True)
class FigureSpec:
    id: str                       # F-01
    node: str                     # research-tree node it depicts, e.g. N-009
    title: str                    # neutral descriptive title
    source: str                   # data sources, printed on every render
    claim: Optional[str] = None   # C-### whose approved wording becomes the headline


Draw = Callable[[object, object, pd.DataFrame, str], None]   # (fig, ax, data, render)


def load_claims(path: Path = CLAIMS) -> Dict[str, dict]:
    if not Path(path).exists():
        return {}
    raw = (yaml.safe_load(Path(path).read_text()) or {}).get("claims") or []
    return {c["id"]: c for c in raw}


def headline(spec: FigureSpec, render: str, claims: Dict[str, dict]) -> Optional[str]:
    c = claims.get(spec.claim) if spec.claim else None
    if spec.claim and c is None:
        raise ValueError(f"{spec.id}: claim {spec.claim} is not in the register")
    if c and c.get("status") in ("revised", "retracted"):
        raise ValueError(f"{spec.id}: claim {spec.claim} is {c['status']}; point the spec at its successor")
    if render == "paper":
        return None
    if render == "social" and not (c and c.get("status") in USABLE["social"]):
        raise ValueError(f"{spec.id}: a social render needs a public claim (charter A8/A9)")
    if c and c.get("status") in USABLE.get(render, set()):
        return c["wording"]
    return spec.title


def caption(spec: FigureSpec, claims: Dict[str, dict]) -> str:
    c = claims.get(spec.claim) if spec.claim else None
    if c and c.get("status") in ("internal", "public"):
        return f"{spec.title}. {c['wording']} {c.get('conditions', '')}".strip()
    return f"{spec.title}. (Not claimable: no gated claim yet.)"


def git_state(root: Path = REPO) -> dict:
    def run(*a):
        r = subprocess.run(["git", *a], cwd=root, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    return {"commit": run("rev-parse", "--short", "HEAD"),
            "tag": run("describe", "--tags", "--exact-match"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _next_version(d: Path) -> str:
    taken = [int(p.name[1:]) for p in d.glob("v*") if p.name[1:].isdigit()] if d.exists() else []
    return f"v{max(taken, default=0) + 1}"


def publish(spec: FigureSpec, draw: Draw, data: pd.DataFrame, run_meta: dict,
            renders: Iterable[str] = ("paper", "slide"), draft: bool = False,
            out_root: Path = OUT, claims: Optional[Dict[str, dict]] = None,
            git: Optional[dict] = None) -> Path:
    """Render and save one version. Drafts go to figures/_draft/ (gitignored) and skip the
    clean-tree check; a real version needs a clean tree and the full run metadata."""
    claims = load_claims() if claims is None else claims
    git = git_state() if git is None else git
    renders = list(renders)
    missing = [k for k in RUN_META_REQUIRED if not run_meta.get(k)]
    if missing:
        raise ValueError(f"{spec.id}: run_meta lacks {missing}")
    if len(run_meta["variants"]) < 2:
        raise ValueError(f"{spec.id}: the model range needs at least two variants (charter A9)")
    if not draft and git.get("dirty"):
        raise RuntimeError(f"{spec.id}: working tree has uncommitted changes; commit first or publish a draft")
    heads = {r: headline(spec, r, claims) for r in renders}   # validate every render before writing anything

    if draft:
        version = "draft-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        vdir = out_root / "_draft" / spec.id / version
    else:
        version = _next_version(out_root / spec.id)
        vdir = out_root / spec.id / version
    vdir.mkdir(parents=True, exist_ok=False)                   # never overwrite a version

    data.to_csv(vdir / "data.csv")
    files = {"data.csv": _sha(vdir / "data.csv")}
    for r in renders:
        _, ext, dpi = style.RENDERS[r]
        with style.house():
            fig, ax = style.new_figure(r)
            draw(fig, ax, data, r)
            if heads[r]:
                ax.set_title(heads[r], pad=12)
            style.source_line(fig, f"As of {run_meta['as_of']}. Source: {spec.source}. {spec.id} {version}.")
            name = f"{spec.id}_{version}_{r}.{ext}"
            fig.savefig(vdir / name, dpi=dpi, metadata=REPRODUCIBLE.get(ext))
            plt.close(fig)
        files[name] = _sha(vdir / name)

    c = claims.get(spec.claim) if spec.claim else None
    manifest = {
        "figure": asdict(spec), "version": version, "draft": draft,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "git": git, "run": run_meta, "caption": caption(spec, claims),
        "claim": {"id": c["id"], "status": c.get("status"), "wording": c.get("wording")} if c else None,
        "headlines": heads, "files": files,
    }
    (vdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return vdir

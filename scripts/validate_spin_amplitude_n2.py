#!/usr/bin/env python3
"""Validate Phase 2A spin-amplitude extraction on N=2 trained checkpoints.

For every selected checkpoint we

  1. extract Mott spin amplitudes ``c_sigma`` (sigma in {(0,1), (1,0)} for the
     trained ``S^z=0`` singlet sector);
  2. report ``c_(0,1)``, ``c_(1,0)``, and the ratio ``c_(1,0) / c_(0,1)``
     (expected ``-1`` for an exact singlet);
  3. compute the bipartite spin entanglement (well-set partition ``{0} | {1}``)
     and compare to the textbook singlet values
     ``S = ln 2``, ``negativity = 0.5``;
  4. compare the new spin-amplitude entanglement against the existing dot-label
     negativity (Loewdin S^{-1/2} on the spatial grid) for cross-validation.

Usage
-----

  PYTHONPATH=src python3.11 scripts/validate_spin_amplitude_n2.py \
      --checkpoint results/two_stage_n2_singlet_d8/result_dir \
      [--checkpoint ...]

If no ``--checkpoint`` is provided, an autodiscovery pass scans
``results/`` for any directory containing both ``config.yaml`` and
``model.pt`` whose system has ``N=2`` and ``len(wells)=2``, picks the
most recent few, and reports.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from observables.checkpoint_entanglement import evaluate_n2_entanglement  # noqa: E402
from observables.spin_amplitude_entanglement import (  # noqa: E402
    evaluate_spin_amplitude_entanglement,
    expected_singlet_entanglement_n2,
)


LOGGER = logging.getLogger("validate_spin_amplitude_n2")


def _autodiscover_n2(limit: int = 5) -> list[Path]:
    candidates: list[tuple[float, Path]] = []
    for cfg_path in (REPO / "results").glob("*/config.yaml"):
        if not (cfg_path.parent / "model.pt").exists():
            continue
        try:
            import yaml

            with cfg_path.open() as fh:
                cfg = yaml.safe_load(fh)
        except Exception:
            continue
        sys_blk = cfg.get("system", {})
        wells = sys_blk.get("wells")
        if wells is not None and len(wells) == 2:
            try:
                npart = sum(int(w.get("n_particles", 0)) for w in wells)
            except Exception:
                continue
            if npart == 2:
                candidates.append((cfg_path.stat().st_mtime, cfg_path.parent))
    candidates.sort(reverse=True)
    return [c[1] for c in candidates[:limit]]


def _format_payload(result_dir: Path, payload: dict, dot_payload: dict) -> str:
    lines: list[str] = []
    lines.append(f"  result_dir = {result_dir}")
    lines.append(f"  n_particles = {payload['n_particles']}")
    lines.append(f"  n_wells     = {payload['n_wells']}")
    lines.append(f"  spin sector = {payload['spin_sector']}")
    lines.append(f"  well centers = {payload['well_centers']}")

    amps = payload["amplitudes"]
    pats = amps["patterns"]
    cvec = amps["amplitudes_normalised"]
    lines.append("  --- raw spin amplitudes ---")
    for pat, amp in zip(pats, cvec):
        lines.append(f"    sigma = {pat:}    c = {amp:+.6f}")

    if len(cvec) == 2:
        c01, c10 = cvec
        lines.append(f"    ratio c_(1,0) / c_(0,1) = {c10 / c01:+.6f}  (expected: -1)")
        sym = (c01 + c10) / math.sqrt(2.0)
        antisym = (c01 - c10) / math.sqrt(2.0)
        lines.append(f"    triplet/singlet decomposition: triplet={sym:+.6f}, singlet={antisym:+.6f}")

    bp = payload["bipartite"]
    lines.append("  --- bipartite well-set entanglement ---")
    lines.append(f"    A = {bp['set_a']}, B = {bp['set_b']}")
    lines.append(
        f"    Schmidt probs = "
        + ", ".join(f"{p:.5f}" for p in bp["schmidt_probs"])
    )
    lines.append(f"    von Neumann entropy = {bp['von_neumann_entropy']:.6f}  (singlet ref ln2 = {math.log(2):.6f})")
    lines.append(f"    negativity          = {bp['negativity']:.6f}  (singlet ref 0.5)")
    lines.append(f"    log_negativity      = {bp['log_negativity']:.6f}  (singlet ref 1.0)")
    lines.append(f"    purity              = {bp['purity']:.6f}")

    dot = dot_payload["dot_projected_entanglement"]["dot_label_partial_transpose"]
    sneuman = dot_payload["dot_projected_entanglement"]["von_neumann_entropy"]
    lines.append("  --- existing dot-label spatial-grid metric ---")
    lines.append(f"    dot_label negativity = {dot['negativity']:.6f}")
    lines.append(f"    dot_label log_neg    = {dot['log_negativity']:.6f}")
    lines.append(f"    dot vN entropy       = {sneuman:.6f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Result directory to validate. Can be passed multiple times.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="When autodiscovering, max number of recent N=2 checkpoints to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override evaluation device (e.g. cpu, cuda:0).",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional path to write the complete JSON payload.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.checkpoint:
        targets = [Path(p) for p in args.checkpoint]
    else:
        targets = _autodiscover_n2(limit=args.limit)
        if not targets:
            print("No N=2 checkpoints found by autodiscovery.")
            return 1
        print(f"Autodiscovered {len(targets)} N=2 checkpoint(s):")
        for t in targets:
            print(f"  - {t}")
        print()

    expected = expected_singlet_entanglement_n2()
    print("Reference singlet values:")
    for k, v in expected.items():
        print(f"  {k:30s} = {v:+.6f}")
    print()

    all_payloads: list[dict] = []
    for ckpt in targets:
        ckpt = Path(ckpt)
        try:
            sa_payload = evaluate_spin_amplitude_entanglement(ckpt, device=args.device)
        except Exception as exc:
            print(f"[ERROR] {ckpt}: {exc}")
            continue
        try:
            dot_payload = evaluate_n2_entanglement(ckpt, device=args.device)
        except Exception as exc:
            dot_payload = {
                "dot_projected_entanglement": {
                    "von_neumann_entropy": float("nan"),
                    "dot_label_partial_transpose": {
                        "negativity": float("nan"),
                        "log_negativity": float("nan"),
                    },
                }
            }
            LOGGER.warning("dot-label evaluation failed for %s: %s", ckpt, exc)

        print("=" * 72)
        print(_format_payload(ckpt, sa_payload, dot_payload))
        print()
        all_payloads.append({
            "checkpoint": str(ckpt),
            "spin_amplitude": sa_payload,
            "dot_label": dot_payload,
        })

    if args.out_json is not None:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(all_payloads, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        print(f"Wrote payload to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

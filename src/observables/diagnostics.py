from __future__ import annotations

from typing import Any

import numpy as np


def summarize_training_diagnostics(
    history: dict[str, Any], *, n_coll: int, sampler: str
) -> dict[str, Any]:
    """Compute compact diagnostics from the training history."""
    energy = np.asarray(history.get("energy", []), dtype=float)
    loss = np.asarray(history.get("loss", []), dtype=float)
    ess = np.asarray(history.get("ess", []), dtype=float)
    out: dict[str, Any] = {
        "n_epochs": int(len(loss)),
        "n_coll": int(n_coll),
        "sampler": str(sampler),
    }
    if energy.size:
        window = min(100, energy.size)
        out["energy_last"] = float(energy[-1])
        out["energy_mean_last_window"] = float(energy[-window:].mean())
        out["energy_std_last_window"] = float(energy[-window:].std(ddof=0))
    if loss.size:
        out["loss_last"] = float(loss[-1])
    if ess.size:
        out["ess_last"] = float(ess[-1])
        out["ess_mean"] = float(ess.mean())
    return out

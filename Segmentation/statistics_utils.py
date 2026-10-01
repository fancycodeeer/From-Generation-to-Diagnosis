from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy import stats


def load_weights(model, weight_path: str, keys: Iterable[str]) -> None:
    import torch

    if not weight_path:
        raise ValueError("Weight path is empty.")
    path = Path(weight_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in keys:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Cannot find a state_dict in {path}")
    load_error = None
    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError as error:
        load_error = error
    for prefix in ("module.", "model.", "student.", "translator.", "generator.", "netG.", "G_A."):
        if state and all(str(name).startswith(prefix) for name in state):
            stripped = {str(name)[len(prefix):]: value for name, value in state.items()}
            try:
                model.load_state_dict(stripped, strict=True)
                return
            except RuntimeError:
                continue
    raise load_error


def paired_statistics(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    id_column: str,
    metrics: Dict[str, str],
    method: str = "paired_t",
    bootstrap_resamples: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    for name, frame in (("candidate", candidate), ("reference", reference)):
        if id_column not in frame:
            raise ValueError(f"{name} output has no {id_column!r} column")
        if frame[id_column].astype(str).duplicated().any():
            raise ValueError(f"{name} output contains duplicate {id_column} values")

    candidate = candidate.copy()
    reference = reference.copy()
    candidate[id_column] = candidate[id_column].astype(str)
    reference[id_column] = reference[id_column].astype(str)
    left_ids = set(candidate[id_column])
    right_ids = set(reference[id_column])
    if left_ids != right_ids:
        raise ValueError(
            f"Outputs are not paired: candidate-only={sorted(left_ids-right_ids)[:5]}, "
            f"reference-only={sorted(right_ids-left_ids)[:5]}"
        )

    merged = candidate.merge(
        reference,
        on=id_column,
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    rows: List[dict] = []
    for display_name, column in metrics.items():
        left_col = f"{column}_candidate"
        right_col = f"{column}_reference"
        if left_col not in merged or right_col not in merged:
            raise ValueError(f"Metric column {column!r} is missing")
        left = pd.to_numeric(merged[left_col], errors="coerce").to_numpy(float)
        right = pd.to_numeric(merged[right_col], errors="coerce").to_numpy(float)
        valid = np.isfinite(left) & np.isfinite(right)
        if not valid.all() and display_name != "Conditional mean localization error (px)":
            raise ValueError(f"{display_name} contains missing paired values")
        left = left[valid]
        right = right[valid]
        delta = left - right
        if len(delta) < 2:
            raise ValueError(f"{display_name} has fewer than two valid paired cases")

        row = {
            "metric": display_name,
            "n": len(delta),
            "candidate_mean": float(left.mean()),
            "candidate_sd": float(left.std(ddof=1)),
            "reference_mean": float(right.mean()),
            "reference_sd": float(right.std(ddof=1)),
            "mean_delta": float(delta.mean()),
            "mean_delta_ci_low": np.nan,
            "mean_delta_ci_high": np.nan,
            "median_delta": np.nan,
            "median_delta_ci_low": np.nan,
            "median_delta_ci_high": np.nan,
            "p_value": np.nan,
            "test": method,
        }
        if method == "paired_t":
            sem = float(stats.sem(delta))
            margin = float(stats.t.ppf(0.975, len(delta) - 1) * sem)
            row["mean_delta_ci_low"] = float(delta.mean() - margin)
            row["mean_delta_ci_high"] = float(delta.mean() + margin)
            row["p_value"] = 1.0 if np.all(delta == 0) else float(stats.ttest_rel(left, right).pvalue)
        elif method == "wilcoxon":
            p_value = 1.0 if np.all(delta == 0) else float(stats.wilcoxon(delta, alternative="two-sided").pvalue)
            rng = np.random.default_rng(seed)
            indices = rng.integers(0, len(delta), size=(bootstrap_resamples, len(delta)))
            boot_medians = np.median(delta[indices], axis=1)
            low, high = np.percentile(boot_medians, [2.5, 97.5])
            row["median_delta"] = float(np.median(delta))
            row["median_delta_ci_low"] = float(low)
            row["median_delta_ci_high"] = float(high)
            row["p_value"] = p_value
        else:
            raise ValueError("method must be 'paired_t' or 'wilcoxon'")
        rows.append(row)
    return pd.DataFrame(rows)

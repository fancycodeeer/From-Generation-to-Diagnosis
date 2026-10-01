"""Evaluate two translation weights on the same test set and run paired statistics."""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from statistics_utils import paired_statistics


def case_id_from_key(value: str, pattern: str = "") -> str:
    stem = Path(str(value)).stem
    if pattern:
        match = re.search(pattern, stem)
        if match is None:
            raise ValueError(f"case_id_regex did not match {stem}")
        return match.group(1) if match.groups() else match.group(0)
    tokens = stem.split("_")
    if tokens and tokens[-1].isdigit():
        tokens = tokens[:-1]
    if tokens and tokens[-1].lower() in {"t1", "t1ce", "t2", "flair", "ct", "mr", "mri", "pd"}:
        tokens = tokens[:-1]
    if tokens and tokens[0].startswith("BraTS-"):
        return "-".join(tokens[0].split("-")[:4])
    if tokens and tokens[0].startswith("IXI"):
        return tokens[0].split("-")[0]
    if tokens and tokens[0].startswith("BraTS") and len(tokens) >= 2:
        return "_".join(tokens[:2])
    return "_".join(tokens[:2]) if len(tokens) >= 2 else tokens[0]


def run_test(args, weight: str, folder: Path) -> None:
    evaluator = {
        "cyclegan": "test.py",
        "cut": "test_cut.py",
        "syndiff": "test_syndiff.py",
    }[args.backbone]
    folder.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        evaluator,
        "--dataroot",
        str(Path(args.dataroot).expanduser().resolve()),
        "--phase",
        args.phase,
        "--mode",
        args.mode,
        "--ckpt",
        str(Path(weight).expanduser().resolve()),
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--metrics_csv",
        str(folder / "summary.csv"),
        "--slice_csv",
        str(folder / "slice.csv"),
        "--lpips_case_csv",
        str(folder / "lpips_case.csv"),
        "--lpips_epoch",
        "0",
    ]
    if args.a_dir:
        command += ["--a_dir", str(Path(args.a_dir).expanduser().resolve())]
    if args.b_dir:
        command += ["--b_dir", str(Path(args.b_dir).expanduser().resolve())]
    extra_args = args.extra_args[1:] if args.extra_args[:1] == ["--"] else args.extra_args
    command += extra_args
    subprocess.run(command, cwd=Path(__file__).parent, check=True)


def build_case_table(folder: Path, case_id_regex: str) -> pd.DataFrame:
    slices = pd.read_csv(folder / "slice.csv", dtype=str)
    if slices.empty:
        raise ValueError("Evaluation slice CSV is empty")

    if case_id_regex:
        if "key" not in slices:
            raise ValueError(
                "case_id_regex was specified, but evaluation slice CSV has no key column"
            )
        slices["case_id"] = slices["key"].map(
            lambda value: case_id_from_key(value, case_id_regex)
        )
    elif "case_id" not in slices:
        if "key" not in slices:
            raise ValueError("Evaluation slice CSV has neither case_id nor key")
        slices["case_id"] = slices["key"].map(case_id_from_key)

    if (
        slices["case_id"].isna().any()
        or slices["case_id"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Evaluation slice CSV contains missing case_id values")

    for column in ("ssim", "psnr"):
        slices[column] = pd.to_numeric(slices[column], errors="raise")

    slices["case_id"] = slices["case_id"].astype(str)

    cases = (
        slices.groupby("case_id", as_index=False)
        .agg(
            n_slices=("case_id", "size"),
            ssim_mean=("ssim", "mean"),
            psnr_mean=("psnr", "mean"),
        )
    )

    if "lpips" in slices:
        slices["lpips"] = pd.to_numeric(slices["lpips"], errors="raise")
        lpips = slices.groupby("case_id", as_index=False)["lpips"].mean()
        lpips = lpips.rename(columns={"lpips": "lpips_mean"})
    else:
        lpips = pd.read_csv(folder / "lpips_case.csv", dtype=str)

        if not {"case_id", "lpips_mean"}.issubset(lpips.columns):
            raise ValueError(
                "LPIPS case CSV must contain case_id and lpips_mean"
            )

        lpips = lpips[["case_id", "lpips_mean"]]

        if (
            lpips["case_id"].isna().any()
            or lpips["case_id"].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(
                "LPIPS case CSV contains missing case_id values"
            )

        lpips["case_id"] = lpips["case_id"].astype(str)

        if lpips["case_id"].duplicated().any():
            duplicates = (
                lpips.loc[
                    lpips["case_id"].duplicated(keep=False),
                    "case_id",
                ]
                .unique()
                .tolist()
            )
            raise ValueError(
                "LPIPS case CSV contains duplicate case_id values: "
                f"{duplicates[:5]}"
            )

        lpips["lpips_mean"] = pd.to_numeric(
            lpips["lpips_mean"],
            errors="raise",
        )

    lpips["case_id"] = lpips["case_id"].astype(str)

    case_ids = set(cases["case_id"])
    lpips_ids = set(lpips["case_id"])

    if case_ids != lpips_ids:
        raise ValueError(
            "SSIM/PSNR and LPIPS case IDs do not match: "
            f"metrics-only={sorted(case_ids - lpips_ids)[:5]}, "
            f"lpips-only={sorted(lpips_ids - case_ids)[:5]}"
        )

    return (
        cases.merge(
            lpips,
            on="case_id",
            validate="one_to_one",
        )
        .sort_values("case_id")
    )


def patient_average(cases: pd.DataFrame, map_path: str) -> pd.DataFrame:
    if not map_path:
        return cases

    mapping = pd.read_csv(map_path, dtype=str)

    if not {"case_id", "patient_id"}.issubset(mapping.columns):
        raise ValueError(
            "patient_map must contain case_id and patient_id"
        )

    cases = cases.copy()
    cases["case_id"] = cases["case_id"].astype(str)

    merged = cases.merge(
        mapping[["case_id", "patient_id"]],
        on="case_id",
        how="left",
        validate="one_to_one",
    )

    if merged["patient_id"].isna().any():
        raise ValueError(
            "patient_map does not cover every evaluated case"
        )

    return merged.groupby(
        "patient_id",
        as_index=False,
    )[
        [
            "ssim_mean",
            "psnr_mean",
            "lpips_mean",
        ]
    ].mean()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="")
    parser.add_argument("--candidate_weight", default="")
    parser.add_argument("--reference_weight", default="")
    parser.add_argument(
        "--backbone",
        choices=["cyclegan", "cut", "syndiff"],
        default="cyclegan",
    )
    parser.add_argument(
        "--phase",
        choices=["val", "test"],
        default="test",
    )
    parser.add_argument(
        "--mode",
        choices=["t1", "t2"],
        default="t1",
    )
    parser.add_argument("--a_dir", default="")
    parser.add_argument("--b_dir", default="")
    parser.add_argument(
        "--patient_map",
        default="",
        help="Optional case_id,patient_id CSV for external validation",
    )
    parser.add_argument("--case_id_regex", default="")
    parser.add_argument(
        "--method",
        choices=["paired_t", "wilcoxon"],
        default="paired_t",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["ssim", "psnr", "lpips"],
        default=["ssim", "psnr", "lpips"],
    )
    parser.add_argument(
        "--output_dir",
        default="statistics_output",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are passed to both test scripts",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if (
        not args.dataroot
        or not args.candidate_weight
        or not args.reference_weight
    ):
        raise ValueError(
            "Set --dataroot, --candidate_weight, and --reference_weight"
        )

    output = Path(
        args.output_dir
    ).expanduser().resolve()

    run_test(
        args,
        args.candidate_weight,
        output / "candidate",
    )

    run_test(
        args,
        args.reference_weight,
        output / "reference",
    )

    candidate = patient_average(
        build_case_table(
            output / "candidate",
            args.case_id_regex,
        ),
        args.patient_map,
    )

    reference = patient_average(
        build_case_table(
            output / "reference",
            args.case_id_regex,
        ),
        args.patient_map,
    )

    id_column = (
        "patient_id"
        if args.patient_map
        else "case_id"
    )

    metric_columns = {
        "ssim": ("SSIM", "ssim_mean"),
        "psnr": ("PSNR", "psnr_mean"),
        "lpips": ("LPIPS", "lpips_mean"),
    }

    selected = {
        metric_columns[name][0]:
            metric_columns[name][1]
        for name in args.metrics
    }

    result = paired_statistics(
        candidate,
        reference,
        id_column,
        selected,
        args.method,
    )

    candidate.to_csv(
        output / "candidate_case_metrics.csv",
        index=False,
    )

    reference.to_csv(
        output / "reference_case_metrics.csv",
        index=False,
    )

    result.to_csv(
        output / "statistics.csv",
        index=False,
    )

    print(
        result.to_string(index=False)
    )


if __name__ == "__main__":
    main()
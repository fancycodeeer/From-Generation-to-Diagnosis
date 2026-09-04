"""Load CenterNet and SFD weights, test them, and calculate patient-level statistics."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from statistics_utils import load_weights, paired_statistics

from datasets import DeepLesionHeatmapDataset, heatmap_collate
from models import UNetSECoord
from models_ import ResnetGenerator, get_norm_layer
from train import compute_batch_metrics, summarize_case_metrics


def sfd_input(images, translator):
    fake = translator(images.float().clamp(0, 1) * 2 - 1)
    if isinstance(fake, (tuple, list)):
        fake = fake[0]
    if isinstance(fake, dict):
        for key in ("fake", "fake_B", "out", "output", "image"):
            if key in fake:
                fake = fake[key]
                break
    if not isinstance(fake, torch.Tensor):
        raise RuntimeError("Translator did not return a tensor")
    if fake.shape[-2:] != images.shape[-2:]:
        fake = F.interpolate(fake, images.shape[-2:], mode="bilinear", align_corners=False)
    fake = ((fake.clamp(-1, 1) + 1) * 0.5).clamp(0, 1)
    return torch.cat([images.float().clamp(0, 1), fake], dim=1)


@torch.no_grad()
def evaluate(model, loader, device, args, translator=None):
    model.eval()
    if translator is not None:
        translator.eval()
    rows = []
    for images, heatmaps, metas in loader:
        images = images.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)
        with autocast(enabled=args.amp and device.type == "cuda"):
            model_input = images if translator is None else sfd_input(images, translator)
            probability = torch.sigmoid(model(model_input))
        metrics = compute_batch_metrics(probability.float(), heatmaps.float(), metas, args)
        for values, meta in zip(metrics, metas):
            row = {
                "case_id": str(meta["case_id"]),
                "patient_id": str(meta["patient_id"]),
                "series_id": str(meta.get("series_id", "")),
                "slice_id": str(meta.get("slice_id", "")),
            }
            row.update(values)
            rows.append(row)
    return summarize_case_metrics(pd.DataFrame(rows), args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="")
    parser.add_argument("--candidate_weight", default="", help="SFD student weight or joint checkpoint")
    parser.add_argument("--translator_weight", default="", help="SFD translator weight or joint checkpoint")
    parser.add_argument("--reference_weight", default="", help="CenterNet weight")
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pred_peak_threshold", type=float, default=0.8)
    parser.add_argument("--gt_peak_threshold", type=float, default=0.99)
    parser.add_argument("--peak_nms_kernel", type=int, default=9)
    parser.add_argument("--match_radius", type=float, default=15.0)
    parser.add_argument("--max_pred_peaks", type=int, default=3)
    parser.add_argument("--max_gt_peaks", type=int, default=3)
    parser.add_argument("--froc_min_peak_score", type=float, default=1e-4)
    parser.add_argument("--max_froc_pred_peaks", type=int, default=50)
    parser.add_argument("--froc_fp_targets", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--output_dir", default="statistics_output")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data_root or not args.candidate_weight or not args.translator_weight or not args.reference_weight:
        raise ValueError("Set data_root and all three weight paths")
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    dataset = DeepLesionHeatmapDataset(data_root=args.data_root, split="test")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=heatmap_collate,
    )
    reference = UNetSECoord(in_ch=1, out_ch=1, base=args.base).to(device)
    candidate = UNetSECoord(in_ch=2, out_ch=1, base=args.base).to(device)
    translator = ResnetGenerator(
        input_nc=1,
        output_nc=1,
        ngf=64,
        norm_layer=get_norm_layer("instance"),
        use_dropout=False,
        n_blocks=9,
    ).to(device)
    load_weights(reference, args.reference_weight, ("model", "state_dict", "model_state_dict"))
    load_weights(candidate, args.candidate_weight, ("student", "model", "state_dict"))
    load_weights(translator, args.translator_weight, ("translator", "G_A", "generator", "state_dict"))

    reference_cases = evaluate(reference, loader, device, args)
    candidate_cases = evaluate(candidate, loader, device, args, translator)
    result = paired_statistics(
        candidate_cases,
        reference_cases,
        "patient_id",
        {
            "Sensitivity at 0.5 FP/image": "sens_at_0_5fp",
            "Sensitivity at 1 FP/image": "sens_at_1fp",
            "Sensitivity at 2 FP/image": "sens_at_2fp",
            "KL divergence": "kl",
            "Conditional mean localization error (px)": "mean_match_dist_px",
        },
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidate_cases.to_csv(output / "candidate_patient_metrics.csv", index=False)
    reference_cases.to_csv(output / "reference_patient_metrics.csv", index=False)
    result.to_csv(output / "statistics.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

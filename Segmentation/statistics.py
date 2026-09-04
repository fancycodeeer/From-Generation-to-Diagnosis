"""Load two segmentation weights, test them, and calculate patient-level statistics."""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from statistics_utils import load_weights, paired_statistics

from dataset import BraTS2021BinarySegDataset
from models import UNetCoordinateChannelAttention
from models_ import ResnetGenerator, get_norm_layer


def generate_fake(images, translator):
    fake = translator(images)
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
    return fake.clamp(-1, 1)


@torch.no_grad()
def evaluate(model, loader, device, threshold, translator=None, amp=False):
    model.eval()
    if translator is not None:
        translator.eval()
    cases = defaultdict(lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0})
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with autocast(enabled=amp and device.type == "cuda"):
            if translator is None:
                logits = model(torch.cat([images, images], dim=1))
            else:
                fake = generate_fake(images, translator)
                logits = model(torch.cat([images, fake], dim=1))
        prediction = (torch.sigmoid(logits) >= threshold).float()
        for index, case_id in enumerate(batch["case_id"]):
            pred = prediction[index]
            target = masks[index]
            cases[str(case_id)]["tp"] += float((pred * target).sum())
            cases[str(case_id)]["fp"] += float((pred * (1 - target)).sum())
            cases[str(case_id)]["fn"] += float(((1 - pred) * target).sum())

    rows = []
    for case_id, counts in sorted(cases.items()):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        rows.append(
            {
                "case_id": case_id,
                "precision": tp / max(tp + fp, 1e-8),
                "recall": tp / max(tp + fn, 1e-8),
                "dice": 2 * tp / max(2 * tp + fp + fn, 1e-8),
                "iou": tp / max(tp + fp + fn, 1e-8),
            }
        )
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="")
    parser.add_argument("--candidate_weight", default="", help="SFD student weight or joint checkpoint")
    parser.add_argument("--translator_weight", default="", help="SFD translator weight or joint checkpoint")
    parser.add_argument("--reference_weight", default="")
    parser.add_argument("--test_phase", default="test")
    parser.add_argument("--modality", default="t1", choices=["t1", "t1ce", "t2", "flair"])
    parser.add_argument("--load_size", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--translator_ngf", type=int, default=64)
    parser.add_argument("--translator_n_blocks", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output_dir", default="statistics_output")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.dataroot or not args.candidate_weight or not args.translator_weight or not args.reference_weight:
        raise ValueError("Set dataroot and all three weight paths")
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    dataset = BraTS2021BinarySegDataset(
        dataroot=args.dataroot,
        phase=args.test_phase,
        modality=args.modality,
        load_size=args.load_size,
        crop_size=0,
        no_flip=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    reference = UNetCoordinateChannelAttention(
        in_channels=2, num_classes=1, base_channels=args.base_channels, bilinear=True, use_attention=True
    ).to(device)
    candidate = UNetCoordinateChannelAttention(
        in_channels=2, num_classes=1, base_channels=args.base_channels, bilinear=True, use_attention=True
    ).to(device)
    translator = ResnetGenerator(
        1,
        1,
        ngf=args.translator_ngf,
        norm_layer=get_norm_layer("instance"),
        use_dropout=False,
        n_blocks=args.translator_n_blocks,
    ).to(device)
    load_weights(reference, args.reference_weight, ("model", "state_dict", "model_state_dict"))
    load_weights(candidate, args.candidate_weight, ("student", "model", "state_dict"))
    load_weights(translator, args.translator_weight, ("translator", "G_A", "generator", "state_dict"))

    reference_cases = evaluate(reference, loader, device, args.threshold, amp=args.amp)
    candidate_cases = evaluate(candidate, loader, device, args.threshold, translator, args.amp)
    result = paired_statistics(
        candidate_cases,
        reference_cases,
        "case_id",
        {"Precision": "precision", "Recall": "recall", "Dice": "dice", "IoU": "iou"},
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidate_cases.to_csv(output / "candidate_case_metrics.csv", index=False)
    reference_cases.to_csv(output / "reference_case_metrics.csv", index=False)
    result.to_csv(output / "statistics.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

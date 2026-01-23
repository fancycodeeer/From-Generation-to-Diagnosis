import argparse
import os
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import CenterNetDataset, detection_collate_fn
from models import CenterNet
from models_t import Generator
from utils import top1_peak_coords
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ---------------- Seed control (deterministic) ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------- Metric computation ----------------
@torch.inference_mode()
def compute_metrics(model, loader, device, thresh=0.01):
    """Return (avg_kl, avg_euclidean_err). KL 逐样本平均，欧氏误差逐通道平均。"""
    total_kl, total_err = 0.0, 0.0
    sample_cnt, channel_cnt = 0, 0
    model.eval()

    for idx, (imgs, tgt) in enumerate(loader):
        imgs = imgs.to(device)
        hm_pred = model(imgs)
        hm_gt = torch.stack([t['heatmap'] for t in tgt]).to(device)

        # 保存
        gt = hm_gt[0, 0].cpu().detach().numpy()  # [fh,fw]
        pr = hm_pred[0, 0].cpu().detach().numpy()  # [fh,fw]
        # 自定义保存路径和文件名
        save_dir = "./heatmap_output/"
        os.makedirs(save_dir, exist_ok=True)
        filename_gt = "gt_{}.png".format(idx)

        plt.figure(figsize=(8, 4))
        plt.title("GT Heatmap")
        plt.imshow(gt, cmap='jet')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')

        # plt.subplot(1, 2, 2)
        # plt.title("CenterNet Predicted Heatmap")
        # plt.imshow(pr, cmap='jet')
        # plt.colorbar(fraction=0.046, pad=0.04)
        # plt.axis('off')

        # plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename_gt), dpi=1400, bbox_inches='tight', pad_inches=0)

        B, C, _, _ = hm_pred.shape
        sample_cnt += B
        channel_cnt += B * C

        # KL divergence per sample (batchmean)
        pred_flat = hm_pred.view(B, -1)
        gt_flat   = hm_gt.view(B, -1)
        pred_dist = pred_flat / (pred_flat.sum(dim=1, keepdim=True) + 1e-6)
        gt_dist   = gt_flat   / (gt_flat.sum(dim=1, keepdim=True) + 1e-6)
        kl = F.kl_div(pred_dist.log(), gt_dist, reduction='batchmean')
        total_kl += kl.item()

        # Euclidean distance of top‑1 peaks per channel
        coords_pred, _ = top1_peak_coords(hm_pred, thresh=thresh)
        coords_gt, _   = top1_peak_coords(hm_gt,   thresh=thresh)
        dists = torch.norm(coords_pred.float() - coords_gt.float(), dim=2)  # (B, C)
        total_err += dists.sum().item()

    avg_kl  = total_kl / sample_cnt
    avg_err = total_err / channel_cnt
    return avg_kl, avg_err

# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_img', default='./data/val/images')
    parser.add_argument('--val_ann', default='./data/val/annotations')
    parser.add_argument('--weights_dir', default='./logs/detect/', help='Directory containing *.pth weights')
    parser.add_argument('--pattern', default='*.pth', help='Glob pattern to match weight files')
    parser.add_argument('--threshold', type=float, default=0.05, help='Peak threshold, keep same as training')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset / loader (no shuffle)
    val_ds = CenterNetDataset(args.val_img, args.val_ann)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4,
                            collate_fn=detection_collate_fn)

    weight_files = sorted(glob.glob(os.path.join(args.weights_dir, args.pattern)))
    if not weight_files:
        print(f"No weight files found in {args.weights_dir} with pattern {args.pattern}")
        return

    best_kl_val, best_kl_file = float('inf'), None
    best_err_val, best_err_file = float('inf'), None

    print(f"Evaluating {len(weight_files)} weight files...\n")
    for wf in weight_files:
        try:
            # Model skeleton once, reuse
            model = CenterNet(num_classes=1).to(device)
            state = torch.load(wf, map_location=device)
            model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"[Skip] {wf}: load error {e}")
            continue

        # Model skeleton once, reuse
        # model = CenterNet(num_classes=1).to(device)
        # state = torch.load('./logs/detect/best_kl.pth', map_location=device)
        # model.load_state_dict(state, strict=False)

        avg_kl, avg_err = compute_metrics(model, val_loader, device, args.threshold)
        print(avg_kl)
        print(f"{os.path.basename(wf):<25} KL={avg_kl:.4f} | Err={avg_err:.4f}")

        if avg_kl < best_kl_val:
            best_kl_val, best_kl_file = avg_kl, wf
        if avg_err < best_err_val:
            best_err_val, best_err_file = avg_err, wf

    print("\n===== Best Results =====")
    print(f"Best KL  : {best_kl_val:.4f} -> {best_kl_file}")
    print(f"Best Err : {best_err_val:.4f} -> {best_err_file}")

if __name__ == '__main__':
    main()

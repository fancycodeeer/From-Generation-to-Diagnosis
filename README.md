# Frequency-consistent Synthesis and Selective Transfer for Medical Image Analysis

Training and testing code for the paper *Frequency-consistent synthesis and selective transfer for medical image analysis*.

## 1. Code Structure

```text
From-Generation-to-Diagnosis/
├── FDDT/               # Frequency-domain image translation and AFD
├── Segmentation/       # Brain tumour segmentation and SFD
├── Localization/       # DeepLesion centre-point detection and SFD
├── requirements.txt
└── README.md
```

## 2. Installation

```bash
git clone https://github.com/fancycodeeer/From-Generation-to-Diagnosis.git
cd From-Generation-to-Diagnosis
pip install -r requirements.txt
```

## 3. Training

The following commands provide representative training configurations for FDDT, segmentation, and lesion localization. Dataset paths and pretrained model paths should be set according to the local environment.

### 3.1 FDDT + AFD — Image Translation

Example: BraTS 2021, T1 → T1ce.

```bash
python FDDT/train_fddt.py \
  --dataroot ./data/BraTS2021 \
  --checkpoints_dir ./checkpoints \
  --name brats_t1_t1ce \
  --source_domain t1 \
  --target_domain t1ce \
  --eval_phase val \
  --eval_pairing key \
  --case_id_regex "^(BraTS2021_[^_]+)" \
  --load_size 256 \
  --batch_size 1 \
  --num_workers 4 \
  --lr 0.0001 \
  --ngf 64 \
  --ndf 64 \
  --netG_blocks 9 \
  --n_epochs 100 \
  --n_epochs_decay 100 \
  --beta1 0.5 \
  --beta2 0.999 \
  --lambda_A 10 \
  --lambda_B 10 \
  --lambda_identity 0.1 \
  --lambda_freq 1.0 \
  --afd_embed_dim 64 \
  --afd_num_heads 4 \
  --afd_high_min 6 \
  --afd_high_max 40 \
  --afd_low_min 6 \
  --afd_low_max 40 \
  --afd_input_size 256 \
  --fddt_bidir \
  --early_metric both \
  --patience 50 \
```

### 3.2 Brain Tumour Segmentation — Baseline

Example: BraTS 2021, T1 MRI.

```bash
python Segmentation/train.py \
  --dataroot ./data/BraTS2021 \
  --train_phase train \
  --eval_phase val \
  --modality t1 \
  --load_size 0 \
  --crop_size 0 \
  --epochs 300 \
  --batch_size 2 \
  --num_workers 4 \
  --lr 0.0001 \
  --base_channels 64 \
  --threshold 0.5 \
  --patience 50 \
  --save_dir ./checkpoints/segmentation_baseline
```
### 3.3 Brain Tumour Segmentation — SFD

Example: BraTS 2021, T1 → T1ce.

```bash
python Segmentation/sfd.py \
  --dataroot ./data/BraTS2021 \
  --translator_ckpt ./pretrained/brats_seg900_t1_to_t1ce_G.pth \
  --translation_target_root ./data/BraTS2021/train/t1ce \
  --save_dir ./checkpoints/segmentation_sfd \
  --train_phase train \
  --val_phase val \
  --test_phase test \
  --modality t1 \
  --load_size 0 \
  --crop_size 0 \
  --epochs 300 \
  --batch_size 2 \
  --eval_batch_size 2 \
  --num_workers 4 \
  --lr 0.0001 \
  --lr_d 0.0001 \
  --base_channels 64 \
  --translator_ngf 64 \
  --translator_n_blocks 9 \
  --translator_norm instance \
  --feature_layers inc down1 down2 \
  --lambda_distill 0.5 \
  --kd_warmup_epochs 20 \
  --kd_ramp_epochs 30 \
  --sfd_margin 0.0 \
  --lambda_trans 0.05 \
  --threshold 0.5 \
  --patience 50 \
```

### 3.4 DeepLesion Detection — Baseline

```bash
python Localization/train.py \
  --data_root ./data/DeepLesion \
  --out_dir ./checkpoints/deeplesion_baseline \
  --epochs 500 \
  --batch_size 3 \
  --eval_batch_size 8 \
  --num_workers 8 \
  --base 64 \
  --lr 0.0001 \
  --weight_decay 0.0001 \
  --hard_bg_fraction 0.003 \
  --hard_bg_min_k 128 \
  --hard_bg_max_k 1024 \
  --min_peak_pixels 9 \
  --pred_peak_threshold 0.8 \
  --gt_peak_threshold 0.99 \
  --peak_nms_kernel 9 \
  --match_radius 15 \
  --max_pred_peaks 3 \
  --max_gt_peaks 3 \
  --froc_min_peak_score 0.0001 \
  --max_froc_pred_peaks 50 \
  --froc_fp_targets 0.5 1 2 \
  --monitor case_sens_at_1fp \
  --patience 30 \
```

### 3.5 DeepLesion Detection — SFD

```bash
python Localization/train_sfd.py \
  --data_root ./data/DeepLesion \
  --out_dir ./checkpoints/deeplesion_sfd \
  --translator_ckpt ./pretrained/synthrad_ct_to_mr_G.pth \
  --translation_target_root ./data/SynthRAD2025/train/mr \
  --epochs 500 \
  --batch_size 3 \
  --eval_batch_size 8 \
  --num_workers 8 \
  --base 64 \
  --lr 0.0001 \
  --lr_g 0.0001 \
  --lr_d 0.0001 \
  --weight_decay 0.0001 \
  --weight_decay_g 0.0 \
  --hard_bg_fraction 0.003 \
  --hard_bg_min_k 128 \
  --hard_bg_max_k 1024 \
  --min_peak_pixels 9 \
  --distill_mode sfd \
  --feature_layers \
    up4.conv.se.net.1 \
    up4.conv.se.net.3 \
    up4.conv.coord.conv1 \
    up4.conv.coord.conv_h \
    up4.conv.coord.conv_w \
  --lambda_feat_kd 0.5 \
  --kd_warmup_epochs 20 \
  --kd_ramp_epochs 30 \
  --sfd_margin 0.02 \
  --kd_fg_weight 1.0 \
  --kd_bg_weight 0.03 \
  --kd_peak_weight 2.0 \
  --kd_peak_rel_thr 0.7 \
  --kd_teacher_attn_weight 0.5 \
  --sfd_risk_kl_weight 0.03 \
  --sfd_risk_mse_weight 0.05 \
  --sfd_risk_peak_weight 1.0 \
  --lambda_trans 0.05 \
  --gan_mode lsgan \
  --pred_peak_threshold 0.8 \
  --gt_peak_threshold 0.99 \
  --peak_nms_kernel 9 \
  --match_radius 15 \
  --max_pred_peaks 3 \
  --max_gt_peaks 3 \
  --froc_min_peak_score 0.0001 \
  --max_froc_pred_peaks 50 \
  --froc_fp_targets 0.5 1 2 \
  --monitor case_sens_at_1fp \
  --patience 30 \
```

# Frequency-consistent Synthesis and Selective Transfer for Medical Image Analysis

Training and testing code for the paper *Frequency-consistent synthesis and selective transfer for medical image analysis*.

## 1. Code Structure

```text
From-Generation-to-Diagnosis/
├── FDDT/               # Image translation with FDDT and AFD
├── Segmentation/       # Brain tumour segmentation and SFD
├── Localization/       # DeepLesion centre-point detection and SFD
├── manifest/           # Dataset partition records
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

The following commands provide one training example for each experiment. Replace dataset and checkpoint paths with the corresponding local paths. Numerical hyperparameters are example settings and should be replaced with the original configurations for exact reproduction of the reported results.

### 3.1 FDDT + AFD — Image Translation

Example: BraTS 2021, T1 → T1ce.

```bash
python FDDT/train_fddt.py \
  --dataroot ./data/BraTS2021 \
  --checkpoints_dir ./checkpoints \
  --name brats_t1_t1ce \
  --source_domain t1 \
  --target_domain t1ce \
  --case_id_regex "^(BraTS2021_[^_]+)" \
  --load_size 256 \
  --batch_size 1 \
  --num_workers 2 \
  --lambda_freq 1.0 \
  --afd_embed_dim 64 \
  --afd_num_heads 4 \
  --afd_high_min 14 \
  --afd_high_max 16 \
  --afd_low_min 12 \
  --afd_low_max 13.5 \
  --afd_input_size 256
```

### 3.2 Brain Tumour Segmentation with SFD

Example: BraTS 2021, T1 → T1ce.

```bash
python Segmentation/sfd.py \
  --dataroot ./data/BraTS2021 \
  --translator_ckpt ./checkpoints/brats_t1_t1ce/best_ssim_G_t1.pth \
  --save_dir ./checkpoints/segmentation_sfd \
  --translation_target_root ./data/BraTS2021/train/t1ce \
  --train_phase train \
  --val_phase val \
  --test_phase test \
  --modality t1 \
  --load_size 256 \
  --epochs 300 \
  --batch_size 2 \
  --eval_batch_size 2 \
  --num_workers 4 \
  --base_channels 64 \
  --translator_ngf 64 \
  --translator_n_blocks 9 \
  --feature_layers inc down1 down2 \
  --lambda_distill 0.5 \
  --kd_warmup_epochs 20 \
  --kd_ramp_epochs 30 \
  --sfd_margin 0.0 \
  --lambda_trans 0.05
```

### 3.3 DeepLesion Detection with SFD

```bash
python Localization/train_sfd.py \
  --data_root ./data/DeepLesion \
  --out_dir ./checkpoints/deeplesion_sfd \
  --translator_ckpt ./pretrained/synthrad_ct2mr_G.pth \
  --translation_target_root ./data/SynthRAD_MR \
  --epochs 500 \
  --batch_size 3 \
  --eval_batch_size 8 \
  --num_workers 4 \
  --base 64 \
  --lr 0.0001 \
  --lr_g 0.0001 \
  --weight_decay_g 0.0 \
  --hard_bg_fraction 0.003 \
  --hard_bg_min_k 128 \
  --hard_bg_max_k 1024 \
  --min_peak_pixels 9 \
  --distill_mode sfd \
  --feature_layers inc down1 down2 \
  --lambda_feat_kd 0.5 \
  --sfd_margin 0.02 \
  --kd_fg_weight 1.0 \
  --kd_bg_weight 0.03 \
  --kd_peak_weight 2.0 \
  --kd_peak_rel_thr 0.7 \
  --kd_teacher_attn_weight 0.5
```

## 4. Statistical Analysis

The `statistics_utils.py` scripts calculate unadjusted p-values, effect estimates, and pointwise confidence intervals. They do not perform Holm multiple-comparison correction.

Holm correction for the manuscript was applied separately within the complete comparison families of Supplementary Tables S1–S6, containing 36, 12, 8, 36, 8, and 5 comparisons, respectively.

Correction must be applied to all raw p-values within each complete table-wise family, rather than separately to individual two-model comparisons.

Statistical significance was defined as Holm-adjusted p < 0.05.

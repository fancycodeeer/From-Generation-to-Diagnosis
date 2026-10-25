# Frequency-consistent Synthesis and Selective Transfer for Medical Image Analysis

Official implementation of the paper **"Frequency-consistent synthesis and selective transfer for medical image analysis"**.

## 1. Code Structure

```text
From-Generation-to-Diagnosis/
├── FDDT/               # Frequency-domain image translation and AFD
├── Segmentation/       # Brain tumour segmentation
├── Localization/       # DeepLesion detection and SFD
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

Run the following commands from the repository root.

Paths must be adjusted to the locally prepared datasets and pretrained checkpoints. Numerical hyperparameters below are example configurations; reproducing the paper requires the original experimental settings.

### 3.1 FDDT + AFD

BraTS 2021 T1 → T1ce:

```bash
python FDDT/train_fddt.py --dataroot ./data/BraTS2021 --checkpoints_dir ./checkpoints --name brats_t1_t1ce --source_domain t1 --target_domain t1ce --case_id_regex "^(BraTS2021_[^_]+)" --load_size 256 --lambda_freq 1.0 --afd_embed_dim 64 --afd_num_heads 4 --afd_high_min 14 --afd_high_max 16 --afd_low_min 12 --afd_low_max 13.5 --afd_input_size 256
```

BraTS 2021 T2 → FLAIR:

```bash
python FDDT/train_fddt.py --dataroot ./data/BraTS2021 --checkpoints_dir ./checkpoints --name brats_t2_flair --source_domain t2 --target_domain flair --case_id_regex "^(BraTS2021_[^_]+)" --load_size 256 --lambda_freq 1.0 --afd_embed_dim 64 --afd_num_heads 4 --afd_high_min 14 --afd_high_max 16 --afd_low_min 12 --afd_low_max 13.5 --afd_input_size 256
```

IXI PD → T2 (Fig. 3, AFD):

```bash
python FDDT/train_fddt.py --dataroot ./data/IXI --checkpoints_dir ./checkpoints --name ixi_pd_t2_afd --source_domain pd --target_domain t2 --case_id_regex "^(IXI[^_]+)" --load_size 256 --lambda_freq 1.0 --afd_embed_dim 64 --afd_num_heads 4 --afd_high_min 14 --afd_high_max 16 --afd_low_min 12 --afd_low_max 13.5 --afd_input_size 256
```

IXI T2 → PD (Fig. 3, AFD):

```bash
python FDDT/train_fddt.py --dataroot ./data/IXI --checkpoints_dir ./checkpoints --name ixi_t2_pd_afd --source_domain t2 --target_domain pd --case_id_regex "^(IXI[^_]+)" --load_size 256 --lambda_freq 1.0 --afd_embed_dim 64 --afd_num_heads 4 --afd_high_min 14 --afd_high_max 16 --afd_low_min 12 --afd_low_max 13.5 --afd_input_size 256
```

**Fig. 3:** Fixed-threshold FDDT uses predefined frequency cutoffs, whereas FDDT + AFD uses learned image-adaptive cutoffs. The current `train_fddt.py` provides the AFD implementation; the original fixed-threshold implementation and configuration are required to reproduce that comparison.

### 3.2 Brain Tumour Segmentation

```bash
python Segmentation/train.py --dataroot ./data/BraTS2021 --train_phase train --eval_phase val --modality t1 --epochs 300 --batch_size 2 --num_workers 4 --lr 0.0001 --base_channels 64 --seed 42 --save_dir ./checkpoints/segmentation
```

### 3.3 DeepLesion Baseline

```bash
python Localization/train.py --data_root ./data/DeepLesion --out_dir ./checkpoints/deeplesion_baseline --epochs 500 --batch_size 3 --base 64 --lr 0.0001 --hard_bg_fraction 0.003 --hard_bg_min_k 128 --hard_bg_max_k 1024 --min_peak_pixels 9
```

### 3.4 DeepLesion + SFD

Requires a pretrained CT-to-MR generator checkpoint and SynthRAD target-domain images.

```bash
python Localization/train_sfd.py --data_root ./data/DeepLesion --out_dir ./checkpoints/deeplesion_sfd --translator_ckpt ./pretrained/synthrad_ct2mr_G.pth --translation_target_root ./data/SynthRAD_MR --epochs 500 --batch_size 3 --base 64 --lr 0.0001 --lr_g 0.0001 --weight_decay_g 0.0 --hard_bg_fraction 0.003 --hard_bg_min_k 128 --hard_bg_max_k 1024 --min_peak_pixels 9 --distill_mode sfd --feature_layers inc down1 down2 --lambda_feat_kd 0.5 --sfd_margin 0.02 --kd_fg_weight 1.0 --kd_bg_weight 0.03 --kd_peak_weight 2.0 --kd_peak_rel_thr 0.7 --kd_teacher_attn_weight 0.5
```

## 4. Statistical Analysis

The `statistics_utils.py` scripts calculate unadjusted p-values, effect estimates, and pointwise confidence intervals. They do not perform Holm multiple-comparison correction.

The Holm-adjusted p-values reported in the manuscript are calculated separately within the complete comparison families defined by Supplementary Tables S1–S6 (36, 12, 8, 36, 8, and 5 comparisons, respectively).

Holm correction must be applied to all raw p-values within each complete table-wise family, rather than separately to individual two-model comparisons.

Statistical significance is defined as Holm-adjusted p < 0.05.

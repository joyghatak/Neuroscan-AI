# NeuroScan AI v3.0 — Brain MRI Tumour Segmentation and Classification

Generated 2026-09-13T09:38:58 · dataset: BRISC2025 (briscdataset/brisc2025)

## Design

| Component | Detail |
|---|---|
| Segmenter | Attention U-Net (5-level encoder, attention gate on every skip) |
| Segmentation supervision | annotated masks shipped with the dataset |
| Classifier | EfficientNet-B0 + Linear-BN-ReLU-Dropout head |
| Fine-tuning | 3 stages (head → top-20 → top-50 non-BN modules), cosine LR, real gradient clipping |
| Evaluation | Official held-out test folder, n=995, untouched during training |
| Uncertainty | MC dropout, 30 passes, BatchNorm frozen |
| Robustness | Deterministic TTA bank, 10 transforms |
| Fusion | Weights fitted on validation: 0.00/0.75/0.25 |

## Segmentation results

| Metric | Validation | Held-out test |
|---|---|---|
| Dice | 0.8493 | 0.8400 |
| IoU | 0.7884 | 0.7756 |

## Controlled classification comparison

Both arms share one architecture, seed, schedule and monitor. The only
difference is the input crop, so the delta is attributable to ROI guidance.

| Metric | full_image (control) | roi_guided | Delta |
|---|---|---|---|
| accuracy | 0.9859 | 0.9749 | -0.0111 |
| balanced_accuracy | 0.9871 | 0.9761 | -0.0111 |
| macro_f1 | 0.9862 | 0.9756 | -0.0106 |
| macro_recall | 0.9871 | 0.9761 | -0.0111 |
| cohen_kappa | 0.9808 | 0.9657 | -0.0151 |

McNemar exact test on paired test items: 25 discordant pairs, p = 0.0433 — the difference is **significant** at alpha = 0.05.

Deployed arm: **full_image**. With the fitted ensemble the held-out test accuracy is 98.79% and macro F1 is 98.84%.

## Reproducing

Seed 42, deterministic cuDNN. Run the notebook top to bottom; every
hyperparameter lives in the CFG dataclass in the configuration cell.

## Intended use

Research and benchmarking only. Not a medical device and not validated for
clinical use. Outputs derive from single 2D slices.

## Artefacts

- `artifacts/checkpoints/classifier_full_image.pt` — Final classifier for arm 'full_image'
- `artifacts/checkpoints/classifier_roi_guided.pt` — Final classifier for arm 'roi_guided'
- `artifacts/checkpoints/full_image_stageA.pt` — Arm 'full_image' stage A best weights
- `artifacts/checkpoints/full_image_stageB.pt` — Arm 'full_image' stage B best weights
- `artifacts/checkpoints/full_image_stageC.pt` — Arm 'full_image' stage C best weights
- `artifacts/checkpoints/roi_guided_stageA.pt` — Arm 'roi_guided' stage A best weights
- `artifacts/checkpoints/roi_guided_stageB.pt` — Arm 'roi_guided' stage B best weights
- `artifacts/checkpoints/roi_guided_stageC.pt` — Arm 'roi_guided' stage C best weights
- `artifacts/checkpoints/segmenter.pt` — Attention U-Net, best val Dice
- `artifacts/figures/arm_comparison.png` — Controlled arm comparison
- `artifacts/figures/arm_curves.png` — Per-arm validation curves with stage boundaries
- `artifacts/figures/segmentation_curves.png` — Attention U-Net training curves
- `artifacts/figures/test_panels.png` — Confusion matrices and ROC on the held-out test set
- `artifacts/metrics/comparison.json` — Arm comparison and significance test
- `artifacts/metrics/roi_boxes.json` — Cached ROI bounding boxes (one per image)
- `artifacts/metrics/run_summary.json` — Complete machine-readable run summary
- `artifacts/metrics/segmentation.json` — Segmentation history and test metrics
- `artifacts/reports/report_glioma.pdf` — Clinical PDF report — glioma
- `artifacts/reports/report_meningioma.pdf` — Clinical PDF report — meningioma
- `artifacts/reports/report_no_tumor.pdf` — Clinical PDF report — no_tumor
- `artifacts/reports/report_pituitary.pdf` — Clinical PDF report — pituitary
- `artifacts/xai/unified_glioma.png` — Five-panel unified explanation — glioma
- `artifacts/xai/unified_meningioma.png` — Five-panel unified explanation — meningioma
- `artifacts/xai/unified_no_tumor.png` — Five-panel unified explanation — no_tumor
- `artifacts/xai/unified_pituitary.png` — Five-panel unified explanation — pituitary

# Research protocol

1. Freeze official signer-disjoint train/validation/test partitions before model selection.
2. Report dataset version, source, licensed use, class counts, exclusions, signer counts, seed, hardware and preprocessing.
3. Train a majority-class baseline, an image-only model and the landmark temporal model under comparable compute. Treat fusion as a separate ablation.
4. Select hyperparameters on validation only. Run the untouched test set once after model selection.
5. Record top-1 and balanced accuracy, macro precision/recall/F1, per-class recall, confusion matrix, and model latency (p50/p95).
6. Evaluate robustness under lighting, motion blur, occlusion and missing-hand frames using documented perturbation levels.
7. Compare multiple seeds, report confidence intervals and inspect signer-level failures.
8. Report any data leak, out-of-distribution failure and training cost honestly. Never invent test metrics or checkpoints.

This repository ships no measured model accuracy until actual held-out experiments are run.

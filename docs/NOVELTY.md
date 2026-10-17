# Research contribution and validation design

**Proposed working title:** Reliable isolated-sign recognition with tracker-failure robustness and uncertainty-aware abstention.

The codebase contains proof-of-method components, not a validated claim of novelty.

## Controlled hypotheses

- H1: explicit tracker-failure simulation improves worst-case recall under detection dropout, holding dataset, class split and encoder constant.
- H2: signer-disjoint split-conformal prediction sets improve selective usability compared with one fixed softmax threshold.
- H3: retaining body-relative trajectories improves motion-sensitive signs over wrist-only features.
- H4: calibrated few-shot prototype retrieval can identify new *classes* from a small support set; use a sign-disjoint query split.

## Required comparisons

- Single-hand image EfficientNetB0 baseline
- Hands-only temporal Conv+BiGRU baseline
- Whole-body + velocity/bones features with comparable parameter count
- Landmark dropout on/off, time warp on/off
- Argmax vs temperature scaling vs split conformal abstention
- Closed-set vs sign-disjoint few-shot retrieval

Report top-1/top-5, macro-F1, MRR, worst-signer accuracy, ECE, prediction set coverage/size, latency p50/p95, model bytes, and robustness curves. Use official signer splits for closed-set experiments; use strictly disjoint sign classes for open-vocabulary experiments.

**Do not use ASL-LEX phonological metadata without confirming a valid gloss mapping and dataset/license compatibility.** This is an optional future branch, not a trained feature. A phonology prediction head is not included until annotation quality is verified.

No benchmark outcomes or on-device/browser inference performance have been measured.

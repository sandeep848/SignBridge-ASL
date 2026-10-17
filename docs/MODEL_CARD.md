# Model card (untrained)

**Task:** static fingerspelling or isolated-sign classification; not continuous ASL sentence translation.
**Input:** RGB images or 32-frame streams of two-hand 21-joint normalized landmarks with visibility indicators.
**Outputs:** class probabilities for a label space established by each training dataset.
**Limitations:** missing hands, occlusion, motion-dependent signs, camera framing and signer generalization; testing required across different signers and backgrounds.
**Ethical use:** assistive research, not a replacement for qualified interpreters.
**Status:** code available, training and external validation pending. No model weights or numerical performance claims are published.

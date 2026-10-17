# Public reference release gate

Do not tag a production release or publish accuracy claims until all items below are documented.

- [ ] Clean Python 3.11 install on supported OS, dependency locking, CI and model import tests passed
- [ ] Training, validation, calibration and test splits are provenance-checked, signer-disjoint where appropriate, and frozen
- [ ] Entire training run repeated with at least three seeds and checkpoint hashes
- [ ] Compare independent baselines and all novelty ablations at matched splits and budget
- [ ] Report accuracy/F1/MRR/calibration/coverage/worst-group and bootstrap CIs on untouched test set
- [ ] Verify video feature schema, label mapping and preprocessing in runtime checkpoint loader
- [ ] On-device/browser inference, quantization and benchmark claims independently tested on target hardware
- [ ] Public API threat model and privacy review completed
- [ ] Deaf-community consultation and any necessary ethics approval documented
- [ ] Contributor/license rights confirmed; dataset and weight redistribution permissions checked
- [ ] Accessibility checks, user instructions, reproducible commands and release notes completed

These gates are not equivalent to coding tasks. Do not check them without real evidence.

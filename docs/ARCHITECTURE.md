# System architecture and trust boundaries

## Independent experimental tasks

1. **Static fingerspelling**: approved class directories → optional hand-crop preprocessing → EfficientNet baseline / image+landmark fusion → validation → untouched static test set.
2. **Closed-set isolated sign**: signer-grouped manifest → per-video MediaPipe landmarks → hands-only Conv+BiGRU / holistic research encoder → validation → untouched isolated-sign test set.
3. **Few-shot retrieval**: separately learned embeddings + consented reference library → cosine prototypes → sign-disjoint retrieval evaluation. The index is implemented but is **not** supplied with trained embeddings.
4. **Uncertainty**: calibration probabilities from separate held-out signers → split-conformal score threshold → prediction set; empirical coverage must be reported on separate test signers.

## Checkpoint contracts

Hands-only schema is `signbridge-two-hand-v2`, 32 frames, 128 features. Holistic schema is `holistic-body-v1`, 187 raw features/frame, 370 derived features/frame. These are incompatible by design. All new training runs should write a model metadata file and the manifest checksum. Do not load anonymous weights from untrusted sources.

## Deployment boundary

The default app uses a local FastAPI service for camera/image or uploaded-video inference; it **does not** run entirely in the browser. pywebview is a thin local desktop client. The proposed all-browser ONNX/TFLite path is not implemented or benchmarked. The repo must never claim video remains on-device when hosted on a remote server.

## Known engineering limitations

- Static fusion training now streams samples but MediaPipe extraction is repeated on epochs; precomputed landmark caching would be faster for very large datasets.
- Static production service currently accepts image-only EfficientNet models. A fusion-specific serving contract requires metadata and input tests before enablement.
- Python and browser smoothing are not identical; the Python emitter is a library utility, while the actual browser composer uses its own implementation.
- The hand-only pipeline remains a baseline; whole-body code is a separate research track pending empirical evaluation.
- ASL Citizen, WLASL, ASL-LEX, Kaggle alphabet and any third-party model must be used only under their actual license and access terms.

# SignBridge ASL — research-grade sign recognition

Reconstruction and extension of the 2024 thesis **Deep Gesture Interpretation for American Sign Language** (Sandeep Das, Shruti Sharma, Diya Tembhurne). This repository implements *static fingerspelling* and a separate *isolated-sign video* research track. They are distinct tasks and their results must not be conflated.

## Research tracks

- **Static:** train and compare EfficientNet-B0 and MobileNetV3 with RGB + hand-landmark fusion. Image crops preserve aspect ratio. Evaluation reports macro F1, per-class recall, confusion matrix and confidence calibration.
- **Temporal:** signer-independent isolated-sign classification using normalized hand landmarks, temporal convolution and bidirectional GRU, with explicit missing-hand masks.
- **Deployment:** local web server and desktop wrapper load genuine, locally trained checkpoints. No fake recognition is returned if a checkpoint is unavailable.

## Installation

Use Python 3.11 and create an isolated environment. Install `requirements.txt`; serving dependencies are in `requirements-serving.txt`. Run `python -m pytest -q` for offline tests.

## Data and reproducibility

Read `docs/DATASETS.md` and `docs/RESEARCH_PROTOCOL.md` before training. The static alphabet benchmark and the isolated-sign video benchmark must use separate models, labels and test reports. Do not put raw datasets, personally identifiable video recordings, or model weights with uncertain redistribution rights into Git. Dataset licenses and user privacy must be respected.

## Current status

**Code implementation is not an accuracy claim.** The repository does not contain trained checkpoints or verified accuracy results. Those require local dataset acquisition, GPU training and held-out evaluation. Historical thesis figures are not reproduced as new results.

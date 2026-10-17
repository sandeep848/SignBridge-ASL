# SignBridge-ASL

**Reproducible research on American Sign Language fingerspelling and isolated-sign recognition**

[![CI](https://github.com/sandeep848/SignBridge-ASL/actions/workflows/ci.yml/badge.svg)](https://github.com/sandeep848/SignBridge-ASL/actions/workflows/ci.yml) [![Model smoke tests](https://github.com/sandeep848/SignBridge-ASL/actions/workflows/model-smoke.yml/badge.svg)](https://github.com/sandeep848/SignBridge-ASL/actions/workflows/model-smoke.yml)

> **Research software; not an established sign-language translator.** The repository provides real implementations and tests, but contains **no released trained checkpoints or independently verified test accuracy**. Research claims and public deployments must pass the [release checklist](docs/RELEASE_CHECKLIST.md).

## Origin and purpose

Reconstruction and research extension of *Deep Gesture Interpretation for American Sign Language* (2024), by Sandeep Das, Shruti Sharma, and Diya Tembhurne. The project distinguishes static alphabet recognition, isolated sign classification, and sign-disjoint retrieval. It does not claim to translate continuous ASL discourse.

## Research tracks

| Track | Implemented research components | Required evidence |
|---|---|---|
| Static image | EfficientNetB0 baseline; MobileNetV3 RGB/landmark fusion; detector-crop preprocessing; held-out metrics | Real static benchmark and camera-domain evaluation |
| Isolated video | Two-hand CNN/BiGRU/attention baseline, normalized holistic body-relative features and independent holistic encoder | Signer-disjoint trials with at least three seeds |
| Reliability | Tracker-failure augmentation; group bootstrap CIs; ECE; split-conformal prediction sets | Robustness and independent calibration coverage |
| Few-shot retrieval | Cosine-prototype index for separately trained embeddings | Sign-disjoint few-shot evaluation, retrieval MRR |
| Interfaces | Local FastAPI webcam image and isolated-video endpoints; browser and pywebview desktop shell | End-to-end system tests with actual checkpoints |

Read [Architecture](docs/ARCHITECTURE.md), [Novelty](docs/NOVELTY.md), [Research protocol](docs/RESEARCH_PROTOCOL.md), [Ethics](docs/ETHICS.md), and [Model card](docs/MODEL_CARD.md).

## Installation

Python 3.11 is recommended. For native Windows NVIDIA GPU training, use WSL2 with a compatible CUDA stack rather than assuming native TensorFlow GPU support.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pytest -q
```

For browser/server install `requirements-serving.txt`; for the desktop app install `requirements-desktop.txt`. See [Compatibility](docs/COMPATIBILITY.md). The CI dependencies are intentionally smaller than the full ML stack.

## Acquire data and prevent leakage

No original image data, user recordings or third-party checkpoints are distributed here. See [Datasets](docs/DATASETS.md) for provider links and license considerations. Obtain official signer IDs and split annotations instead of generating arbitrary frame-level splits.

Static data must be organized as `data/split/{train,val,test}/{label}/image.jpg`. To align the training images with webcam preprocessing, generate a **new** split-preserving cropped corpus:

```powershell
python -m asl.prep_static --input data/split --output data/crops
python -m asl.train --data data/crops --out outputs/static
python -m asl.evaluate_static --model outputs/static/best.keras --labels outputs/static/labels.json --data data/crops/test
```

To train the fusion model on the same cropped corpus:

```powershell
python -m asl.fusion --data data/crops --out outputs/fusion
```

For signer-disjoint isolated-video research, import a **dataset-authoritative** CSV with `path,label,split,signer`. The importer never invents a signer's identity or an official split.

```powershell
python -m asl.manifest --source data/videos/official_metadata.csv --output data/videos/manifest.csv
python -m asl.video_features --manifest data/videos/manifest.csv --output data/temporal
python -m asl.temporal --data data/temporal --out outputs/temporal
python -m asl.evaluate_temporal --model outputs/temporal/best.keras --labels outputs/temporal/labels.json --data data/temporal
```

For the separate holistic experiment:

```powershell
python -m asl.holistic_extract --manifest data/videos/manifest.csv --out data/holistic
python -m asl.holistic_train --data data/holistic --output outputs/holistic
```

**Important:** the two-hand and holistic checkpoints use different feature schemas. They are not interchangeable. Neither experiment yields a meaningful score until trained and evaluated on real data.

## Optional edge export

The repository includes genuine TensorFlow Lite conversion for trained Keras models. Float16 conversion:

```powershell
python -m asl.export_tflite --checkpoint outputs/temporal/best.keras --output outputs/edge/temporal.tflite --mode float16
```

INT8 export requires aligned representative **real training examples** in an NPZ, supplied with `--representative`, and must be followed by device-level accuracy and latency validation. A TFLite file alone does not constitute verified private in-browser inference.

## Local inference

Once a static or temporal checkpoint exists, configure the local API in PowerShell:

```powershell
$env:SIGNBRIDGE_STATIC_MODEL = "outputs/static/best.keras"
$env:SIGNBRIDGE_STATIC_LABELS = "outputs/static/labels.json"
$env:SIGNBRIDGE_TEMPORAL_MODEL = "outputs/temporal/best.keras"
$env:SIGNBRIDGE_TEMPORAL_LABELS = "outputs/temporal/labels.json"
python -m uvicorn apps.server:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Start the desktop shell with `python -m apps.desktop` after installing the desktop requirements. The runtime rejects unavailable checkpoints rather than returning fake signs.

The fusion and holistic models are **not** yet supported by the deployed API and should not be configured as though they were the image-only or two-hand models. The local server is **not** security-hardened for Internet exposure.

## Reproducibility and contribution

- **Testing:** `python -m pytest -q`. GitHub Actions includes lightweight CI and a separate TensorFlow CPU model-construction job.
- **Data:** preserve source, license, signer splits, feature schema and manifest checksums.
- **Evaluation:** choose hyperparameters on validation, reserve a separate calibration set, run untouched test data only after model selection; report multi-seed variation and failure cases.
- **Community:** [Contributing](CONTRIBUTING.md), [Security](SECURITY.md), and [Ethics](docs/ETHICS.md).
- **Licensing:** no repository-wide license is granted pending confirmation of source-work rights and co-author/rights-holder permissions. Public visibility alone does not authorize reuse.

### Release status

No evidence-backed research release is ready yet. Code, documentation and automation are under active verification. Please do not cite historical thesis results as benchmark outcomes for this rebuilt system.

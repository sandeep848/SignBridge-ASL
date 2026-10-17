# SignBridge-ASL

Research-oriented rebuild and extension of the 2024 thesis **Deep Gesture Interpretation for American Sign Language** by Sandeep Das, Shruti Sharma and Diya Tembhurne.

**Current scope:** static fingerspelling recognition and separate isolated-sign video recognition. These are distinct classification problems. The project does **not** claim full continuous ASL translation.

## Research models

- `asl.train`: EfficientNetB0 static image baseline with pretrained backbone, augmentation, fine-tuning and best-validation checkpoint.
- `asl.fusion`: MobileNetV3-Large or EfficientNetB0 RGB + normalized hand-landmark fusion with missing-landmark dropout.
- `asl.temporal`: temporal convolution + bidirectional GRU + self-attention for isolated-sign video sequences.
- `asl.video_features`: MediaPipe-based two-hand extraction with missing-hand indicators, 32-frame resampling and signer-overlap checks.
- `asl.evaluate_temporal`: held-out metrics including macro-F1, balanced accuracy, per-class report and confusion matrix.
- `apps.server`: checkpoint-backed local REST inference for webcam images and uploaded videos.
- `apps.desktop`: local desktop shell for the same server.
- `apps.index.html`: webcam and isolated-video UI. No synthetic predictions or dummy checkpoints.

## Local installation

Install Python 3.11, then from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-serving.txt
python -m pytest -q
```

Linux/macOS: activate using `source .venv/bin/activate`.

### Data

Read [dataset guidance](docs/DATASETS.md) and the [evaluation protocol](docs/RESEARCH_PROTOCOL.md). Datasets are **not** included. Obtain them directly from the providers, respecting their licenses and consent requirements. Do not upload private signer videos to the repository.

The static benchmark requires separated directory trees:

```text
data/split/
  train/A/*.jpg ... train/Z/*.jpg
  val/A/*.jpg ... val/Z/*.jpg
  test/A/*.jpg ... test/Z/*.jpg
```

Use only the classes your dataset actually supports; folders must agree across splits. J and Z require motion for genuine full-sign evaluation.

Train the image baseline:

```powershell
python -m asl.train --data data/split --out outputs/static --epochs 15 --fine-tune-epochs 8
```

Evaluate the held-out static test split (after finalizing the model):

```powershell
python -m asl.evaluate_static --model outputs/static/best.keras --labels outputs/static/labels.json --data data/split/test --out outputs/static/test_metrics.json
```

Train RGB+landmark fusion:

```powershell
python -m asl.fusion --data data/split --out outputs/fusion --backbone mobilenet --epochs 25
```

For isolated-sign video training, create a CSV manifest matching the official dataset split and signer identifiers, with header `path,label,split,signer`. Paths are relative to the manifest directory. Then:

```powershell
python -m asl.video_features --manifest data/videos/manifest.csv --output data/temporal
python -m asl.temporal --data data/temporal --out outputs/temporal
python -m asl.evaluate_temporal --model outputs/temporal/best.keras --labels outputs/temporal/labels.json --data data/temporal --out outputs/temporal/test_metrics.json
```

### Browser or desktop interface

The current static serving endpoint accepts the single-input EfficientNet baseline checkpoint (not the three-input fusion model).

Set local variables in PowerShell:

```powershell
$env:SIGNBRIDGE_STATIC_MODEL = "outputs/static/best.keras"
$env:SIGNBRIDGE_STATIC_LABELS = "outputs/static/labels.json"
$env:SIGNBRIDGE_TEMPORAL_MODEL = "outputs/temporal/best.keras"
$env:SIGNBRIDGE_TEMPORAL_LABELS = "outputs/temporal/labels.json"
python -m uvicorn apps.server:app --host 127.0.0.1 --port 8000
```

Then visit `http://127.0.0.1:8000`. Use `python -m apps.desktop` for the desktop client (requires native WebView support). Models that are absent return an explicit HTTP 503 error.

## Community and maintenance

See [Contributing](CONTRIBUTING.md), [Security](SECURITY.md), [Compatibility](docs/COMPATIBILITY.md), [Research protocol](docs/RESEARCH_PROTOCOL.md), and [Model card](docs/MODEL_CARD.md).

**Status: research software, not a certified public translation service.** The local API has not been security-reviewed for Internet exposure. The fusion pipeline presently materializes the dataset in memory; use a limited subset for local feasibility testing until streaming is implemented. Static browser inference uses an EfficientNet checkpoint trained from image folders; cropped live hands may introduce domain shift. Evaluate this before real-world deployment.

**Licensing:** no project-wide open-source license has been selected. Until the authors and relevant rights holders confirm permission, the repository being public does not mean unrestricted reuse. Contributors should not assume they can redistribute the original thesis, private recordings, third-party datasets or model weights.

## Scientific integrity and status

- The source and CI checks are available; no pretrained ASL checkpoints, locally trained weights or fabricated benchmark results have been committed.
- Held-out, signer-independent accuracy is **not yet measured**. GitHub tests are unit tests, **not** GPU training or real-world validation.
- Static images and isolated-sign videos have separate datasets, outputs, labels and metrics.
- Model novelty is an experimental hypothesis until verified by controlled ablations. 
- Runtime constraints: TensorFlow and MediaPipe support vary by OS, Python version and GPU backend. Verify your local CUDA/CPU configuration before scaling training.

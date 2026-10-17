# Compatibility and deployment boundaries

The supported research environment is Python 3.11 with TensorFlow 2.16–2.18, MediaPipe 0.10.20 and NumPy 1.x.

* **Linux with NVIDIA GPU:** recommended for accelerated TensorFlow training after CUDA/cuDNN compatibility checks.
* **Windows native:** TensorFlow 2.11+ does not provide native-Windows CUDA support. Use WSL2 with compatible GPU drivers for NVIDIA acceleration or train on CPU.
* **macOS:** TensorFlow acceleration depends on platform-specific packages and configuration.
* **CI:** only lightweight, hardware-free unit tests are run; CI does not certify the TensorFlow GPU build, browser camera access, model quality or complete dataset ingestion.
* **Desktop:** pywebview requires platform WebView runtime; use local HTTP in a browser when it is unavailable.

Production deployment needs real model artifacts, a model signature/metadata contract, stable image preprocessing, model load tests, video resource limits, real-device privacy review, and operational monitoring. This repository is not presently certified for public multi-user hosting.

For reproducibility capture: OS, Python/package lockfile, GPU/driver/CUDA versions, data manifest hash, random seed, checkpoint hash, inference preprocessing version and split methodology.

"""Split-conformal prediction sets, fitted ONLY on held-out calibration examples.

Guarantees require exchangeability; signer and dataset shift can invalidate coverage.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

class ConformalClassifier:
    def __init__(self, alpha: float = 0.1):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between zero and one")
        self.alpha = float(alpha)
        self.threshold = None
        self.class_count = None

    def fit(self, probabilities, labels):
        p = np.asarray(probabilities, dtype=np.float64)
        y = np.asarray(labels, dtype=int)
        if p.ndim != 2 or len(p) != len(y) or not len(y) or not np.isfinite(p).all():
            raise ValueError("Invalid probability matrix")
        if np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(1), 1, atol=1e-4):
            raise ValueError("Expected normalized class probabilities")
        if np.any(y < 0) or np.any(y >= p.shape[1]):
            raise ValueError("Class index out of bounds")
        scores = 1.0 - p[np.arange(len(y)), y]
        k = int(np.ceil((len(y) + 1) * (1 - self.alpha)))
        self.threshold = 1.0 if k > len(y) else float(np.partition(scores, k - 1)[k - 1])
        self.class_count = int(p.shape[1])
        return self

    def predict_sets(self, probabilities):
        if self.threshold is None:
            raise RuntimeError("Calibrate before prediction")
        p = np.asarray(probabilities, dtype=float)
        if p.ndim != 2 or p.shape[1] != self.class_count or not np.isfinite(p).all():
            raise ValueError("Probability dimension mismatch")
        return [[int(i) for i in np.flatnonzero(1.0 - row <= self.threshold)] for row in p]

    def save(self, path):
        if self.threshold is None:
            raise RuntimeError("Not fitted")
        Path(path).write_text(json.dumps({"alpha":self.alpha,"threshold":self.threshold,"class_count":self.class_count},indent=2))

    @classmethod
    def load(cls, path):
        d=json.loads(Path(path).read_text())
        obj=cls(d["alpha"])
        obj.threshold=float(d["threshold"])
        obj.class_count=int(d["class_count"])
        return obj

def coverage(sets, labels):
    if len(sets) != len(labels) or not labels:
        raise ValueError("Length mismatch or empty dataset")
    return {"coverage":float(np.mean([int(int(y) in s) for s,y in zip(sets,labels)])),
            "mean_set_size":float(np.mean([len(s) for s in sets]))}

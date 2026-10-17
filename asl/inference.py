"""Checkpoint-backed inference and conservative prediction emission."""
import json
from collections import deque
from pathlib import Path
import numpy as np
from .constants import IMAGE_SIZE

class GestureClassifier:
    def __init__(self,model_path,labels_path,threshold=0.8):
        import tensorflow as tf
        self.model=tf.keras.models.load_model(model_path)
        self.labels=json.loads(Path(labels_path).read_text())["classes"]
        self.threshold=threshold
        if int(self.model.output_shape[-1])!=len(self.labels):
            raise ValueError("Checkpoint and label mapping disagree")
    def predict(self,rgb):
        if rgb.shape!=(IMAGE_SIZE,IMAGE_SIZE,3):
            raise ValueError("Unexpected input dimensions")
        scores=self.model.predict(rgb.astype(np.float32)[None],verbose=0)[0]
        index=int(np.argmax(scores))
        conf=float(scores[index])
        return self.labels[index] if conf>=self.threshold else None,conf

class StableEmitter:
    def __init__(self,window=7,min_votes=5):
        if not 1<=min_votes<=window:
            raise ValueError("Invalid smoothing configuration")
        self.history=deque(maxlen=window)
        self.min_votes=min_votes
        self.latched=None
    def update(self,label):
        if label is None:
            self.history.clear()
            self.latched=None
            return None
        self.history.append(label)
        if self.history.count(label)>=self.min_votes and self.latched!=label:
            self.latched=label
            return label
        return None

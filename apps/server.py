"""Local-only checkpoint-backed SignBridge API.

Synchronous endpoints run in Starlette's threadpool. Never expose without
authentication, quotas and an independent deployment security review.
"""
import io
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

app=FastAPI(title="SignBridge ASL",version="0.3.0")
ROOT=Path(__file__).parent

def configured(key):
    path=os.environ.get(key)
    if not path or not Path(path).is_file():
        raise HTTPException(503,f"Trained checkpoint/configuration unavailable: {key}")
    return path

@lru_cache(maxsize=2)
def classifier(model_path,labels_path):
    from asl.inference import GestureClassifier
    return GestureClassifier(model_path,labels_path,float(os.getenv("SIGNBRIDGE_CONFIDENCE","0.8")))

@lru_cache(maxsize=1)
def temporal_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path)

@app.get("/")
def home():
    return FileResponse(ROOT/"index.html")

@app.get("/health")
def health():
    keys=("SIGNBRIDGE_STATIC_MODEL","SIGNBRIDGE_STATIC_LABELS","SIGNBRIDGE_TEMPORAL_MODEL","SIGNBRIDGE_TEMPORAL_LABELS")
    return {"service":"SignBridge","models":{key:bool(os.getenv(key) and Path(os.environ[key]).is_file()) for key in keys}}

@app.post("/api/static/predict")
def predict_static(image:UploadFile=File(...)):
    import cv2
    from asl.preprocess import crop_box,letterbox
    from asl.hands import HandDetector
    model_path=configured("SIGNBRIDGE_STATIC_MODEL")
    labels_path=configured("SIGNBRIDGE_STATIC_LABELS")
    raw=image.file.read(5_000_001)
    if len(raw)>5_000_000:
        raise HTTPException(413,"Image exceeds 5 MB")
    try:
        with Image.open(io.BytesIO(raw)) as original:
            original.verify()
        with Image.open(io.BytesIO(raw)) as original:
            rgb=np.asarray(original.convert("RGB"))
    except (UnidentifiedImageError,ValueError,OSError) as error:
        raise HTTPException(422,"Invalid input image") from error
    if rgb.shape[0]*rgb.shape[1]>16_000_000:
        raise HTTPException(413,"Image dimensions exceed limit")
    detector=HandDetector()
    try:
        boxes=detector.detect(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    finally:
        detector.close()
    if not boxes:
        return {"label":None,"confidence":0.0,"reason":"hand_not_detected"}
    try:
        cropped=letterbox(crop_box(rgb,boxes[0]))
    except ValueError:
        return {"label":None,"confidence":0.0,"reason":"invalid_hand_crop"}
    try:
        label,confidence=classifier(model_path,labels_path).predict(cropped)
    except (ValueError,KeyError) as error:
        raise HTTPException(503,"Incompatible model, labels or preprocessing") from error
    return {"label":label,"confidence":confidence,"reason":None if label else "low_confidence"}

@app.post("/api/temporal/predict")
def predict_temporal(video:UploadFile=File(...)):
    model_path=configured("SIGNBRIDGE_TEMPORAL_MODEL")
    labels_path=configured("SIGNBRIDGE_TEMPORAL_LABELS")
    suffix=Path(video.filename or "").suffix.lower()
    if suffix not in {".mp4",".mov",".avi",".webm"}:
        raise HTTPException(415,"Unsupported video format")
    raw=video.file.read(50_000_001)
    if len(raw)>50_000_000:
        raise HTTPException(413,"Video exceeds 50 MB")
    import mediapipe as mp
    from asl.video_features import extract_video
    if not hasattr(mp,"solutions"):
        raise HTTPException(503,"MediaPipe hand extraction unavailable")
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/("upload"+suffix)
        path.write_bytes(raw)
        with mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=2) as hands:
            try:
                sequence=extract_video(path,hands)
            except ValueError as error:
                raise HTTPException(422,str(error)) from error
    try:
        labels=json.loads(Path(labels_path).read_text())["classes"]
        probs=temporal_model(model_path).predict(sequence[None],verbose=0)[0]
    except (ValueError,KeyError,OSError) as error:
        raise HTTPException(503,"Incompatible temporal checkpoint or metadata") from error
    if len(probs)!=len(labels):
        raise HTTPException(503,"Checkpoint and label mapping mismatch")
    idx=int(np.argmax(probs))
    confidence=float(probs[idx])
    threshold=float(os.getenv("SIGNBRIDGE_CONFIDENCE","0.8"))
    accepted=confidence>=threshold
    return {"label":labels[idx] if accepted else None,"confidence":confidence,"reason":None if accepted else "low_confidence"}

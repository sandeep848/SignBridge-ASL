"""Extract holistic full-body feature sequences from approved manifest videos.

Output is an independent schema; legacy hand-only checkpoints are incompatible.
"""
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
from .holistic import DIM,extract_frame
from .video_features import validate_manifest

def extract(path,hands,max_frames=96):
    capture=cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unreadable video: {path}")
    count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if count<1:
        capture.release()
        raise ValueError(f"Video frame count unavailable: {path}")
    selected=set(np.rint(np.linspace(0,count-1,min(count,max_frames))).astype(int))
    frames=[]
    index=0
    try:
        while True:
            ok,image=capture.read()
            if not ok:
                break
            if index in selected:
                response=hands.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                frames.append(extract_frame(response))
            index+=1
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames extracted: {path}")
    return np.stack(frames)

def main():
    import mediapipe as mp
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--out",default="data/holistic")
    p.add_argument("--max-frames",type=int,default=96)
    a=p.parse_args()
    if a.max_frames < 1:
        raise ValueError("max-frames must be positive")
    with open(a.manifest,encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    validate_manifest(rows)
    root=Path(a.manifest).parent
    out=Path(a.out)
    for i,row in enumerate(rows):
        dest=out/row["split"]/f"{i:07d}.npz"
        if dest.exists():
            raise FileExistsError(dest)
        path=(root/row["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        # A fresh tracker per clip prevents cross-video state contamination.
        with mp.solutions.holistic.Holistic(static_image_mode=False,refine_face_landmarks=False) as holistic:
            points=extract(path,holistic,max_frames=a.max_frames)
        dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,points=points,label=np.asarray(row["label"]))

if __name__=="__main__":
    main()

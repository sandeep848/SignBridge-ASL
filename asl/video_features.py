"""Extract real hand landmarks from source videos; reject split leakage."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .temporal import FRAMES, resample_sequence

def extract_video(path: Path, hands, max_frames: int = 96):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {path}")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, length // max_frames) if length else 1
    values = []
    index = 0
    try:
        while len(values) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride:
                index += 1
                continue
            index += 1
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            xyz = np.zeros((2, 21, 3), dtype=np.float32)
            visible = np.zeros(2, dtype=np.float32)
            detected = sorted(result.multi_hand_landmarks or [], key=lambda hand: hand.landmark[0].x)
            for hand_index, hand in enumerate(detected[:2]):
                coords = np.asarray([[pt.x, pt.y, pt.z] for pt in hand.landmark], dtype=np.float32)
                coords -= coords[0].copy()
                coords /= max(float(np.max(np.linalg.norm(coords[:, :2], axis=1))), 1e-5)
                xyz[hand_index] = coords
                visible[hand_index] = 1
            values.append(np.concatenate([xyz.ravel(), visible]))
    finally:
        cap.release()
    if not values:
        raise ValueError(f"Empty video {path}")
    return resample_sequence(np.stack(values), FRAMES)

def build(manifest: Path, output: Path, max_frames: int):
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise RuntimeError("A MediaPipe version with solutions.hands is required")
    with manifest.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "label", "split", "signer"} <= set(reader.fieldnames or []):
            raise ValueError("Manifest must have path,label,split,signer")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty")
    assigned = {}
    for row in rows:
        split, signer = row["split"], row["signer"]
        if split not in {"train", "val", "test"} or not signer:
            raise ValueError("Invalid split or missing signer")
        if signer in assigned and assigned[signer] != split:
            raise ValueError(f"Signer leakage: {signer}")
        assigned[signer] = split
    failures = []
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2) as hands:
        for i, row in enumerate(rows):
            source = (manifest.parent / row["path"]).resolve()
            destination = output / row["split"] / f"{i:07d}.npz"
            if destination.exists():
                continue
            try:
                if not source.is_file():
                    raise FileNotFoundError(source)
                points = extract_video(source, hands, max_frames)
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination, points=points, label=np.asarray(row["label"]))
            except (ValueError, FileNotFoundError) as error:
                failures.append({"path": str(source), "error": str(error)})
    if failures:
        output.mkdir(parents=True, exist_ok=True)
        (output / "extraction_failures.json").write_text(json.dumps(failures, indent=2))
        raise RuntimeError(f"{len(failures)} extraction failures; see extraction_failures.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="data/temporal")
    parser.add_argument("--max-frames", type=int, default=96)
    options = parser.parse_args()
    build(Path(options.manifest), Path(options.output), options.max_frames)

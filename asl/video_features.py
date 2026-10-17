"""Extract normalized two-hand landmarks with stable left/right slots."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .temporal import FRAMES, resample_sequence

def frame_features(result):
    """Assign detections to hand slots using handedness, not frame-varying x-position."""
    xyz = np.zeros((2, 21, 3), dtype=np.float32)
    visible = np.zeros(2, dtype=np.float32)
    hands = result.multi_hand_landmarks or []
    handedness = result.multi_handedness or []
    taken = set()
    for i, hand in enumerate(hands[:2]):
        slot = None
        if i < len(handedness) and handedness[i].classification:
            name = handedness[i].classification[0].label.lower()
            # MediaPipe handedness assumes a horizontally mirrored selfie input.
            # Slot names are used only as stable model features, not anatomical claims.
            slot = {"left": 0, "right": 1}.get(name)
        if slot is None or slot in taken:
            slot = next((j for j in range(2) if j not in taken), None)
        if slot is None:
            break
        coords = np.asarray([[pt.x, pt.y, pt.z] for pt in hand.landmark], dtype=np.float32)
        coords -= coords[0].copy()
        coords /= max(float(np.max(np.linalg.norm(coords[:, :2], axis=1))), 1e-5)
        xyz[slot] = coords
        visible[slot] = 1.0
        taken.add(slot)
    return np.concatenate([xyz.ravel(), visible])

def extract_video(path: Path, hands, max_frames: int = 96):
    import cv2
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, count // max_frames) if count > 0 else 1
    values = []
    index = 0
    try:
        while len(values) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                values.append(frame_features(result))
            index += 1
    finally:
        cap.release()
    if not values:
        raise ValueError(f"Empty video: {path}")
    return resample_sequence(np.stack(values), FRAMES)

def validate_manifest(rows):
    if not rows:
        raise ValueError("Manifest is empty")
    assigned = {}
    files = set()
    for row in rows:
        split, signer = row["split"].strip(), row["signer"].strip()
        path, label = row["path"].strip(), row["label"].strip()
        if split not in {"train", "val", "test"} or not signer or not path or not label:
            raise ValueError("Invalid row: split, signer, path and label are required")
        if signer in assigned and assigned[signer] != split:
            raise ValueError(f"Signer leakage: {signer}")
        if path in files:
            raise ValueError(f"Duplicate source recording: {path}")
        assigned[signer] = split
        files.add(path)

def build(manifest: Path, output: Path, max_frames: int):
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise RuntimeError("MediaPipe Hands API unavailable; consult the pinned requirements")
    with manifest.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "label", "split", "signer"} <= set(reader.fieldnames or []):
            raise ValueError("Manifest needs path,label,split,signer")
        rows = list(reader)
    validate_manifest(rows)
    failures = []
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2) as hands:
        for index, row in enumerate(rows):
            source = (manifest.parent / row["path"]).resolve()
            destination = output / row["split"] / f"{index:07d}.npz"
            if destination.exists():
                raise FileExistsError(f"Refusing to reuse possibly stale features: {destination}")
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
    args = parser.parse_args()
    build(Path(args.manifest), Path(args.output), args.max_frames)

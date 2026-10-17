"""Generate consistent detector-cropped static training and test images.

Output images are 224x224 RGB letterboxes, matching the webcam API. Use before
splitting only if split metadata is preserved; this script expects fixed splits.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from .preprocess import crop_box, letterbox

def process(root, out):
    from .hands import HandDetector
    counts={"written":0,"missing_hand":0,"unreadable":0}
    detector=HandDetector(max_hands=1)
    try:
        for split in ("train","val","test"):
            source=root/split
            if not source.is_dir():
                raise FileNotFoundError(source)
            for folder in sorted(p for p in source.iterdir() if p.is_dir()):
                for file in sorted(folder.iterdir()):
                    if file.suffix.lower() not in {".jpg",".jpeg",".png",".webp"}:
                        continue
                    image=cv2.imread(str(file))
                    if image is None:
                        counts["unreadable"]+=1
                        continue
                    boxes=detector.detect(image)
                    if not boxes:
                        counts["missing_hand"]+=1
                        continue
                    try:
                        rgb=letterbox(crop_box(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),boxes[0]))
                    except ValueError:
                        counts["missing_hand"]+=1
                        continue
                    dest=out/split/folder.name/(file.stem+".png")
                    if dest.exists():
                        raise FileExistsError(dest)
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    if not cv2.imwrite(str(dest),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):
                        raise OSError(f"Could not write {dest}")
                    counts["written"]+=1
    finally:
        detector.close()
    out.mkdir(parents=True,exist_ok=True)
    (out/"preprocessing_report.json").write_text(json.dumps(counts,indent=2))
    return counts

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    print(process(Path(a.input),Path(a.output)))

"""Safe, aspect-ratio-preserving image transforms."""
import numpy as np
import cv2
from .constants import IMAGE_SIZE

def letterbox(image, size=IMAGE_SIZE):
    if image is None or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("Expected nonempty HxWx3 image")
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = max(1, min(size, round(h * scale))), max(1, min(size, round(w * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    result = np.full((size, size, 3), 255, dtype=np.uint8)
    y, x = (size-nh)//2, (size-nw)//2
    result[y:y+nh, x:x+nw] = resized
    return result

def crop_box(image, box, padding=0.20):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid bounding box")
    dx, dy = round((x2-x1)*padding), round((y2-y1)*padding)
    x1, y1, x2, y2 = max(0,x1-dx), max(0,y1-dy), min(w,x2+dx), min(h,y2+dy)
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Bounding box outside image")
    return image[y1:y2,x1:x2].copy()

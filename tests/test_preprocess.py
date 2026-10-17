import numpy as np
import pytest
from asl.preprocess import crop_box,letterbox

def test_letterbox_preserves_shape():
    x=np.zeros((40,100,3),dtype=np.uint8)
    y=letterbox(x,224)
    assert y.shape==(224,224,3)
    assert y.dtype==np.uint8

def test_invalid_crop_raises():
    with pytest.raises(ValueError):
        crop_box(np.zeros((10,10,3),dtype=np.uint8),(5,5,2,2))

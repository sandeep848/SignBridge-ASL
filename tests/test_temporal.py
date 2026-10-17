import numpy as np
import pytest
from asl.temporal import FEATURES, resample_sequence

def test_preserves_dimension_and_frames():
    x = np.zeros((10, FEATURES), dtype=np.float32)
    assert resample_sequence(x, 32).shape == (32, FEATURES)

def test_missing_hand_not_interpolated():
    x = np.zeros((2, FEATURES), dtype=np.float32)
    x[1, -1] = 1.0
    assert set(np.unique(resample_sequence(x, 7)[:, -1])) == {0.0, 1.0}

def test_rejects_bad_input():
    with pytest.raises(ValueError):
        resample_sequence(np.zeros((0, FEATURES), dtype=np.float32))
    with pytest.raises(ValueError):
        resample_sequence(np.zeros((4, 3), dtype=np.float32))

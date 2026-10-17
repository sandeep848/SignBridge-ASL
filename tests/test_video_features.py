from types import SimpleNamespace
import numpy as np
import pytest
from asl.video_features import frame_features, validate_manifest

def row(path, label="A", split="train", signer="s1"):
    return dict(path=path, label=label, split=split, signer=signer)

def test_rejects_signer_overlap():
    with pytest.raises(ValueError, match="Signer leakage"):
        validate_manifest([row("a.mp4"), row("b.mp4", split="test")])

def test_rejects_duplicate_source():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_manifest([row("a.mp4"), row("a.mp4")])

def test_empty_detection_features():
    output = frame_features(SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None))
    assert output.shape == (128,)
    assert np.count_nonzero(output) == 0

def test_handedness_assigns_consistent_slot():
    point = SimpleNamespace(x=0.4, y=0.4, z=0.0)
    hand = SimpleNamespace(landmark=[point] * 21)
    classification = SimpleNamespace(label="Right")
    handed = SimpleNamespace(classification=[classification])
    features = frame_features(SimpleNamespace(multi_hand_landmarks=[hand], multi_handedness=[handed]))
    assert np.allclose(features[-2:], [0, 1])

from types import SimpleNamespace
import numpy as np
from asl.holistic import DIM,extract_frame,add_kinematics
from asl.augment import tracker_dropout,temporal_speed

def test_all_missing_frame_is_finite():
    result=SimpleNamespace(left_hand_landmarks=None,right_hand_landmarks=None,
        pose_landmarks=None,face_landmarks=None)
    features=extract_frame(result)
    assert features.shape==(DIM,)
    assert np.isfinite(features).all()

def test_kinematics():
    seq=np.zeros((4,DIM),np.float32)
    assert add_kinematics(seq).shape==(4,DIM*2-4)

def test_tracker_dropout():
    arr=np.ones((20,128),np.float32)
    masked=tracker_dropout(arr,probability=1,seed=42)
    assert np.any(masked[:,:126]==0)

def test_speed():
    assert temporal_speed(np.zeros((30,128),np.float32),factor=1.5).shape==(20,128)

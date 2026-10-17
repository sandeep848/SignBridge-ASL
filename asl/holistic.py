"""Body-relative holistic features for the advanced research track.

Schema is distinct from the legacy hands-only model; never mix checkpoints.
MediaPipe Holistic is processed locally. Inference never writes camera frames.
"""
import numpy as np

HAND_JOINTS=21
POSE_JOINTS=(0,11,12,13,14,15,16,23,24)
FACE_JOINTS=(4,33,61,133,152,168,263,291)
SCHEMA="holistic-body-v1"
# Hand local xyz (126), wrist body xyz (6), pose xyz (27), face xyz (24), flags (4)
DIM=126+6+27+24+4

def _landmarks(container):
    if container is None:
        return None
    return np.asarray([[p.x,p.y,p.z] for p in container.landmark],dtype=np.float32)

def extract_frame(result):
    """Returns DIM finite values; body-relative locations are kept."""
    pose=_landmarks(getattr(result,"pose_landmarks",None))
    face=_landmarks(getattr(result,"face_landmarks",None))
    left=_landmarks(getattr(result,"left_hand_landmarks",None))
    right=_landmarks(getattr(result,"right_hand_landmarks",None))
    center=np.array([.5,.5,0],dtype=np.float32)
    scale=1.0
    if pose is not None and len(pose)>12:
        center=(pose[11]+pose[12])/2
        scale=max(float(np.linalg.norm((pose[11]-pose[12])[:2])),0.03)
    parts=[]
    wrist=[]
    for hand in (left,right):
        if hand is None or len(hand)!=HAND_JOINTS:
            parts.append(np.zeros(63,dtype=np.float32))
            wrist.append(np.zeros(3,dtype=np.float32))
        else:
            local=hand-hand[0]
            local/=max(float(np.linalg.norm(local[:,:2],axis=1).max()),1e-5)
            parts.append(local.flatten())
            wrist.append((hand[0]-center)/scale)
    if pose is not None and len(pose)>max(POSE_JOINTS):
        p=(pose[list(POSE_JOINTS)]-center)/scale
    else:
        p=np.zeros((len(POSE_JOINTS),3),dtype=np.float32)
    if face is not None and len(face)>max(FACE_JOINTS):
        f=(face[list(FACE_JOINTS)]-center)/scale
    else:
        f=np.zeros((len(FACE_JOINTS),3),dtype=np.float32)
    flags=np.asarray([left is not None,right is not None,pose is not None,face is not None],dtype=np.float32)
    output=np.concatenate([parts[0],parts[1],wrist[0],wrist[1],p.ravel(),f.ravel(),flags])
    if len(output)!=DIM or not np.isfinite(output).all():
        raise ValueError("Invalid Holistic features")
    return output.astype(np.float32)

def add_kinematics(frames):
    """Add temporal differences, preserving raw absolute body-relative features."""
    x=np.asarray(frames,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=DIM:
        raise ValueError("Expected (T,DIM)")
    speed=np.diff(x[:,:-4],axis=0,prepend=x[:1,:-4])
    return np.concatenate([x,speed],axis=-1)

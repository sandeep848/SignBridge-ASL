"""Physically bounded landmark augmentations, training partition only."""
import numpy as np

def tracker_dropout(sequence, *, probability=.1, max_run=4, seed=None):
    x=np.asarray(sequence,dtype=np.float32).copy()
    if x.ndim!=2 or x.shape[1]<128 or not 0<=probability<=1 or max_run<1:
        raise ValueError("Expected two-hand feature sequence")
    rng=np.random.default_rng(seed)
    t=0
    while t<len(x):
        if rng.random()<probability:
            length=int(rng.integers(1,max_run+1))
            hand=int(rng.integers(0,2))
            x[t:t+length,hand*63:(hand+1)*63]=0
            x[t:t+length,126+hand]=0
            t+=length
        else:
            t+=1
    return x

def temporal_speed(sequence, *, factor=1.0):
    arr=np.asarray(sequence,dtype=np.float32)
    if arr.ndim!=2 or len(arr)<2 or factor<=0:
        raise ValueError("Bad input sequence")
    size=max(2,int(round(len(arr)/factor)))
    chosen=np.rint(np.linspace(0,len(arr)-1,size)).astype(int)
    return arr[chosen]

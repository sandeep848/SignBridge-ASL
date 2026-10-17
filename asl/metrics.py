"""Metrics and reproducible bootstrap CIs for grouped evaluations."""
import numpy as np

def grouped_bootstrap_accuracy(truth, prediction, groups, *, iterations=1000, seed=42):
    y=np.asarray(truth)
    p=np.asarray(prediction)
    g=np.asarray(groups)
    if not(len(y)==len(p)==len(g)) or not len(y) or iterations < 1:
        raise ValueError("Invalid inputs")
    ids=np.unique(g)
    rng=np.random.default_rng(seed)
    scores=[]
    for _ in range(iterations):
        sampled=rng.choice(ids,size=len(ids),replace=True)
        index=np.concatenate([np.flatnonzero(g==name) for name in sampled])
        scores.append(float(np.mean(y[index]==p[index])))
    return {"accuracy":float(np.mean(y==p)),
            "ci_low":float(np.quantile(scores,0.025)),
            "ci_high":float(np.quantile(scores,0.975))}

def expected_calibration_error(probabilities, labels, bins=15):
    p=np.asarray(probabilities,dtype=float)
    y=np.asarray(labels)
    if p.ndim!=2 or len(p)!=len(y) or not len(y) or bins<2:
        raise ValueError("Invalid inputs")
    confidence=p.max(axis=1)
    correct=(p.argmax(axis=1)==y)
    edges=np.linspace(0,1,bins+1)
    result=0.0
    for i in range(bins):
        mask=(confidence>=edges[i]) & (confidence<edges[i+1] if i<bins-1 else confidence<=edges[i+1])
        if mask.any():
            result+=mask.mean()*abs(float(correct[mask].mean())-float(confidence[mask].mean()))
    return float(result)

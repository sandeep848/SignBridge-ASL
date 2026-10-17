"""Fit separate split-conformal calibration from untouched calibration predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
from .conformal import ConformalClassifier,coverage

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--probabilities",required=True,help="NPZ containing probs (N,C), labels (N,)")
    p.add_argument("--out",required=True)
    p.add_argument("--alpha",type=float,default=0.1)
    args=p.parse_args()
    with np.load(args.probabilities,allow_pickle=False) as npz:
        probs=npz["probs"]
        labels=npz["labels"]
    calibrator=ConformalClassifier(args.alpha).fit(probs,labels)
    dest=Path(args.out)
    dest.parent.mkdir(parents=True,exist_ok=True)
    calibrator.save(dest)
    print(json.dumps({"calibration_samples":len(labels),"alpha":args.alpha,
        "empirical_in_sample":coverage(calibrator.predict_sets(probs),labels.tolist()),
        "note":"In-sample diagnostics, NOT claimed test coverage"},indent=2))

if __name__=="__main__":
    main()

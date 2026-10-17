import numpy as np
import pytest
from asl.conformal import ConformalClassifier, coverage
from asl.experiment import metadata, validate
from asl.metrics import grouped_bootstrap_accuracy, expected_calibration_error

def test_conformal_coverage():
    p=np.array([[.9,.1],[.2,.8],[.7,.3],[.1,.9]])
    c=ConformalClassifier(.2).fit(p,np.array([0,1,0,1]))
    sets=c.predict_sets(p)
    assert coverage(sets,[0,1,0,1])["coverage"]>=.75

def test_conformal_requires_calibration():
    with pytest.raises(RuntimeError):
        ConformalClassifier().predict_sets(np.ones((1,2)) / 2)

def test_metadata_mismatch():
    m=metadata("temporal",["A","B"])
    with pytest.raises(ValueError):
        validate(m,"static",["A","B"],"other")

def test_signer_grouped_interval():
    result=grouped_bootstrap_accuracy([0,1,0,1],[0,1,1,1],["a","a","b","b"],iterations=30)
    assert 0 <= result["ci_low"] <= result["ci_high"] <= 1

def test_ece_perfect_confidence():
    assert expected_calibration_error([[1,0],[0,1]],[0,1])==0

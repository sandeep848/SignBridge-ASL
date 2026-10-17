"""Construct real TensorFlow models on CPU without downloading pretrained weights."""
import pytest
tf=pytest.importorskip("tensorflow")

def test_image_architecture():
    from asl.train import build_model
    model,_=build_model(3,pretrained=False)
    assert model.input_shape==(None,224,224,3)
    assert model.output_shape==(None,3)

def test_temporal_architecture():
    from asl.temporal import model,FEATURES
    network=model(5)
    assert network.input_shape==(None,32,FEATURES)
    assert network.output_shape==(None,5)

def test_fusion_architecture():
    from asl.fusion import build_fusion_model
    model=build_fusion_model(3,pretrained=False,backbone="mobilenet")
    assert len(model.inputs)==3
    assert model.output_shape==(None,3)

def test_holistic_architecture():
    from asl.holistic_train import build_model,FEATURES
    network=build_model(4)
    assert network.input_shape==(None,32,FEATURES)
    assert network.output_shape==(None,4)

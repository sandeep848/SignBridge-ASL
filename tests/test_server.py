"""API without checkpoints must fail closed, not fabricate predictions."""
import pytest
pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient
from apps.server import app

client=TestClient(app)

def test_health_reports_configured_models():
    response=client.get("/health")
    assert response.status_code==200
    assert "models" in response.json()

def test_missing_static_checkpoint(monkeypatch):
    monkeypatch.delenv("SIGNBRIDGE_STATIC_MODEL",raising=False)
    result=client.post("/api/static/predict",files={"image":("frame.jpg",b"bad","image/jpeg")})
    assert result.status_code==503

def test_wrong_video_type(monkeypatch,tmp_path):
    model=tmp_path/"model.keras"
    model.touch()
    monkeypatch.setenv("SIGNBRIDGE_TEMPORAL_MODEL",str(model))
    monkeypatch.setenv("SIGNBRIDGE_TEMPORAL_LABELS",str(model))
    result=client.post("/api/temporal/predict",files={"video":("bad.txt",b"bad","text/plain")})
    assert result.status_code==415

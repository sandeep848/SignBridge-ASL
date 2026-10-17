"""Immutable experiment metadata to detect train/serve schema mismatch."""
import hashlib
import json
import platform
import sys
from pathlib import Path

SCHEMA_VERSION = "signbridge-two-hand-v2"
STATIC_VERSION = "letterbox-rgb-v1"

def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def metadata(task, labels, *, manifest=None, extra=None):
    if task not in {"static","temporal","fusion"}:
        raise ValueError("Unknown model task")
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("Labels must be unique")
    record={"task":task,"labels":list(labels),
            "feature_schema":SCHEMA_VERSION if task=="temporal" else STATIC_VERSION,
            "python":sys.version.split()[0],"os":platform.platform()}
    if manifest is not None:
        record["manifest_sha256"]=file_sha256(manifest)
    if extra:
        record["config"]=extra
    return record

def validate(record, task, labels, schema):
    if record.get("task") != task or record.get("labels") != list(labels) or record.get("feature_schema") != schema:
        raise ValueError("Model provenance does not match inference configuration")

def save(record, path):
    Path(path).write_text(json.dumps(record,indent=2,sort_keys=True))

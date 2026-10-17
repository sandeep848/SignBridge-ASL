"""Validate or import explicit signer-disjoint video manifests.

No guessed signers or fabricated partitions. Official dataset metadata is mandatory.
"""
import argparse
import csv
from pathlib import Path
from .video_features import validate_manifest

FIELDS=("path","label","split","signer")

def build(source: Path, destination: Path, *, paths_root=None):
    with source.open("r",newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Need path,label,split,signer from authoritative dataset metadata")
        rows=[{key:row[key].strip() for key in FIELDS} for row in reader]
    validate_manifest(rows)
    root=Path(paths_root) if paths_root is not None else source.parent
    for row in rows:
        if not (root / row["path"]).is_file():
            raise FileNotFoundError(root / row["path"])
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows,key=lambda row:(row["split"],row["label"],row["path"])))
    return {"clips":len(rows),"signers":len({r["signer"] for r in rows}),"classes":len({r["label"] for r in rows})}

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,help="Dataset-provided CSV with signer and split columns")
    p.add_argument("--output",required=True)
    p.add_argument("--root",default=None)
    a=p.parse_args()
    print(build(Path(a.source),Path(a.output),paths_root=a.root))

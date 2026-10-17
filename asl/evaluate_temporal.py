"""Held-out temporal evaluation, never used for hyperparameter tuning."""
import argparse
import json
from pathlib import Path
import numpy as np
from .temporal import load_partition

def main():
    import tensorflow as tf
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--labels",required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--out",default="outputs/temporal/test_metrics.json")
    args=p.parse_args()
    labels=json.loads(Path(args.labels).read_text())["classes"]
    x,y=load_partition(Path(args.data)/"test",{name:i for i,name in enumerate(labels)})
    probs=tf.keras.models.load_model(args.model).predict(x,verbose=0)
    if probs.shape!=(len(y),len(labels)):
        raise ValueError("Prediction shape mismatch")
    pred=np.argmax(probs,axis=1)
    report=classification_report(y,pred,labels=list(range(len(labels))),target_names=labels,output_dict=True,zero_division=0)
    result={"samples":int(len(y)),"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"macro_f1":float(report["macro avg"]["f1-score"]),"per_class":report,"confusion_matrix":confusion_matrix(y,pred,labels=list(range(len(labels)))).tolist(),"classes":labels}
    target=Path(args.out)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ("per_class","confusion_matrix","classes")},indent=2))

if __name__=="__main__":
    main()

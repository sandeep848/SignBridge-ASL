"""Evaluate the EfficientNet static classifier against a disjoint test directory."""
import argparse
import json
from pathlib import Path
import numpy as np

def main():
    import tensorflow as tf
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--labels",required=True)
    p.add_argument("--data",required=True,help="Test image directory with one subfolder per class")
    p.add_argument("--out",default="outputs/static/test_metrics.json")
    p.add_argument("--batch-size",type=int,default=32)
    args=p.parse_args()
    classes=json.loads(Path(args.labels).read_text())["classes"]
    test=tf.keras.utils.image_dataset_from_directory(args.data,image_size=(224,224),shuffle=False,batch_size=args.batch_size,label_mode="int")
    if test.class_names!=classes:
        raise ValueError("Test classes and checkpoint labels disagree")
    model=tf.keras.models.load_model(args.model)
    probs=model.predict(test,verbose=0)
    truth=np.concatenate([labels.numpy() for _,labels in test])
    if probs.shape!=(len(truth),len(classes)):
        raise ValueError("Output shape does not match labels")
    pred=np.argmax(probs,axis=1)
    report=classification_report(truth,pred,labels=list(range(len(classes))),target_names=classes,output_dict=True,zero_division=0)
    result={"samples":int(len(truth)),"accuracy":float(accuracy_score(truth,pred)),"balanced_accuracy":float(balanced_accuracy_score(truth,pred)),"macro_f1":float(report["macro avg"]["f1-score"]),"per_class":report,"confusion_matrix":confusion_matrix(truth,pred,labels=list(range(len(classes)))).tolist(),"classes":classes}
    dest=Path(args.out)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2))
    print(json.dumps({"samples":result["samples"],"accuracy":result["accuracy"],"macro_f1":result["macro_f1"]},indent=2))

if __name__=="__main__":
    main()

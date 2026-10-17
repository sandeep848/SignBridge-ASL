"""Train a body-relative isolated-sign classifier on real precomputed features."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .holistic import DIM,SCHEMA,add_kinematics
from .temporal import FRAMES

FEATURES=2*DIM-4

def data(files,labels):
    for file in files:
        with np.load(file,allow_pickle=False) as sample:
            x=sample["points"].astype(np.float32)
            label=str(sample["label"].item())
        if x.ndim!=2 or x.shape[1]!=DIM or not np.isfinite(x).all():
            raise ValueError(f"Invalid holistic sample: {file}")
        from .temporal import resample_sequence
        indices=np.rint(np.linspace(0,len(x)-1,FRAMES)).astype(int)
        x=add_kinematics(x[indices])
        yield x,np.int32(labels[label])

def make_dataset(folder,lookup,batch,shuffle):
    import tensorflow as tf
    files=sorted(Path(folder).glob("*.npz"))
    if not files:
        raise ValueError(f"No holistic features in {folder}")
    ds=tf.data.Dataset.from_generator(lambda:data(files,lookup),output_signature=(
        tf.TensorSpec((FRAMES,FEATURES),tf.float32),tf.TensorSpec((),tf.int32)))
    if shuffle:
        ds=ds.shuffle(min(len(files),2048),seed=42,reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

def build_model(classes):
    import tensorflow as tf
    inp=tf.keras.Input((FRAMES,FEATURES),name="holistic_frames")
    # Per-feature normalization parameters must be fitted on training data only.
    x=tf.keras.layers.Dense(192,activation="gelu")(inp)
    x=tf.keras.layers.Conv1D(192,5,padding="same",activation="gelu")(x)
    x=tf.keras.layers.Bidirectional(tf.keras.layers.GRU(96,return_sequences=True))(x)
    x=tf.keras.layers.Add()([x,tf.keras.layers.MultiHeadAttention(num_heads=4,key_dim=48)(x,x)])
    x=tf.keras.layers.GlobalAveragePooling1D()(x)
    emb=tf.keras.layers.Dense(128,name="embedding")(x)
    out=tf.keras.layers.Dense(classes,activation="softmax",name="class_probabilities")(emb)
    return tf.keras.Model(inp,out,name="SignBridgeHolistic")

def main():
    import tensorflow as tf
    from .experiment import metadata,save
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--output",default="outputs/holistic")
    p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--batch-size",type=int,default=24)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    root=Path(args.data)
    files=sorted((root/"train").glob("*.npz"))
    labels=set()
    for file in files:
        with np.load(file,allow_pickle=False) as sample:
            labels.add(str(sample["label"].item()))
    classes=sorted(labels)
    if len(classes)<2:
        raise ValueError("At least two labels required")
    lookup={label:index for index,label in enumerate(classes)}
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    (out/"labels.json").write_text(json.dumps({"classes":classes},indent=2))
    save({"task":"holistic","labels":classes,"feature_schema":SCHEMA,
        "frames":FRAMES,"feature_dimensions":FEATURES,"seed":args.seed},out/"model_metadata.json")
    model=build_model(len(classes))
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                  loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    history=model.fit(make_dataset(root/"train",lookup,args.batch_size,True),
             validation_data=make_dataset(root/"val",lookup,args.batch_size,False),
             epochs=args.epochs,callbacks=[
                tf.keras.callbacks.ModelCheckpoint(str(out/"best.keras"),save_best_only=True,monitor="val_loss"),
                tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=7,restore_best_weights=True)])
    (out/"history.json").write_text(json.dumps({k:[float(v) for v in values] for k,values in history.history.items()},indent=2))

if __name__=="__main__":
    main()

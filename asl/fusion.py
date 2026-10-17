"""Image and hand-geometry fusion with explicit visibility masking."""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from .constants import IMAGE_SIZE

def build_fusion_model(classes, pretrained=True, backbone="mobilenet"):
    img = tf.keras.Input((IMAGE_SIZE,IMAGE_SIZE,3),name="image")
    coords = tf.keras.Input((63,),name="landmarks")
    visible = tf.keras.Input((1,),name="visible")
    options = {
        "mobilenet": tf.keras.applications.MobileNetV3Large,
        "efficientnet": tf.keras.applications.EfficientNetB0,
    }
    if backbone not in options:
        raise ValueError("Unsupported backbone")
    kwargs = {"include_preprocessing":True} if backbone == "mobilenet" else {}
    cnn = options[backbone](include_top=False,weights="imagenet" if pretrained else None,input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),**kwargs)
    cnn.trainable = False
    visual = tf.keras.layers.Dense(128,activation="relu")(tf.keras.layers.GlobalAveragePooling2D()(cnn(img,training=False)))
    geometric = tf.keras.layers.Dense(128,activation="relu")(coords)
    geometric = tf.keras.layers.Multiply()([geometric,visible])
    fused = tf.keras.layers.Concatenate()([visual,geometric,visible])
    fused = tf.keras.layers.Dense(128,activation="relu")(fused)
    fused = tf.keras.layers.Dropout(0.3)(fused)
    out = tf.keras.layers.Dense(classes,activation="softmax")(fused)
    net = tf.keras.Model([img,coords,visible],out)
    net.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return net

def normalized_landmarks(rgb,hands):
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return np.zeros(63,np.float32),np.array([0],np.float32)
    xyz = np.asarray([[p.x,p.y,p.z] for p in result.multi_hand_landmarks[0].landmark],dtype=np.float32)
    xyz -= xyz[0].copy()
    xyz /= max(float(np.max(np.linalg.norm(xyz[:,:2],axis=1))),1e-5)
    return xyz.ravel(),np.array([1],np.float32)

def image_files(path,labels):
    """Stream files and labels, never materialize an entire image corpus."""
    files=[]
    for index,label in enumerate(labels):
        part=path/label
        matches=sorted(p for p in part.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"})
        if not matches:
            raise ValueError(f"No images for {label} in {path}")
        files.extend((file,index) for file in matches)
    return files

def build_dataset(entries,batch,training=False):
    import cv2
    import mediapipe as mp
    from .preprocess import letterbox
    def generator():
        # Detection is intentionally stateless across individual images.
        with mp.solutions.hands.Hands(static_image_mode=True,max_num_hands=1) as hands:
            for file,label in entries:
                image=cv2.imread(str(file))
                if image is None:
                    raise ValueError(f"Unreadable image: {file}")
                rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                rgb=letterbox(rgb)
                xyz,mask=normalized_landmarks(rgb,hands)
                yield {"image":rgb.astype(np.float32),"landmarks":xyz,"visible":mask},np.int32(label)
    signature=({"image":tf.TensorSpec((IMAGE_SIZE,IMAGE_SIZE,3),tf.float32),
                "landmarks":tf.TensorSpec((63,),tf.float32),
                "visible":tf.TensorSpec((1,),tf.float32)},tf.TensorSpec((),tf.int32))
    data=tf.data.Dataset.from_generator(generator,output_signature=signature)
    if training:
        data=data.shuffle(min(2048,len(entries)),seed=42,reshuffle_each_iteration=True)
        def stochastic_augmentation(features,label):
            # Randomized for each example on every epoch; no fixed one-time mask.
            keep=tf.cast(tf.random.uniform((1,))>=0.2,tf.float32)
            visible=features["visible"]*keep
            return {"image":features["image"],"landmarks":features["landmarks"]*visible,"visible":visible},label
        data=data.map(stochastic_augmentation,num_parallel_calls=tf.data.AUTOTUNE)
    return data.batch(batch).prefetch(tf.data.AUTOTUNE)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/split")
    p.add_argument("--out",default="outputs/fusion")
    p.add_argument("--backbone",choices=["mobilenet","efficientnet"],default="mobilenet")
    p.add_argument("--epochs",type=int,default=25)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--no-pretrained",action="store_true")
    args=p.parse_args()
    tf.keras.utils.set_random_seed(42)
    root=Path(args.data)
    labels=sorted(p.name for p in (root/"train").iterdir() if p.is_dir())
    if len(labels)<2:
        raise ValueError("Need at least two classes")
    train_set=build_dataset(image_files(root/"train",labels),args.batch_size,training=True)
    val_set=build_dataset(image_files(root/"val",labels),args.batch_size)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    (out/"labels.json").write_text(json.dumps({"classes":labels},indent=2))
    net=build_fusion_model(len(labels),not args.no_pretrained,args.backbone)
    callbacks=[tf.keras.callbacks.ModelCheckpoint(str(out/"best.keras"),monitor="val_loss",save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=6,restore_best_weights=True)]
    history=net.fit(train_set,validation_data=val_set,epochs=args.epochs,callbacks=callbacks)
    (out/"history.json").write_text(json.dumps({k:[float(v) for v in values] for k,values in history.history.items()},indent=2))

if __name__=="__main__":
    main()

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

def partition(path, labels):
    import cv2
    import mediapipe as mp
    if not hasattr(mp,"solutions"):
        raise RuntimeError("MediaPipe solutions.hands is required")
    images,landmarks,masks,ys=[],[],[],[]
    with mp.solutions.hands.Hands(static_image_mode=True,max_num_hands=1) as hands:
        for idx,label in enumerate(labels):
            files=sorted(p for p in (path/label).iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"})
            if not files:
                raise ValueError(f"No images for {label}")
            for file in files:
                image=cv2.imread(str(file))
                if image is None:
                    raise ValueError(f"Unreadable: {file}")
                rgb=cv2.cvtColor(cv2.resize(image,(IMAGE_SIZE,IMAGE_SIZE)),cv2.COLOR_BGR2RGB)
                xyz,mask=normalized_landmarks(rgb,hands)
                images.append(rgb)
                landmarks.append(xyz)
                masks.append(mask)
                ys.append(idx)
    return (np.asarray(images,dtype=np.float32),np.asarray(landmarks,dtype=np.float32),np.asarray(masks,dtype=np.float32)),np.asarray(ys,dtype=np.int32)

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
    xt,yt=partition(root/"train",labels)
    xv,yv=partition(root/"val",labels)
    rng=np.random.default_rng(42)
    mask=xt[2].copy()
    mask[rng.random(len(mask))<0.2]=0
    train_x={"image":xt[0],"landmarks":xt[1]*mask,"visible":mask}
    val_x={"image":xv[0],"landmarks":xv[1],"visible":xv[2]}
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    (out/"labels.json").write_text(json.dumps({"classes":labels},indent=2))
    net=build_fusion_model(len(labels),not args.no_pretrained,args.backbone)
    callbacks=[tf.keras.callbacks.ModelCheckpoint(str(out/"best.keras"),monitor="val_loss",save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=6,restore_best_weights=True)]
    history=net.fit(train_x,yt,validation_data=(val_x,yv),batch_size=args.batch_size,epochs=args.epochs,callbacks=callbacks)
    (out/"history.json").write_text(json.dumps({k:[float(v) for v in values] for k,values in history.history.items()},indent=2))

if __name__=="__main__":
    main()

"""Export trained Keras checkpoints to TFLite, optionally with integer calibration.

This script does not claim that the exported model has been validated on hardware.
Use representative real training samples for full INT8 quantization.
"""
import argparse
import json
from pathlib import Path
import numpy as np

def convert(checkpoint, output, *, mode="float32", representative=None):
    import tensorflow as tf
    model=tf.keras.models.load_model(checkpoint)
    converter=tf.lite.TFLiteConverter.from_keras_model(model)
    if mode=="float16":
        converter.optimizations=[tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types=[tf.float16]
    elif mode=="int8":
        if representative is None:
            raise ValueError("INT8 requires a representative training dataset")
        with np.load(representative,allow_pickle=False) as data:
            # Named arrays input_0, input_1 etc. must match model inputs.
            arrays=[np.asarray(data[f"input_{i}"],dtype=np.float32) for i in range(len(model.inputs))]
        count=len(arrays[0])
        if count<32 or any(len(a)!=count for a in arrays):
            raise ValueError("At least 32 aligned representative samples required")
        def representative_data():
            for i in range(count):
                yield [a[i:i+1] for a in arrays]
        converter.optimizations=[tf.lite.Optimize.DEFAULT]
        converter.representative_dataset=representative_data
        converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type=tf.int8
        converter.inference_output_type=tf.int8
    elif mode!="float32":
        raise ValueError("mode must be float32, float16 or int8")
    buffer=converter.convert()
    dest=Path(output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(buffer)
    interpreter=tf.lite.Interpreter(model_path=str(dest))
    interpreter.allocate_tensors()
    result={"source_checkpoint":str(checkpoint),"mode":mode,"size_bytes":len(buffer),
      "inputs":[{"shape":x["shape"].tolist(),"dtype":str(x["dtype"])} for x in interpreter.get_input_details()],
      "outputs":[{"shape":x["shape"].tolist(),"dtype":str(x["dtype"])} for x in interpreter.get_output_details()],
      "accuracy_validated":False,"latency_validated":False}
    dest.with_suffix(".json").write_text(json.dumps(result,indent=2))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--mode",choices=["float32","float16","int8"],default="float32")
    p.add_argument("--representative",help="NPZ of actual training inputs: input_0, input_1...")
    a=p.parse_args()
    print(json.dumps(convert(a.checkpoint,a.output,mode=a.mode,representative=a.representative),indent=2))

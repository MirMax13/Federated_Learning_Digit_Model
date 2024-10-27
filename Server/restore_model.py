import tensorflow as tf
import numpy as np

SAVED_MODEL_DIR_CL = "classifier_model"
cl_model = tf.saved_model.load(SAVED_MODEL_DIR_CL)


def save_model():
    tf.saved_model.save(
        cl_model,
        SAVED_MODEL_DIR_CL,
        signatures={
            'train': cl_model.train.get_concrete_function(),
            'infer': cl_model.infer.get_concrete_function(),
            'save': cl_model.save.get_concrete_function(),
            'restore': cl_model.restore.get_concrete_function(),
        })
    
def convert_to_tflite():
    converter_cl = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR_CL)
    converter_cl.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
    converter_cl.experimental_enable_resource_variables = True
    converter_cl.allow_custom_ops = True
    cl_model_tflite = converter_cl.convert()
    cl_model_name = "init_classifier_model.tflite"
    with open(cl_model_name, "wb") as f:
        f.write(cl_model_tflite)
    print(f"TFLite model saved. First 10 bytes: {list(cl_model_tflite[:10])}")


cl_model.restore(checkpoint_path=np.array("./init_lite.ckpt", dtype=np.string_))

save_model()
convert_to_tflite()
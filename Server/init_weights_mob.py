import tensorflow as tf
import numpy as np
classifier_path = "./classifier_model.tflite"




classifier_interpreter = tf.lite.Interpreter(model_path=classifier_path)

classifier_interpreter.allocate_tensors()

# Отримання сигнатур
infer = classifier_interpreter.get_signature_runner("infer")
restore = classifier_interpreter.get_signature_runner("restore")
train = classifier_interpreter.get_signature_runner("train")
save = classifier_interpreter.get_signature_runner("save")


# save(checkpoint_path=np.array("./init_check.ckpt", dtype=np.string_))


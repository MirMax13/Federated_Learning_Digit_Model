from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import numpy as np
import os
import random
import string
import shutil as sh
from digit_model import Classifier

app = Flask(__name__)
checkpoints_dir = 'checkpoints'

feature_extractor_path = "feature_extractor_model.tflite"
classifier_path = "./classifier_model.tflite"

CHECKPOINT_DIR = './checkpoints'

SAVED_MODEL_DIR_FE = "feature_extractor_model"
SAVED_MODEL_DIR_CL = "classifier_model"

fe_model = tf.saved_model.load(SAVED_MODEL_DIR_FE)
num_classes = 10
cl_model = Classifier(num_classes)
img_size = 28
class_names = [str(i) for i in range(10)]

def convert_to_tflite():
    converter_cl = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR_CL)
    converter_cl.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
    converter_cl.experimental_enable_resource_variables = True
    converter_cl.allow_custom_ops = True
    cl_model_tflite = converter_cl.convert()
    cl_model_name = "classifier_model.tflite"
    with open(cl_model_name, "wb") as f:
        f.write(cl_model_tflite)
    print(f"TFLite model saved. First 10 bytes: {list(cl_model_tflite[:10])}")

def aggregate():
    checkpoint_files = [os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.ckpt')]

    if not checkpoint_files:
        raise ValueError("No checkpoint files found")
    
    all_weights = []

    for checkpoint in checkpoint_files:
        
        if not os.path.exists(checkpoint) or os.path.getsize(checkpoint) == 0:
            print(f"Checkpoint {checkpoint} is empty or does not exist", flush=True)
            continue
        cl_model.restore(checkpoint)

        
        all_weights.append(cl_model.model.get_weights())

    avg_weights = [np.mean(weights, axis=0) for weights in zip(*all_weights)]
    
    cl_model.model.set_weights(avg_weights)

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
    
@app.route('/load_model', methods=['GET'])
def load_model():
    aggregate()
    sh.rmtree(SAVED_MODEL_DIR_CL)
    save_model()
    convert_to_tflite()
    print("Model sending", flush=True)
    print(f"Model size: {os.path.getsize(classifier_path)} bytes")
    return send_file(classifier_path,as_attachment=True)

@app.route('/upload-weights', methods=['POST']) #TODO: restrict same weights to be uploaded
def upload_weights():
    os.makedirs(checkpoints_dir, exist_ok=True)
    weights = request.data
    random_filename = ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '.ckpt'
    with open(os.path.join(checkpoints_dir, random_filename), 'wb') as f:
        f.write(weights)
    return 'Weights received and saved', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
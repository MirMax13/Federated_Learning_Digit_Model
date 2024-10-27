import tensorflow as tf
import random
import numpy as np

img_height, img_width = 28, 28
batch_size = 20

SEED = 3
FILE_PATH_KEY = "FILE_PATH"
FILE_CLASS_KEY = "FILE_CLASS"
PATH_TO_TRAIN_DATA_ORIG = "./data_pc/training"
PATH_TO_TEST_DATA = "./data_pc/testing"
PATH_TO_TRAIN_DATA = PATH_TO_TEST_DATA

def set_global_seed(seed):
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

img_height, img_width = 28, 28
batch_size = 128

def normalize_and_convert_img(image, label):
    # Convert to grayscale by taking mean across color channels
    image = tf.image.rgb_to_grayscale(image)
    # Normalize to [0,1]
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = tf.keras.utils.image_dataset_from_directory(
    "data_pc/training",
    image_size = (img_height, img_width),
    batch_size = batch_size
).map(normalize_and_convert_img)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "data_pc/testing",
    image_size = (img_height, img_width),
    batch_size = batch_size
).map(normalize_and_convert_img)


IMG_SIZE = 28
num_classes = 10

class FeatureExtractor(tf.Module):
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE,1)),
            tf.keras.layers.Conv2D(32, (3, 3)),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(32, (3, 3)),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(64, (3, 3)),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(64, (3, 3)),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Flatten(),
        ])

    @tf.function(input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 1], tf.float32)])
    def extract(self, x):
        return self.model(x)

class Classifier(tf.Module):
    def __init__(self, num_classes):
        self.model = tf.keras.Sequential([
           tf.keras.layers.Dense(512, activation="relu", input_shape=(1024,)),
           tf.keras.layers.Dense(num_classes, activation="softmax")
        ])

        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

    @tf.function(input_signature=[
        tf.TensorSpec([None, 1024], tf.float32),
        tf.TensorSpec([None, num_classes], tf.float32),
    ])
    def train(self, x, y):
        with tf.GradientTape() as tape:
            prediction = self.model(x)
            loss = self.model.loss(y, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return {"loss": loss}

    @tf.function(input_signature=[tf.TensorSpec([None, 1024], tf.float32)])
    def infer(self, x):
        logits = self.model(x)
        return {
            "output": logits,
            "classes": tf.cast(tf.argmax(logits, axis=-1), tf.float32)
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save(self, checkpoint_path):
        tensor_names = [weight.name for weight in self.model.weights]
        tensors_to_save = [weight.read_value() for weight in self.model.weights]
        tf.raw_ops.Save(
            filename=checkpoint_path, tensor_names=tensor_names,
            data=tensors_to_save, name='save')
        return {"checkpoint_path": checkpoint_path}

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore(self, checkpoint_path):
        restored_tensors = {}
        for var in self.model.weights:
            restored = tf.raw_ops.Restore(
                file_pattern=checkpoint_path, tensor_name=var.name, dt=var.dtype,
                name='restore')
            var.assign(restored)
            restored_tensors[var.name] = restored
        return restored_tensors

    
if __name__ == "__main__":
    set_global_seed(SEED)
    feature_extractor = FeatureExtractor()
    classifier = Classifier(num_classes=10)

    # Підготовка параметрів
    BATCH_SIZE = 128
    NUM_EPOCHS = 2
    # Ітерація навчання
    for epoch in range(NUM_EPOCHS):
            for x, y in train_ds:
                y_one_hot = tf.one_hot(y, depth=num_classes)
                features = feature_extractor.extract(x)
                result = classifier.train(features, y_one_hot)
                print(f"Loss: {result['loss']:.4f}")

    weights = classifier.model.get_weights()
    for i, weight in enumerate(weights):
        print(f"Shape of weight {i}: {weight.shape}")


    SAVED_MODEL_DIR_FE = "feature_extractor_model"
    tf.saved_model.save(feature_extractor, SAVED_MODEL_DIR_FE, signatures={
        "extract":feature_extractor.extract.get_concrete_function()
    })

    SAVED_MODEL_DIR_CL = "classifier_model"
    tf.saved_model.save(classifier, SAVED_MODEL_DIR_CL, signatures={
        "train": classifier.train.get_concrete_function(),
        "infer": classifier.infer.get_concrete_function(),
        "save": classifier.save.get_concrete_function(),
        "restore": classifier.restore.get_concrete_function()
    })

    converter_fe = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR_FE)
    converter_fe.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
    converter_fe.experimental_enable_resource_variables = True

    fe_model_tflite = converter_fe.convert()
    fe_model_name = "feature_extractor_model.tflite"
    with open(fe_model_name, "wb") as f:
        f.write(fe_model_tflite)

    converter_cl = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR_CL)
    converter_cl.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
    converter_cl.experimental_enable_resource_variables = True

    cl_model_tflite = converter_cl.convert()
    cl_model_name = "classifier_model.tflite"
    with open(cl_model_name, "wb") as f:
        f.write(cl_model_tflite)
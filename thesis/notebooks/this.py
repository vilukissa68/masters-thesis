#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import pandas as pd
import wave

ROOT_PATH = Path("./")
MODEL_DIR = ROOT_PATH / "models"
IC_MODEL_PATH = MODEL_DIR / "pretrainedResnet_quant.tflite"
VWW_MODEL_PATH = MODEL_DIR / "vww_96_int8.tflite"
KWS_MODEL_PATH = MODEL_DIR / "kws_ref_model.tflite"
AD_MODEL_PATH = MODEL_DIR / "ad01_int8.tflite"
DATA_DIR = ROOT_PATH / "dev_data"
KWS_DATA_DIR = DATA_DIR / "kws01"
VWW_DATA_DIR = DATA_DIR / "vw_coco2014_96"
VWW_NON_PERSON_DATA_DIR =  VWW_DATA_DIR / "non_person"
VWW_PERSON_DATA_DIR = VWW_DATA_DIR / "person"
IC_DATA_DIR = DATA_DIR / "cifar-10-batches-py"
AD_DATA_DIR = DATA_DIR / "ToyCar" / "test"

def load_tflite_model(model_path):
    model = tf.lite.Interpreter(model_path=model_path, experimental_preserve_all_tensors=True)
    model.allocate_tensors()
    return model

def run_inference_for_sample(data, model):
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    model.set_tensor(input_details[0]['index'], np.round(data).astype(np.int8))

    model.invoke()

    output_data = model.get_tensor(output_details[0]['index'])
    return output_data

def accuracy_report(gt, prediction):
    print("Accuracy: {:.3f}".format(accuracy_score(gt, prediction)))
    print("Confusion matrix:\n{}".format(confusion_matrix(gt, prediction)))
    print(classification_report(gt, prediction))



def run_ic(model, total_samples=200):
    import pickle
    with open(IC_DATA_DIR / "test_batch", "rb") as file:
        data = pickle.load(file, encoding='bytes')

    images = data[b'data']
    labels = data[b'labels']

    class_samples = {i: [] for i in range(10)}
    samples_per_class = total_samples // len(class_samples)

    # Find samples
    for img, label in zip(images, labels):
        if len(class_samples[label]) < samples_per_class:
            class_samples[label].append(img)
        if all(len(class_samples[c]) == samples_per_class for c in class_samples):
            break

    selected_images = []
    selected_labels = []
    for label, samples in class_samples.items():
        print(label)
        print(samples)
        selected_images.extend(samples)
        selected_labels.extend([label] * len(samples))

    # Convert to numpy
    selected_images = np.array(selected_images, dtype=np.int8)
    selected_labels = np.array(selected_labels)

    predictions = []

    # Run inference on samples
    for (i, image) in enumerate(selected_images):
        #FROM CHW to HWC
        image = np.reshape(image, (3, 32, 32))
        image = np.rollaxis(image, 0, 3)
        image = np.expand_dims(image, axis=0)
        image = image - 128
        label = selected_labels[i]

        prediction = run_inference_for_sample(image, model)
        print(prediction)
        predictions.append(np.argmax(prediction))

        # Mid-run report
        #accuracy_report(selected_labels[:len(predictions)], predictions)

    print("Final accuracy report for Image Classification:")
    accuracy_report(selected_labels, predictions)

model = load_tflite_model(str(IC_MODEL_PATH.resolve()))
run_ic(model)

def read_vww_file(path):
    #Image loading and preprocessing
    image = tf.io.read_file(str(path))
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [96,96])
    image = np.array(image, dtype=np.int8)
    image = np.expand_dims(image, axis=0)
    image = image + 128
    return image.astype(np.int8)

def run_vww(model, total_samples=200):
    items = os.listdir(VWW_NON_PERSON_DATA_DIR)
    non_persons = [item for item in items if os.path.isfile(os.path.join(VWW_NON_PERSON_DATA_DIR, item)) and item.startswith("COCO_val")]
    non_persons.sort()

    items = os.listdir(VWW_PERSON_DATA_DIR)
    persons = [item for item in items if os.path.isfile(os.path.join(VWW_PERSON_DATA_DIR, item)) and item.startswith("COCO_val")]
    persons.sort()

    print("Number of non_persons", len(non_persons))
    print("Number of persons", len(persons))
    print("Input shape: ", np.shape(read_vww_file(VWW_NON_PERSON_DATA_DIR / non_persons[0])))

    # Calculate balanced number of samples for each category
    samples_per_class = min(len(non_persons), len(persons), total_samples // 2)

    # Select samples for each category
    selected_non_persons = non_persons[:samples_per_class]
    selected_persons = persons[:samples_per_class]

    # Generate ground truth array
    gt = []

    predictions = []
    for non_person, person in zip(selected_non_persons, selected_persons):
        # Non person sample
        data_non = read_vww_file(VWW_NON_PERSON_DATA_DIR / non_person)
        prediction = run_inference_for_sample(data_non, model)
        predictions.append(np.argmax(prediction))
        gt.append(0)

        # Person sample
        data_person = read_vww_file(VWW_PERSON_DATA_DIR / person)
        prediction = run_inference_for_sample(data_person, model)
        predictions.append(np.argmax(prediction))
        gt.append(1)

        # Mid-run report
        #accuracy_report(gt, predictions)

    print("Final accuracy report for Visual Wakeup Word:")
    accuracy_report(gt, predictions)

model = load_tflite_model(str(VWW_MODEL_PATH.resolve()))
run_vww(model)


def read_kws_file(path):
    with open(path, mode="rb") as file:
        content = file.read()
    return content

def run_kws(model):
    total_samples=200
    df = pd.read_csv(KWS_DATA_DIR / "y_labels.csv", names=["filename", "no_classes", "class"])

    class_counts = df["class"].value_counts()

    print(f"Classes in dataset: {class_counts}")

    num_classes = len(class_counts)

    # Calculate the number of samples per class, and how many extra samples to distribute
    base_samples_per_class = total_samples // num_classes
    extra_samples = total_samples % num_classes

    balanced_df = pd.DataFrame()
    for class_id in class_counts.index:
        class_samples = df[df["class"] == class_id].sample(base_samples_per_class)
        balanced_df = pd.concat([balanced_df, class_samples])

    remaining_samples_needed = extra_samples
    for class_id in class_counts.index:
        if remaining_samples_needed == 0:
            break
        # If there are more remaining samples, sample one more from this class
        class_samples = df[df["class"] == class_id].sample(1)
        balanced_df = pd.concat([balanced_df, class_samples])
        remaining_samples_needed -= 1

    # Shuffle the resulting balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42)

    predictions = []
    for (i, filename) in enumerate(balanced_df["filename"]):
        data = read_kws_file(KWS_DATA_DIR / filename)
        data = np.array(list(data))
        data = data.reshape(1,49,10,1)


        prediction = run_inference_for_sample(data, model)
        predictions.append(np.argmax(prediction))

        # Mid run report
        #accuracy_report(balanced_df["class"][:len(predictions)], predictions)

    print("Final accuracy report for Keyword Spotting:")
    accuracy_report(balanced_df["class"], predictions)

model = load_tflite_model(str(KWS_MODEL_PATH.resolve()))
run_kws(model)

import torch
import matplotlib.pyplot as plt

A = torch.tensor([
    [-32, 64, 96],
    [18, -128, 124],
    [64, 76, -127],
], dtype=torch.int32).unsqueeze(0).unsqueeze(0)

B = torch.tensor([[1, 2],
                       [-1, 3]], dtype=torch.int32).unsqueeze(0).unsqueeze(0)  # Add output and input channel dims
#B = B * 10

bias = torch.tensor([0], dtype=torch.int32)  # Single bias for one output channel
output = torch.nn.functional.conv2d(A, B, bias=bias, stride=1, padding=0)
avg = []
for clip_pp in range(0, 9):
    print("Clipping:", clip_pp)
    clipped = torch.floor(output / pow(2, clip_pp))
    clamped = torch.clamp(clipped, min=-128, max=127)
    returned = clamped * pow(2, clip_pp)
    # print("Correct:", output)
    print("Clipped:", clipped)
    # print("Clamped:", clamped)
    #print("Returned:", returned)
    #print("Difference:", np.abs(output - returned))
    print("Relative diff:",   (output-returned) / output)
    print("Avg error:", np.average((np.abs(output-returned) / np.abs(output))))
    avg.append(np.average((np.abs(output-returned) / np.abs(output))))
    print("")

plt.plot(avg, marker='o', linestyle=':', color='r', label='Error')
plt.xlabel('Bits clipped')
plt.ylabel('Error')
#plt.legend()
plt.savefig("/Users/vainogranat/clipping.eps", format='eps', dpi=300)  # Specify file name, format, and resolution
plt.show()

def read_ad_file(path):
    wav = wave.open(path)
    samples = wav.getnframes()
    audio = wav.readframes(samples)
    audio_as_np_int8 = numpy.frombuffer(audio, dtype=numpy.int8)
    return audio_as_np_int8

def run_anomaly_detection():


import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Add
from tensorflow.keras import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow_model_optimization.quantization.keras import quantize_model, quantize_annotate_layer, quantize_scope
import random

# Quantization-friendly Clipped ReLU using native TensorFlow operations
def clipped_relu(x, max_value=127.0):
    return tf.keras.layers.ReLU(max_value=max_value)(x)

# Residual block with clipping and QAT annotation
def resnet_block(x, filters, strides=1):
    shortcut = x
    x = quantize_annotate_layer(Conv2D(filters, kernel_size=3, strides=strides, padding='same'))(x)
    x = quantize_annotate_layer(BatchNormalization())(x)
    x = clipped_relu(x, max_value=127.0)

    x = quantize_annotate_layer(Conv2D(filters, kernel_size=3, strides=1, padding='same'))(x)
    x = quantize_annotate_layer(BatchNormalization())(x)
    x = clipped_relu(x, max_value=127.0)

    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = quantize_annotate_layer(Conv2D(filters, kernel_size=1, strides=strides, padding='same'))(shortcut)
    x = Add()([x, shortcut])
    return x

# Build a simple ResNet-like model with QAT annotations
def build_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = quantize_annotate_layer(Conv2D(64, kernel_size=3, strides=1, padding='same'))(inputs)
    x = quantize_annotate_layer(BatchNormalization())(x)
    x = clipped_relu(x, max_value=127.0)

    # Add a few residual blocks
    x = resnet_block(x, 64)
    x = resnet_block(x, 128, strides=2)
    x = resnet_block(x, 256, strides=2)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = quantize_annotate_layer(tf.keras.layers.Dense(num_classes, activation='softmax'))(x)

    return Model(inputs, outputs)

# Prepare data
def preprocess_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

def data_gen():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('int8')
    yield [x_train[random.randint(0, len(x_train))]]

# Train and save the QAT-enabled TFLite model
def train_and_save_model():
    input_shape = (32, 32, 3)
    num_classes = 10
    x_train, y_train, x_test, y_test = preprocess_data()

    with quantize_scope():
        base_model = build_model(input_shape, num_classes)
        qat_model = quantize_model(base_model)

    qat_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    qat_model.summary()

    qat_model.fit(x_train, y_train, epochs=1, batch_size=64, validation_data=(x_test, y_test))

    # Convert and save the quantized model to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = data_gen

    # Specify supported ops and target spec
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open("resnet_clipped_qat.tflite", "wb") as f:
        f.write(tflite_model)

    print("DONE!")

train_and_save_model()

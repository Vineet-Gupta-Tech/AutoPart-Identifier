import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
from PIL import Image

# ───────── CONFIG ─────────
IMG_SIZE = (224, 224)
NUM_CLASSES = 4
CLASS_NAMES = ['bolt', 'locatingpin', 'nut', 'washer']
WEIGHTS_PATH = "effnet_weights_final.h5"

# ───────── MODEL SETUP ─────────
@st.cache_resource
def load_model():
    base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    base.trainable = False
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.load_weights(WEIGHTS_PATH)
    return model

model = load_model()

# ───────── STREAMLIT UI ─────────
st.title("🔩 Mechanical Part Identifier")
st.write("Upload an image of a mechanical part (bolt, nut, washer, locating pin), and we'll tell you what it is.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    resized = image.resize(IMG_SIZE)
    arr = np.array(resized)
    arr = preprocess_input(arr.astype(np.float32))
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)[0]
    class_idx = np.argmax(pred)
    confidence = pred[class_idx] * 100

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.success(f"Prediction: **{CLASS_NAMES[class_idx]}** ({confidence:.2f}%)")

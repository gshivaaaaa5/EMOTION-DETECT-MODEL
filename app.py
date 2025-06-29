import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Face Emotion Detection with Grad-CAM", layout="wide")
st.title("😊 Face Emotion Detection App with Grad-CAM")

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("emotion_model.keras")
        dummy = np.random.rand(1, 48, 48, 1)
        model.predict(dummy)  # warm-up
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.warning(f"❗ Model load failed. Using fallback demo model.\n\n{e}")
        demo_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(48, 48, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
        return demo_model

model = load_model()

# Grad-CAM function
def grad_cam(img_array, model, last_conv_layer_name="conv2d_5"):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    return heatmap, predictions[0].numpy()

# Image preprocessing
def preprocess_image(image):
    image = image.convert("L").resize((48, 48))
    img_array = np.expand_dims(np.expand_dims(np.array(image), axis=-1), axis=0) / 255.0
    return img_array

# File uploader
uploaded_file = st.file_uploader("📸 Upload a face image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=250)
    img = Image.open(uploaded_file)
    img_array = preprocess_image(img)

    heatmap, prediction = grad_cam(img_array, model)
    pred_label = emotion_labels[np.argmax(prediction)]
    confidence = 100 * np.max(prediction)

    st.markdown(f"### 🔮 Predicted Emotion: **{pred_label.upper()}** ({confidence:.2f}%)")

    # Grad-CAM Overlay
    heatmap = cv2.resize(heatmap, (img.width, img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = np.array(img.resize((img.width, img.height)).convert("RGB"))
    superimposed_img = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    st.image(superimposed_img, caption="🔍 Grad-CAM Heatmap", use_column_width=True)

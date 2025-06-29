import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the trained model
model = tf.keras.models.load_model("emotion_model.h5")

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Predict custom images
def predict_image(image_path):
    img = tf.keras.utils.load_img(image_path, color_mode='grayscale', target_size=(48, 48))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)
    label = emotion_labels[np.argmax(pred)]
    confidence = np.max(pred) * 100
    
    plt.imshow(img, cmap='gray')
    plt.title(f"{label} ({confidence:.1f}%)")
    plt.axis('off')
    plt.show()
    print(f"Predicted: {label} | Confidence: {confidence:.1f}%")

# Predict your custom images
custom_images = [
    "test-pic/test5.jpg"
]

for img_path in custom_images:
    if os.path.exists(img_path):
        predict_image(img_path)
    else:
        print(f"Image not found: {img_path}")

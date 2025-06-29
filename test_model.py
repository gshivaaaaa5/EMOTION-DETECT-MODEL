import tensorflow as tf
import numpy as np
import os

# Load the trained model
model = tf.keras.models.load_model("emotion_model.h5")

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "face-dataset/test",
    image_size=(48, 48),
    color_mode='grayscale',
    label_mode='categorical',
    batch_size=64,
    shuffle=False  # Do not shuffle for evaluation
)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

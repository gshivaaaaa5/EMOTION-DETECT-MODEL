import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model("emotion_model_functional.h5")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load a batch from the test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "face-dataset/test",
    image_size=(48, 48),
    color_mode='grayscale',
    label_mode='categorical',
    batch_size=8,
    shuffle=True,
    seed=42
)

# Get one batch of images and labels
for images, labels in test_ds.take(1):
    preds = model.predict(images)

    for i in range(len(preds)):
        true_label = emotion_labels[np.argmax(labels[i])]
        pred_label = emotion_labels[np.argmax(preds[i])]
        confidence = np.max(preds[i]) * 100
        print(f"🧠 True: {true_label} | 🔮 Predicted: {pred_label} ({confidence:.1f}%)")

        # Optional: show image
        import matplotlib.pyplot as plt
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        plt.title(f"True: {true_label} | Predicted: {pred_label}")
        plt.axis('off')
        plt.show()

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Load the functional model
model = tf.keras.models.load_model("emotion_model_functional.h5")
print("✅ Functional model loaded successfully!")

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Ensure the model is built
dummy_input = tf.random.normal((1, 48, 48, 1))
_ = model(dummy_input)
print("✅ Model warmed up with dummy input!")

# Optional: print layer names
print("\nModel layers:")
for i, layer in enumerate(model.layers):
    print(f"{i}: {layer.name}")

# Grad-CAM function
def grad_cam(image_path, last_conv_layer_name='conv2d_5'):
    try:
        img = tf.keras.utils.load_img(image_path, color_mode='grayscale', target_size=(48, 48))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

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
        heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1).numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        # Read original image
        original_img = cv2.imread(image_path)
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)

        # Get label + confidence
        label = emotion_labels[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        # Display
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Prediction: {label} ({confidence:.1f}%)")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
        plt.title("Grad-CAM Heatmap")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")

# --- Run on test dataset images ---
test_images = []
for emotion in emotion_labels:
    folder = f"face-dataset/test/{emotion}"
    if os.path.exists(folder):
        files = os.listdir(folder)
        if files:
            test_images.append(os.path.join(folder, files[0]))

print("\n🔍 Running Grad-CAM on test dataset:")
for img in test_images[:3]:
    print(f"\nProcessing: {img}")
    grad_cam(img)

# --- Run on custom images ---
custom_imgs = ["test-pic/test1.jpg", "test-pic/test2.jpg", "test-pic/test3.jpg"]
print("\n🔍 Running Grad-CAM on custom images:")
for img in custom_imgs:
    if os.path.exists(img):
        print(f"\nProcessing: {img}")
        grad_cam(img)
    else:
        print(f"❌ File not found: {img}")

print("\n✅ All images processed.")

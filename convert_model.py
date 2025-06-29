# convert_model.py
import tensorflow as tf

# Load the old model (.h5 format)
old_model = tf.keras.models.load_model("emotion_model_functional.h5")

# Save it in new .keras format
old_model.save("emotion_model.keras", save_format="keras")

print("✅ Model converted to emotion_model.keras")

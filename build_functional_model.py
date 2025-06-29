import tensorflow as tf
from tensorflow.keras import layers, models

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Build Functional model
input_img = tf.keras.Input(shape=(48, 48, 1))

x = layers.Rescaling(1./255)(input_img)

x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.25)(x)

x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.25)(x)

x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.25)(x)

x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(len(emotion_labels), activation='softmax')(x)

# Define functional model
functional_model = models.Model(inputs=input_img, outputs=output)

# Load weights from your Sequential model
functional_model.load_weights("emotion_model.h5")
print("✅ Weights loaded into functional model!")

# Save as new H5 model
functional_model.save("emotion_model_functional.h5")
print("✅ Functional model saved as 'emotion_model_functional.h5'")

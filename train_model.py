import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os

# Emotion labels matching FER2013
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# Load dataset with augmentation
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "face-dataset/train",
    image_size=(48, 48),
    color_mode='grayscale',
    label_mode='categorical',
    batch_size=64,
    shuffle=True,
    seed=42
).map(lambda x, y: (data_augmentation(x, training=True), y))

# Calculate class weights
class_counts = np.array([len(os.listdir(f"face-dataset/train/{label}")) for label in emotion_labels])
class_weights = {i: max(class_counts)/count for i, count in enumerate(class_counts)}

# Improved CNN model
model = tf.keras.Sequential([
    layers.Rescaling(1./255, input_shape=(48, 48, 1)),
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(len(emotion_labels), activation='softmax')
])

# Compile with learning rate schedule
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=1000, decay_rate=0.96, staircase=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
]

# Train the model
history = model.fit(
    train_ds,
    epochs=50,
    class_weight=class_weights,
    callbacks=callbacks
)

# Save final model
model.save("emotion_model.h5")
print("Model saved to emotion_model.h5")

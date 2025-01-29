import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

'''
    Hiruna Devadithya, Vaibhav Alaparthi, Ayush Vupalanchi
    1/24/25 

    This class creates the camera CNN model and accesses the training and validation data from the kaggle dataset
'''

# Define constants
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 15
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
NUM_CLASSES = len(EMOTIONS)
TRAIN_DIR = "data\\train"
VAL_DIR = "data\\test"

# Load dataset using ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Build CNN model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
print("Model summary:")
model.summary()

# Train model
print("Training model...")
history = model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator)

# Save model
model.save("emotion_model.keras")
print("Model saved as 'emotion_model.keras'")

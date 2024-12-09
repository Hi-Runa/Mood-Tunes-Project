import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Define paths for training and validation datasets
angry_path = 'images/images/train/angry'
disgust_path = 'images/images/train/disgust'
fear_path = 'images/images/train/fear'
happy_path = 'images/images/train/happy'
neutral_path = 'images/images/train/neutral'
sad_path = 'images/images/train/sad'
surprise_path = 'images/images/train/surprise'

angry_val_path = 'images/images/validation/angry'
disgust_val_path = 'images/images/validation/disgust'
fear_val_path = 'images/images/validation/fear'
happy_val_path = 'images/images/validation/happy'
neutral_val_path = 'images/images/validation/neutral'
sad_val_path = 'images/images/validation/sad'
surprise_val_path = 'images/images/validation/surprise'

# Function to load images from a directory and assign a label
def load_images_from_path(path, label):
    images = []
    labels = []
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (48, 48))
            images.append(img)
            labels.append(label)
    return images, labels

# Load datasets
categories = [
    (angry_path, 0), (disgust_path, 1), (fear_path, 2), (happy_path, 3),
    (neutral_path, 4), (sad_path, 5), (surprise_path, 6)
]

validation_categories = [
    (angry_val_path, 0), (disgust_val_path, 1), (fear_val_path, 2), (happy_val_path, 3),
    (neutral_val_path, 4), (sad_val_path, 5), (surprise_val_path, 6)
]

train_images, train_labels = [], []
val_images, val_labels = [], []

for path, label in categories:
    images, labels = load_images_from_path(path, label)
    train_images.extend(images)
    train_labels.extend(labels)

for path, label in validation_categories:
    images, labels = load_images_from_path(path, label)
    val_images.extend(images)
    val_labels.extend(labels)

# Convert lists to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
val_images = np.array(val_images)
val_labels = np.array(val_labels)

# Normalize image data
train_images = train_images / 255.0
val_images = val_images / 255.0

# Reshape for model input
train_images = train_images.reshape(train_images.shape[0], 48, 48, 1)
val_images = val_images.reshape(val_images.shape[0], 48, 48, 1)

# One-hot encode labels
train_labels = to_categorical(train_labels, num_classes=7)
val_labels = to_categorical(val_labels, num_classes=7)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, 
                    validation_data=(val_images, val_labels), 
                    epochs=25, 
                    batch_size=64)

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

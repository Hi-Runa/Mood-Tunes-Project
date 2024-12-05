import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Configuration
base_path = "images/"
model_save_path = "./best_model.h5"
pic_size = 48
batch_size = 64

# Step 1: Data Augmentation and Generators
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    os.path.join(base_path, "train"),
    target_size=(pic_size, pic_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical"
)

validation_generator = validation_datagen.flow_from_directory(
    os.path.join(base_path, "validation"),
    target_size=(pic_size, pic_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical"
)

# Step 2: Define a Highly Accurate CNN Model
def create_model(input_shape=(pic_size, pic_size, 1), num_classes=7):
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Instantiate the model
model = create_model()

# Step 3: Configure Callbacks for Efficient Training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1),
    ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', verbose=1)
]

# Step 4: Train the Model
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=callbacks
)

# Step 5: Save the Model
model.save(model_save_path)
print(f"Model saved at {model_save_path}")

# Step 6: Visualize Training Performance
def plot_training(history):
    plt

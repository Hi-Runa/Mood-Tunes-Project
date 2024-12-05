# Disable warnings in the notebook to maintain clean output cells
import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
import os
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
import pandas as pd
import random
import urllib.request
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from deepface import DeepFace

# Configure the visual appearance of Seaborn plots
sns.set(rc={'axes.facecolor': '#f0f2f2'}, style='darkgrid')

# After extracting, specify the directories of the actual images
with_mask_path = '/kaggle/input/mask-detection/with_mask'
without_mask_path = '/kaggle/input/mask-detection/without_mask'

# List all filenames in the 'with_mask' and 'without_mask' directories
with_mask_images = os.listdir(with_mask_path)
without_mask_images = os.listdir(without_mask_path)

# Get the count of each class
with_mask_count = len(with_mask_images)
without_mask_count = len(without_mask_images)

# Calculate the percentages
total_images = with_mask_count + without_mask_count
with_mask_percentage = (with_mask_count / total_images) * 100
without_mask_percentage = (without_mask_count / total_images) * 100

# Plotting
labels = ['With Mask', 'Without Mask']
counts = [with_mask_count, without_mask_count]
percentages = [with_mask_percentage, without_mask_percentage]

# Set the figure size
plt.figure(figsize=(15, 4))

# Create a horizontal bar plot
ax = sns.barplot(y=labels, x=counts, orient='h', color='#008281')

# Set x-axis interval
ax.set_xticks(range(0, max(counts) + 500, 500))  

# Annotate each bar with the count and percentage
for i, p in enumerate(ax.patches):
    width = p.get_width()
    ax.text(width + 5, p.get_y() + p.get_height()/2., 
            '{:1.2f}% ({})'.format(percentages[i], counts[i]),
            va="center", fontsize=15)
    
# Set the x-label for the plot
plt.xlabel('Number of Images', fontsize=14)

# Set the title and show the plot
plt.title("Number of images per class (Masked vs Unmasked)", fontsize=18)
plt.show()

# Lists to store heights and widths of all images
heights = []
widths = []

# Initialize a set to store unique dimensions and channels
unique_dims = set()
unique_channels = set()

# Function to process images in a directory
def process_images(image_path):
    filenames = os.listdir(image_path)
    for filename in filenames:
        img = cv2.imread(os.path.join(image_path, filename))
        if img is not None:
            # Add the dimensions (height, width, channels) to the set
            unique_dims.add((img.shape[0], img.shape[1]))
            
            # Add the channels to the set
            unique_channels.add(img.shape[2])
            
            # Append heights and widths for statistical calculations
            heights.append(img.shape[0])
            widths.append(img.shape[1])

# Process images in both 'with_mask' and 'without_mask' directories
process_images(with_mask_path)
process_images(without_mask_path)

# Check if all images have the same dimension
if len(unique_dims) == 1:
    print(f"All images have the same dimensions: {list(unique_dims)[0]}")
else:
    print(f"There are {len(unique_dims)} different image dimensions in the dataset.")
    print(f"Min height: {min(heights)}, Max height: {max(heights)}, Mean height: {np.mean(heights):.2f}")
    print(f"Min width: {min(widths)}, Max width: {max(widths)}, Mean width: {np.mean(widths):.2f}")

# Check if all images have the same number of channels
if len(unique_channels) == 1:
    channel = list(unique_channels)[0]
    if channel == 3:
        print("All images are color images.")
    else:
        print("All images have the same number of channels, but they are not color images.")
else:
    print("Images have different numbers of channels.")


# Function to plot images
def plot_images(images, title, path):
    plt.figure(figsize=(15, 3))
    for i, img_name in enumerate(images):
        plt.subplot(1, 6, i + 1)
        img = cv2.imread(os.path.join(path, img_name))
        # Resize the image to a fixed size (e.g., 224x224)
        img = cv2.resize(img, (224, 224))
        # Convert the BGR image (default in OpenCV) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(title, fontsize=20)
    plt.show()

# Randomly select 6 images from each category
random_with_mask_images = random.sample(with_mask_images, 6)
random_without_mask_images = random.sample(without_mask_images, 6)

# Plot the images
plot_images(random_with_mask_images, "Randomly Selected With-Mask Images", with_mask_path)
plot_images(random_without_mask_images, "Randomly Selected Without-Mask Images", without_mask_path)


# Initialize an empty list to store image file paths and their respective labels
data = []

# Append the 'with_mask' image file paths with label "with_mask" to the data list
data.extend([(os.path.join(with_mask_path, filename), "with_mask") for filename in os.listdir(with_mask_path)])

# Append the 'without_mask' image file paths with label "without_mask" to the data list
data.extend([(os.path.join(without_mask_path, filename), "without_mask") for filename in os.listdir(without_mask_path)])

# Convert the collected data into a DataFrame
df = pd.DataFrame(data, columns=['filepath', 'label'])

# Display the first few entries of the DataFrame
df.head()

print("Total number of images in the dataset:", len(df))

# Deleting unnecessary variables to free up memory
del heights, widths, unique_dims, unique_channels, data, with_mask_images, without_mask_images

# Split the data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Display the shape of the training and validation sets
print("Training data shape:", train_df.shape)
print("Validation data shape:", val_df.shape)

# Deleting the original DataFrame to free up memory
del df

# Display the first few rows of the train DataFrame
train_df.head()

def create_data_generators(train_df, val_df, filepath_column='filepath', label_column='label', 
                           preprocessing_function=None, batch_size=32, image_dimensions=(299, 299)):
    """
    Creates and returns training and validation data generators with optional preprocessing.
    
    Parameters:
    - train_df (DataFrame): DataFrame containing training data.
    - val_df (DataFrame): DataFrame containing validation data.
    - filepath_column (str, optional): Name of the column in DataFrame containing file paths. Defaults to 'filepath'.
    - label_column (str, optional): Name of the column in DataFrame containing labels. Defaults to 'label'.
    - preprocessing_function (function, optional): Preprocessing function specific to a model. Defaults to None.
    - batch_size (int, optional): Number of images per batch for the generators. Defaults to 32.
    - image_dimensions (tuple, optional): Dimensions to which the images will be resized (height, width). Defaults to (299, 299).
    
    Returns:
    - train_generator (ImageDataGenerator): Generator for training data with augmentations.
    - val_generator (ImageDataGenerator): Generator for validation data without augmentations.
    
    Notes:
    - The training generator uses augmentations.
    - The validation generator does not use any augmentations.
    - If provided, the preprocessing function is applied to both generators.
    """

    # Define our training data generator with specific augmentations
    train_datagen = ImageDataGenerator(
        rotation_range=15,                             # Randomly rotate the images by up to 15 degrees
        width_shift_range=0.15,                        # Randomly shift images horizontally by up to 15% of the width
        height_shift_range=0.15,                       # Randomly shift images vertically by up to 15% of the height
        zoom_range=0.15,                               # Randomly zoom in or out by up to 15%
        horizontal_flip=True,                          # Randomly flip images horizontally
        vertical_flip=False,                           # Do not flip images vertically as it doesn't make sense in our context
        shear_range=0.02,                              # Apply slight shear transformations
        preprocessing_function=preprocessing_function  # Apply preprocessing function if provided
    )

    # Define our validation data generator without any augmentations but with the preprocessing function if provided
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function
    )

    # Create an iterable generator for training data
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,                 # DataFrame containing training data
        x_col=filepath_column,              # Column with paths to image files
        y_col=label_column,                 # Column with image labels
        target_size=image_dimensions,       # Resize all images to size of 224x224 
        batch_size=batch_size,              # Number of images per batch
        class_mode='binary',                # Specify binary classification task
        seed=42,                            # Seed for random number generator to ensure reproducibility
        shuffle=True                        # Shuffle the data to ensure the model gets a randomized batch during training
    )

    # Create an iterable generator for validation data
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col=filepath_column,
        y_col=label_column,
        target_size=image_dimensions,
        batch_size=batch_size,
        class_mode='binary',
        seed=42,
        shuffle=False
    )
    
    # Return the training and validation generators
    return train_generator, val_generator

# Load the InceptionV3 model pre-trained on ImageNet data, excluding the top classifier
inceptionv3_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Following the same pattern to add new layers to InceptionV3 base
x = inceptionv3_base.output
x = GlobalAveragePooling2D()(x) 
x = Dense(1024, activation='relu')(x)  
x = Dropout(0.5)(x)  
predictions = Dense(1, activation='sigmoid')(x)

# This is the model we will train
inceptionv3_model = Model(inputs=inceptionv3_base.input, outputs=predictions)

# Compile the model after setting layers to non-trainable
inceptionv3_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# MobileNetV2 model summary
inceptionv3_model.summary()

plot_model(inceptionv3_model, show_shapes=True, show_layer_names=False, dpi=100)

# Define number of epochs
num_epochs = 10

# Create data generators
train_generator, val_generator = create_data_generators(train_df, 
                                                        val_df, 
                                                        preprocessing_function=inceptionv3_preprocess_input, 
                                                        batch_size=32, 
                                                        image_dimensions=(299, 299))

# Define the callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True, verbose=1)

# Train the model
history = inceptionv3_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=num_epochs,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[reduce_lr, early_stopping]
)


# Convert the history.history dict to a pandas DataFrame for easy plotting
hist = pd.DataFrame(history.history)

# Plotting the learning curves
plt.figure(figsize=(15,6))

# Plotting the training and validation loss
plt.subplot(1, 2, 1)
sns.lineplot(x=hist.index+1, y=hist['loss'], color='#006766', label='Train Loss', marker='o', linestyle='--')
sns.lineplot(x=hist.index+1, y=hist['val_loss'], color='orangered', label='Validation Loss', marker='o', linestyle='--')
plt.title('Loss Evolution')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(0,num_epochs+1))

# Plotting the training and validation accuracy
plt.subplot(1, 2, 2)
sns.lineplot(x=hist.index+1, y=hist['accuracy'], color='#006766', label='Train Accuracy', marker='o', linestyle='--')
sns.lineplot(x=hist.index+1, y=hist['val_accuracy'], color='orangered', label='Validation Accuracy', marker='o', linestyle='--')
plt.title('Accuracy Evolution')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(range(0,num_epochs+1))

plt.show()

# Evaluates the model on the validation set
results = inceptionv3_model.evaluate(val_generator, steps=len(val_generator))

# The 'results' list contains loss as the first element and accuracy as the second
accuracy = results[1]

# Print the model accuracy on validation set
print(f'Validation Accuracy: %{round(100*accuracy,2)}')

def analyze_mask_in_image(image, model):
    """
    Processes an image and predicts the likelihood of a face mask being present using the provided model.

    Parameters:
    - image (numpy array): The image containing the face to analyze.
    - model (tensorflow.keras.models.Model): The pre-trained model for mask detection.

    Returns:
    - prediction (float): The predicted probability that the image contains a 'without_mask' face.
    """
    # Resize the image to the expected input size of the model (299, 299)
    resized_img = cv2.resize(image, (299, 299))
    
    # Convert the image from BGR to RGB and normalize
    processed_img = inceptionv3_preprocess_input(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))

    # Add an extra dimension to match the model's input shape and perform prediction
    prediction = model.predict(np.expand_dims(processed_img, axis=0))
    
    return prediction[0][0]

def annotate_image_with_attributes(image, model, detector_backend='retinaface'):
    """
    Receives an image, detects faces, and annotates it with attributes or mask presence.

    Parameters:
    - image (numpy array): The image to annotate.
    - model (tensorflow.keras.models.Model): The pre-trained mask detection model.
    - backends (list): List of face detection backends supported by DeepFace.

    Returns:
    - image_rgb (numpy array): The annotated image in RGB format.
    """
    # Face analysis using DeepFace
    analysis = DeepFace.analyze(img_path=image, actions=['emotion', 'age', 'gender', 'race'], detector_backend=detector_backend)

    # Iterate over each detected face
    for face in analysis:
        x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Crop the face from the image
        face_img = image[y:y+h, x:x+w]
        
        # Predict mask presence using the provided model
        mask_presence = analyze_mask_in_image(face_img, model)

        # Annotate based on mask presence
        if mask_presence < 0.5:  # Threshold might need adjustment
            text_line1 = "Masked"
        else:
            # Annotate with age, gender, race, and emotion if no mask is detected
            text_line1 = f"{face['dominant_race'].capitalize()}, {face['dominant_gender']}, Age: {face['age']}"
            text_line2 = f"Emotion: {face['dominant_emotion']}"
            cv2.putText(image, text_line2, (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Add the first line of text
        cv2.putText(image, text_line1, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return image

# List of available face detection backends in DeepFace.
# Each backend offers different strengths depending on the use-case scenario.
backends = [
  'opencv',     # OpenCV's Haar Cascade, good for basic face detection, fast but less accurate.
  'ssd',        # Single Shot MultiBox Detector, a balance of speed and accuracy.
  'dlib',       # Dlib’s HOG based model, accurate but can be slow without a GPU.
  'mtcnn',      # Multi-task Cascaded Convolutional Networks, very good at detecting small faces.
  'retinaface', # RetinaFace, state-of-the-art in terms of accuracy, requires more computational power.
  'mediapipe',  # MediaPipe, offers robust face detection, even in tough conditions.
  'yolov8',     # YOLOv8, an advancement in speed and accuracy, great for real-time processing.
  'yunet',      # YuNet, an ONNX-based fast and accurate detector.
  'fastmtcnn'  # Fast MTCNN, optimized version of MTCNN for real-time detection on CPUs.
]

from IPython.display import display, HTML

# Youtube
YouTubeVideo_ID = 'GEcT8ajh9pk'

# Adjust the width and height values
width = 1280  
height = 720  

# create a HTML string to center the video
html_str = """
<div style="display: flex; justify-content: center;">
    <iframe width="{}" height="{}" src="https://www.youtube.com/embed/{}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
""".format(width, height, YouTubeVideo_ID)

# Display HTML
display(HTML(html_str))



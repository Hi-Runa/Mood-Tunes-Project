import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('Weights.keras')

# Define the class labels (same order as used during training)
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Initialize webcam
camera = cv2.VideoCapture(0)

# Ensure the camera opened successfully
if not camera.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to quit the program.")

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame to grayscale (as the model was trained on grayscale images)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the frame to match the input size of the model
    resized_frame = cv2.resize(gray_frame, (48, 48))
    processed_frame = img_to_array(resized_frame) / 255.0
    processed_frame = np.expand_dims(processed_frame, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(processed_frame)
    emotion_label = class_labels[np.argmax(prediction)]

    # Display the resulting frame with the detected emotion
    cv2.putText(frame, f"Emotion: {emotion_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()

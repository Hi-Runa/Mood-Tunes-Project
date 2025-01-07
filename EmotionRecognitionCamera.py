import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('emotion_detection_model.h5')

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize the frame to 48x48
    resized = cv2.resize(gray, (48, 48))
    # Normalize the image
    normalized = resized / 255.0
    # Reshape the image to match the input shape of the model
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    
    # Predict the emotion
    result = model.predict(reshaped)
    emotion = emotion_labels[np.argmax(result)]
    
    # Display the emotion on the frame
    cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

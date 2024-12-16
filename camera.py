import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('Weights.keras')

# Emotion labels (make sure these match the order of your training dataset)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Initialize the camera
cap = cv2.VideoCapture(0)

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Read frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the region of interest (face)
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))
        face_normalized = face_resized / 255.0
        face_input = np.expand_dims(face_normalized, axis=-1)  # Add channel dimension
        face_input = np.expand_dims(face_input, axis=0)  # Add batch dimension

        # Predict emotion
        emotion_pred = model.predict(face_input)
        max_index = np.argmax(emotion_pred[0])
        emotion = emotion_labels[max_index]
        print(emotion_pred)
        # Draw rectangle around face and label with emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with the predicted emotion
        cv2.imshow('Emotion Detection', frame)
    

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()

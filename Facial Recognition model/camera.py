import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("emotion_model.keras")
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
IMG_SIZE = (48, 48)

# Start the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale and resize
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    processed_img = np.expand_dims(resized, axis=[0, -1]) / 255.0  # Normalize and reshape
    
    # Predict emotion
    prediction = model.predict(processed_img)
    emotion = EMOTIONS[np.argmax(prediction)]
    
    # Display emotion on frame
    cv2.putText(frame, emotion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Log emotion
    with open("emotion_log.txt", "a") as log_file:
        log_file.write(f"Detected emotion: {emotion}\n")
    
    # Show frame
    cv2.imshow("Emotion Detection", frame)
    
    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

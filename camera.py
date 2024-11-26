from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the model
model = load_model('ERM.keras')

# Define the expected input size for your model
image_width, image_height = 224, 224  # Replace with your model's input size

# Open a connection to the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame (resize, normalize, etc.)
    input_image = cv2.resize(frame, (image_width, image_height))
    input_image = input_image / 255.0  # Normalize if required
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(input_image)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Display the resulting frame with prediction
    cv2.putText(frame, f'Predicted: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Camera', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
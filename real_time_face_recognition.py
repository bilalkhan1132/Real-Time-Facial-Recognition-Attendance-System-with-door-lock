import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model("facial_recognition_model.h5")

# Define the ID-to-label mapping dictionary
id_to_label = {
    1: "Bilal Ahmed",
    2: "Muhammad Faisal",
    3: "Umair Ali",
    # Add more ID-to-label mappings as needed
}

# Initialize the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over each detected face
    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = frame[y:y + h, x:x + w]

        # Preprocess the face image
        face_roi = cv2.resize(face_roi, (224, 224))
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_roi = face_roi.astype("float") / 255.0  # Normalize the pixel values
        face_roi = np.expand_dims(face_roi, axis=0)

        # Make predictions on the face ROI
        pred = model.predict(face_roi)
        label_index = np.argmax(pred)
        confidence = pred[0][label_index]

        # Determine the label based on the ID-to-label mapping
        label = id_to_label.get(label_index + 1, "Unknown")

        # Draw the bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-Time Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()

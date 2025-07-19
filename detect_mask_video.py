import cv2
import numpy as np
from keras.models import load_model

# Load the trained mask detector model
model = load_model("mask_detector.model")

# Load the DNN face detector
net = cv2.dnn.readNetFromCaffe(
    "face_detector/deploy.prototxt",
    "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    # Prepare image for the network
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Loop over all detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter weak detections
        if confidence > 0.5:
            # Get coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the box is within the frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            # Extract the face ROI
            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            # Preprocess the face
            face_resized = cv2.resize(face, (128, 128))
            face_normalized = face_resized / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)

            # Predict mask
            result = model.predict(face_input)
            label = "Mask" if result[0][0] < 0.4 else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Draw the bounding box and label
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Show the frame
    cv2.imshow("Mask Detector", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

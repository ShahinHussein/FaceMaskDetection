# Import required libraries
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
from imutils.video import VideoStream
import numpy as np
import time
import cv2

# Import custom library for BrainPad interaction
from DUELink.DUELinkController import DUELinkController

# Obtain available port for connection and initialize BrainPad controller
available_port = DUELinkController.GetConnectionPort()
brain_pad_controller = DUELinkController(available_port)

# Load pre-trained face detector model
print("[INFO] Loading face detector model...")
prototxt_path = "face_detector/deploy.prototxt"
weights_path = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNet(prototxt_path, weights_path)

# Load pre-trained face mask detector model
print("[INFO] Loading face mask detector model...")
mask_net = load_model("mask_detector.model")

# Set minimum confidence threshold to filter weak detections
confidence_threshold = 0.5

# Initialize video stream and allow camera sensor to warm up
print("[INFO] Starting video stream...")
video_stream = VideoStream(src=0).start()
time.sleep(2.0)

# Clear BrainPad display and show initial state
brain_pad_controller.Display.Clear(0)
brain_pad_controller.Display.Show()


def detect_and_predict_mask(frame, face_net, mask_net):
    # Obtain frame dimensions and construct blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass blob through face detection network and obtain detections
    face_net.setInput(blob)
    detections = face_net.forward()

    # Initialize lists for faces, their locations, and mask predictions
    faces = []
    locs = []
    preds = []

    # Loop over detections
    for i in range(0, detections.shape[2]):
        # Extract confidence associated with detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > confidence_threshold:
            # Compute bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            # Ensure bounding boxes fall within frame dimensions
            (start_x, start_y) = (max(0, start_x), max(0, start_y))
            (end_x, end_y) = (min(w - 1, end_x), min(h - 1, end_y))

            # Extract face region, preprocess it, and add to lists
            face = frame[start_y:end_y, start_x:end_x]
            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((start_x, start_y, end_x, end_y))

    # Make predictions if at least one face was detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = mask_net.predict(faces, batch_size=32)

    return locs, preds


# Main loop for processing video stream frames
while True:
    # Capture frame from video stream and resize
    frame = video_stream.read()
    frame = imutils.resize(frame, width=800)

    # Detect faces and predict if they are wearing masks
    (locs, preds) = detect_and_predict_mask(frame, face_net, mask_net)

    # Loop over detected faces and their predictions
    for (box, pred) in zip(locs, preds):
        # Unpack bounding box and predictions
        (start_x, start_y, end_x, end_y) = box
        (mask, without_mask) = pred

        # Determine class label and color for bounding box
        label = "Mask" if mask > without_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Toggle BrainPad Pulse LED and display message for "No Mask" detection
        if label == "No Mask":
            brain_pad_controller.Frequency.Write('p', 1000, 100, 100)
            brain_pad_controller.Display.DrawTextScale("Wear a Mask", 1, 0, 10, 2, 2)
            brain_pad_controller.Display.Show()
            brain_pad_controller.Led.Set(200, 200, -1)

        brain_pad_controller.Display.Clear(0)
        brain_pad_controller.Display.Show()
        brain_pad_controller.Led.Set(0, 0, -1)

        # Include probability in label
        label = f"{label}: {max(mask, without_mask) * 100:.2f}%"

        # Display label and bounding box on frame
        cv2.putText(frame, label, (start_x, start_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

    # Show output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Break from loop if 'q' key pressed
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
video_stream.stop()

import cv2
import numpy as np
import pandas as pd

# Load YOLO configuration and weights
MODEL_CFG = "yolov4.cfg"  # Replace with "yolov3.cfg" if using YOLOv3
MODEL_WEIGHTS = "yolov4.weights"  # Replace with "yolov3.weights" if using YOLOv3
LABELS_FILE = "coco.names"

# Load class labels
with open(LABELS_FILE, "r") as f:
    LABELS = f.read().strip().split("\n")

# Initialize the YOLO network
net = cv2.dnn.readNetFromDarknet(MODEL_CFG, MODEL_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Confidence and threshold settings
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4  # Non-maximum suppression threshold


def count_people_in_frame(frame):
    (H, W) = frame.shape[:2]

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get the output layer name# Get the output layer names from the YOLO model
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Perform forward pass and get the detections
    detections = net.forward(output_layers)


    # Initialize variables for bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    for output in detections:
        for detection in output:
            # Extract scores, class ID, and confidence
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Only consider "person" class (ID: 0 in COCO dataset)
            if LABELS[class_id] == "person" and confidence > CONFIDENCE_THRESHOLD:
                # Scale the bounding box coordinates to the frame size
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to eliminate redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # Draw bounding boxes and count persons
    person_count = 0
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            person_count += 1

    return person_count, frame


def save_max_count_to_csv(max_count, csv_file="max_person_count.csv"):
    # Save the maximum count to a CSV file
    data = {"Max Count": [max_count]}
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"Max count ({max_count}) saved to {csv_file}")


def main():
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    max_count = 0  # Track the maximum number of persons detected

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Count persons in the frame
        num_people, output_frame = count_people_in_frame(frame)

        # Update the maximum count
        max_count = max(max_count, num_people)

        # Display the count on the frame
        cv2.putText(
            output_frame,
            f"Persons: {num_people}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Show the frame with bounding boxes
        cv2.imshow("YOLO Person Detection", output_frame)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save the maximum person count to CSV
    save_max_count_to_csv(max_count)

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

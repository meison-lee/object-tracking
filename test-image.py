import os
import cv2
import random
from ultralytics import YOLO
from tracker import Tracker

dict = {0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush'}


# Set the path to the folder containing images
image_folder = os.path.join('.', 'images')

# Output video path
video_out_path = os.path.join('.', 'out.mp4')

# Get a list of image file names in the folder
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))], key=lambda x: int(x.split('_')[1].split('.')[0]))

# Open the first image to get its shape for video writer initialization
first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
frame_height, frame_width, _ = first_image.shape

# Initialize video writer
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), 10, (frame_width, frame_height))

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Initialize tracker
tracker = Tracker()

# Generate random colors for visualization
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]

# Detection threshold
detection_threshold = 0.5

# Iterate over each image in the folder
frame_count = 0
for image_file in image_files:
    # Read the image
    frame = cv2.imread(os.path.join(image_folder, image_file))

    # Perform object detection using YOLO
    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if class_id == 0 and score > detection_threshold:  # Filter 'person' class detections
                detections.append([x1, y1, x2, y2, score])
            # if score > detection_threshold:
            #     detections.append([x1, y1, x2, y2, score])

        # Update tracker with detections
        tracker.update(frame, detections)

        # Visualize tracks
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            # Draw bounding box and track ID
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[track_id % len(colors)], 3)
            cv2.putText(frame, f"ID: {track_id} ", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2)

    # Write frame to output video
    frame_path = f"output_folder/frame_{frame_count}.jpg"
    cv2.imwrite(frame_path, frame)  # Save the frame as an image
    frame_count += 1
    cap_out.write(frame)

# Release video writer and close all windows
cap_out.release()
cv2.destroyAllWindows()

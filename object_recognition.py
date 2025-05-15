import cv2
import json
import numpy as np
from ultralytics import YOLO
from pynput import keyboard
import pyttsx3
import torch
from torchvision import models, transforms

# Global variables
latest_description = "No objects detected."
frame_count = 0
last_scene_description = ""  # Store the last confident scene description

# Load configuration from JSON file
def load_config():
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found. Using default settings.")
        return {"trigger_key": "space", "output_mode": "speech"}

# Generate object description based on detected objects
def generate_description(detections, frame_width):
    if not detections:
        return "No objects detected."

    # Filter detections with confidence > 0.5
    detections = [det for det in detections if det['confidence'] > 0.5]
    if not detections:
        return "No objects detected."

    # Define regions
    regions = {"left": [], "center": [], "right": []}
    for det in detections:
        bbox = det['bbox']  # [x1, y1, x2, y2]
        center_x = (bbox[0] + bbox[2]) / 2
        if center_x < frame_width / 3:
            region = "left"
        elif center_x < 2 * frame_width / 3:
            region = "center"
        else:
            region = "right"
        regions[region].append(det['class'])

    # Construct description
    description = ""
    for region, objects in regions.items():
        if objects:
            object_counts = {}
            for obj in objects:
                object_counts[obj] = object_counts.get(obj, 0) + 1
            items = []
            for obj, count in object_counts.items():
                if count == 1:
                    items.append(f"a {obj}")
                else:
                    items.append(f"{count} {obj}s")
            if len(items) == 1 and list(object_counts.values())[0] == 1:
                region_desc = f"On the {region}, there is {items[0]}."
            else:
                region_desc = f"On the {region}, there are " + ", ".join(items[:-1]) + (" and " + items[-1] if len(items) > 1 else items[0]) + "."
            description += region_desc + " "

    return description.strip() if description else "No objects detected."

# Output the description as speech or text
def output_description(description, mode):
    if mode == "speech":
        engine = pyttsx3.init()
        engine.say(description)
        engine.runAndWait()
    else:  # mode == "text"
        print(description)

# Handle key press event
def on_press(key):
    global latest_description
    config = load_config()
    trigger_key = config["trigger_key"]
    try:
        if key == getattr(keyboard.Key, trigger_key):
            output_description(latest_description, config["output_mode"])
    except AttributeError:
        pass

# Main application function
def main():
    global latest_description, frame_count, last_scene_description

    # Load configuration
    config = load_config()

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    # Load YOLOv5 model
    try:
        model = YOLO("yolov5su.pt")  # Small model for speed
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        cap.release()
        return

    # Load scene recognition model and categories
    device = torch.device('cpu')
    scene_model = models.resnet50()
    scene_model.fc = torch.nn.Linear(scene_model.fc.in_features, 365)
    try:
        checkpoint = torch.load("resnet50_places365.pth.tar", map_location=device)
        scene_model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
    except Exception as e:
        print(f"Error loading scene recognition model: {e}")
        cap.release()
        return
    scene_model.eval()

    # Load scene categories
    try:
        with open("categories_places365.txt", "r") as f:
            scene_categories = [line.strip().split()[0][3:] for line in f]  # e.g., 'airport_terminal'
    except FileNotFoundError:
        print("Error: categories_places365.txt not found.")
        cap.release()
        return

    # Preprocess for scene recognition
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            frame_count += 1

            # Get frame width
            frame_width = frame.shape[1]

            # Detect objects
            results = model(frame)
            detections = []
            for result in results:
                for box in result.boxes:
                    detections.append({
                        'class': model.names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': [int(box.xyxy[0][0]), int(box.xyxy[0][1]),
                                 int(box.xyxy[0][2]), int(box.xyxy[0][3])]
                    })

            # Scene recognition on every frame (for debugging)
            scene_description = ""
            input_tensor = preprocess(frame)
            input_batch = input_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                output = scene_model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            max_prob, scene_index = torch.max(probabilities, dim=0)
            print(f"Scene recognition - Top prediction: {scene_categories[scene_index.item()]} with confidence {max_prob.item():.2f}")  # Debug output
            if max_prob > 0.3:  # Lowered threshold for testing
                scene_description = scene_categories[scene_index.item()].replace('_', ' ')
                last_scene_description = scene_description  # Store the last confident scene
            else:
                scene_description = last_scene_description  # Use the last confident scene if current confidence is too low

            # Generate object description
            object_description = generate_description(detections, frame_width)

            # Combine descriptions
            if scene_description:
                full_description = f"You are in a {scene_description}. {object_description}"
            else:
                full_description = object_description

            latest_description = full_description

            # Display video feed with bounding boxes (optional, for visualization)
            for det in detections:
                if det['confidence'] > 0.5:
                    bbox = det['bbox']
                    label = f"{det['class']} ({det['confidence']:.2f})"
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('Object Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Application terminated by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        listener.stop()

if __name__ == "__main__":
    main()
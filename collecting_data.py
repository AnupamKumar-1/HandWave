#!/usr/bin/env python
# coding: utf-8

import os
import cv2

# Configuration
DATA_DIR = "./data"
dataset_size = 100  # Images per class

# Prompt: which class to collect
classes = input(
    "Enter label(s) to collect (comma-separated, e.g., A,B,C,space,del): "
).split(",")

# Create base directory if it doesn’t exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

for label in classes:
    label = label.strip()
    label_dir = os.path.join(DATA_DIR, label)

    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    print(f"\nReady to collect data for label '{label.upper()}'")
    print("👉 Press 'Q' to start recording.")

    # Wait until 'q' to start collecting
    while True:
        ret, frame = cap.read()
        cv2.putText(
            frame,
            f'Label: {label.upper()} - Press "Q" to start',
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    counter = 0
    print(f"📸 Collecting {dataset_size} images for label '{label}'...")

    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame.")
            continue

        # Display image count
        cv2.putText(
            frame,
            f"Collecting {label.upper()} - Image {counter + 1}/{dataset_size}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 100, 0),
            2,
        )
        cv2.imshow("Frame", frame)

        # Save image
        file_path = os.path.join(label_dir, f"{counter}.jpg")
        cv2.imwrite(file_path, frame)
        counter += 1

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit early
            print("❌ Exiting early.")
            break

    print(f"✅ Finished collecting for '{label}'.")

cap.release()
cv2.destroyAllWindows()
print("📁 Dataset collection complete.")

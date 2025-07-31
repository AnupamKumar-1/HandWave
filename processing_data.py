#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import cv2
import mediapipe as mp

# Setup MediaPipe Hands for static-image processing
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3
)

# Paths
DATA_DIR         = './data'
OUTPUT_DATA      = 'data.pickle'
OUTPUT_LABEL_MAP = 'label_map.pickle'

# Containers for features and labels
data   = []
labels = []
label_map      = {}  # e.g. {'A': 0, 'B': 1, …}
next_label_idx = 0

print("📦 Starting image processing…")

# Walk through each class folder, sorted alphabetically
for class_name in sorted(os.listdir(DATA_DIR)):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    # Assign numeric index to this class if new
    if class_name not in label_map:
        label_map[class_name] = next_label_idx
        next_label_idx += 1

    class_idx = label_map[class_name]
    print(f"📂 Processing class '{class_name}' → idx {class_idx}")

    # Process each image in the class folder
    for img_file in sorted(os.listdir(class_path)):
        img_path = os.path.join(class_path, img_file)
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Skipped unreadable: {img_path}")
                continue

            # Convert BGR→RGB and run MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            # If a hand is detected, extract and normalize landmarks
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                xs = [p.x for p in lm.landmark]
                ys = [p.y for p in lm.landmark]
                min_x, min_y = min(xs), min(ys)

                feat = []
                for p in lm.landmark:
                    feat.append(p.x - min_x)
                    feat.append(p.y - min_y)

                data.append(feat)
                labels.append(class_idx)

        except Exception as e:
            print(f"❌ Error on {img_path}: {e}")

print(f"\n✅ Collected {len(data)} samples.")
print("✅ Class → index map:", label_map)

# Save features+labels
with open(OUTPUT_DATA, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
print(f"📁 Wrote dataset to '{OUTPUT_DATA}'")

# Save label map
with open(OUTPUT_LABEL_MAP, 'wb') as f:
    pickle.dump(label_map, f)
print(f"📁 Wrote label map to '{OUTPUT_LABEL_MAP}'")

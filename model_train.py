#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# Paths
DATASET_PATH = "data.pickle"
LABEL_MAP_PATH = "label_map.pickle"
MODEL_PATH = "model.p"

# Load processed feature data
print("Loading dataset...")
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset file '{DATASET_PATH}' not found.")
with open(DATASET_PATH, "rb") as f:
    data_dict = pickle.load(f)
X = np.array(data_dict["data"])
y = np.array(data_dict["labels"])

# Load and invert label map (class_name → index → class_name)
print("Loading label map...")
if not os.path.exists(LABEL_MAP_PATH):
    raise FileNotFoundError(f"Label map file '{LABEL_MAP_PATH}' not found.")
with open(LABEL_MAP_PATH, "rb") as f:
    raw_label_map = pickle.load(f)  # e.g. {'A': 0, 'B': 1, ...}
label_map = {v: k for k, v in raw_label_map.items()}  # {0: 'A', 1: 'B', ...}
print(f"Loaded {len(label_map)} class labels.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
print("Training RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print("Evaluating model...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# Build the list of labels actually seen
unique_labels = np.unique(np.concatenate([y_test, y_pred])).tolist()
# Convert to human-readable names
target_names = [str(label_map[i]) for i in unique_labels]

# Classification report
print("Classification Report:")
print(
    classification_report(
        y_test, y_pred, labels=unique_labels, target_names=target_names
    )
)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=unique_labels)
plt.figure(figsize=(12, 10))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# Save model + label_map
print("Saving model and label map into 'model.p'...")
with open(MODEL_PATH, "wb") as f:
    pickle.dump({"model": model, "label_map": label_map}, f)
print("Done.")

import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import os


def load_model(model_path="model.p", fallback_label_map="label_map.pickle"):
    """Load model and label map from disk, falling back to a separate pickle if needed."""
    with open(model_path, "rb") as f:
        md = pickle.load(f)
    model = md.get("model")
    label_map = md.get("label_map", {}) or {}

    if not label_map and os.path.exists(fallback_label_map):
        with open(fallback_label_map, "rb") as lm_file:
            raw_map = pickle.load(lm_file)
        label_map = {v: k for k, v in raw_map.items()}

    if not label_map:
        print("No label map found; predictions will show raw indices.")
    return model, label_map


def preprocess_landmarks(landmarks):
    """Convert MediaPipe hand landmarks to a normalised feature vector."""
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    min_x, min_y = min(xs), min(ys)
    data = []
    for x, y in zip(xs, ys):
        data.extend([x - min_x, y - min_y])
    return np.array(data).reshape(1, -1)


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def main():
    """Run real-time ASL gesture recognition from webcam."""
    model, label_map = load_model()
    cap = cv2.VideoCapture(0)
    prev_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

                feat = preprocess_landmarks(hand_landmarks)
                pred_idx = model.predict(feat)[0]

                label = label_map.get(pred_idx, str(pred_idx))

                h, w, _ = frame.shape
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                x1, y1 = int(min(xs) * w) - 10, int(min(ys) * h) - 10
                x2, y2 = int(max(xs) * w) + 10, int(max(ys) * h) + 10
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    str(label),
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    2,
                )

        now = time.time()
        fps = 1.0 / (now - prev_time) if prev_time else 0.0
        prev_time = now
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (100, 255, 0),
            2,
        )

        cv2.imshow("ASL Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

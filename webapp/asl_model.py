import sys
import logging
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union

# Add parent directory to Python path to import processing_data
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from processing_data import preprocess  # Must come *after* sys.path adjustment

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ASLModel:
    def __init__(self, model, label_map):
        self.model = model
        self.idx_to_class = {idx: cls for cls, idx in label_map.items()}
        self.class_to_idx = label_map

    def predict(self, pil_image: Image.Image) -> str:
        """
        Accepts a PIL image, runs preprocessing (with MediaPipe),
        and returns predicted class label.
        """
        try:
            features = preprocess(pil_image)
            if not features or sum(features) == 0.0:
                logger.warning("No hand landmarks detected in image.")
                return "No hand detected"

            X = np.array(features, dtype=np.float32).reshape(1, -1)
            pred_idx = self.model.predict(X)[0]
            return self.idx_to_class.get(pred_idx, "Unknown")

        except Exception as ex:
            logger.error("Prediction failed: %s", ex, exc_info=True)
            return "Error"

    def predict_from_landmarks(self, features: list[float]) -> str:
        """
        Accepts a list of 42 floats (MediaPipe landmarks)
        and returns predicted class label.
        """
        try:
            if not features or sum(features) == 0.0:
                logger.warning("Empty or invalid landmark vector.")
                return "No hand detected"

            X = np.array(features, dtype=np.float32).reshape(1, -1)
            pred_idx = self.model.predict(X)[0]
            return self.idx_to_class.get(pred_idx, "Unknown")
        except Exception as ex:
            logger.error("Landmark prediction failed: %s", ex, exc_info=True)
            return "Error"

    def decode_prediction(self, pred_idx: int) -> str:
        """
        Converts prediction index to class label.
        """
        return self.idx_to_class.get(pred_idx, "Unknown")


def load_model(
    model_file: Path = project_root / "model.p",
    label_map_file: Path = project_root / "label_map.pickle",
) -> ASLModel:
    """
    Loads model and label map, returns ASLModel instance.
    Automatically handles if model.p is saved as a dict with 'model' key.
    """
    if not model_file.is_file():
        raise FileNotFoundError(f"ASL model not found at: {model_file}")
    if not label_map_file.is_file():
        raise FileNotFoundError(f"Label map not found at: {label_map_file}")

    with model_file.open("rb") as mf:
        model_obj = pickle.load(mf)
        if isinstance(model_obj, dict) and "model" in model_obj:
            model = model_obj["model"]
            logger.info("✅ Loaded ASL model (dict-wrapped) from %s", model_file)
        else:
            model = model_obj
            logger.info("✅ Loaded ASL model from %s", model_file)

    with label_map_file.open("rb") as lf:
        label_map = pickle.load(lf)
        logger.info("✅ Loaded label map from %s", label_map_file)

    return ASLModel(model, label_map)


def load_label_map(
    label_map_file: Union[str, Path] = project_root / "label_map.pickle",
) -> dict:
    """
    Loads and returns label map from pickle file.
    Accepts either a Path or a string.
    """
    label_map_file = Path(label_map_file)
    if not label_map_file.is_file():
        raise FileNotFoundError(f"Label map not found at: {label_map_file}")
    with label_map_file.open("rb") as f:
        return pickle.load(f)

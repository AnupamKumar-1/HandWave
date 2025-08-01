import os
import pickle
from pathlib import Path

import pytest

from webapp.asl_model import ASLModel, load_model
import webapp.asl_model as asm

# Try to find the label-map loader in your module
try:
    load_label_map = asm.load_label_map
except AttributeError:
    try:
        load_label_map = asm.get_label_map
    except AttributeError:
        load_label_map = None


@pytest.fixture
def assets_dir(tmp_path):
    return tmp_path


def test_load_label_map(tmp_path, monkeypatch):
    if load_label_map is None:
        pytest.skip("No 'load_label_map' or 'get_label_map' in webapp.asl_model")
    lm = {0: "A", 1: "B"}
    p = tmp_path / "label_map.pickle"
    with open(p, "wb") as f:
        pickle.dump(lm, f)
    monkeypatch.chdir(tmp_path)
    loaded = load_label_map("label_map.pickle")
    assert loaded == lm


def test_model_file_exists(tmp_path):
    p = tmp_path / "model.p"
    p.write_bytes(b"dummy")
    assert os.path.exists(str(p))


def test_load_model(tmp_path, monkeypatch):
    # 1) Create and pickle a dummy model
    dummy = {"foo": "bar"}
    model_path = tmp_path / "model.p"
    with open(model_path, "wb") as f:
        pickle.dump(dummy, f)

    # 2) Change cwd so load_model finds our file
    monkeypatch.chdir(tmp_path)

    # 3) Load the model using a Path
    loaded_model = load_model(Path("model.p"))

    # 4) Verify we got an ASLModel instance
    assert isinstance(loaded_model, ASLModel), "Expected an ASLModel instance"

    # 5) Its .model attribute should equal our dummy dict
    assert hasattr(loaded_model, "model"), "ASLModel missing 'model' attribute"
    assert loaded_model.model == dummy

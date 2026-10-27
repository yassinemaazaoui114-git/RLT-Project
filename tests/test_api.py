"""End-to-end tests of the FastAPI endpoints (requires `python train_and_save.py` first)."""
import pytest
from fastapi.testclient import TestClient
from api.app import app
from api.routes.prediction import MODEL_NAMES
from utils.data_loader import load_data

client = TestClient(app)


def _patient(diagnosis):
    """Raw (unscaled) measurements of the first patient with the given diagnosis."""
    df = load_data()
    row = df[df["diagnosis"] == diagnosis].iloc[0]
    return row.drop(["id", "diagnosis"]).astype(float).to_dict()


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    assert r.json()["models"] == MODEL_NAMES


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("diagnosis, expected", [("M", "Malignant"), ("B", "Benign")])
def test_predict_every_model(model_name, diagnosis, expected):
    r = client.post(f"/api/predict?model_name={model_name}", json=_patient(diagnosis))
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"] == model_name
    assert body["diagnosis"] == expected
    assert abs(sum(body["probabilities"].values()) - 100) < 1e-6


def test_predict_unknown_model():
    r = client.post("/api/predict?model_name=SVM", json=_patient("M"))
    assert r.status_code == 400


def test_compare():
    r = client.post("/api/compare", json=_patient("M"))
    assert r.status_code == 200
    assert set(r.json()["comparison"]) == set(MODEL_NAMES)
    assert r.json()["consensus"] == "Malignant"


def test_export_pdf():
    r = client.post("/api/export-pdf?patient_id=TEST-001", json=_patient("M"))
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:4] == b"%PDF"


def test_home_page():
    r = client.get("/")
    assert r.status_code == 200
    assert "RLT" in r.text

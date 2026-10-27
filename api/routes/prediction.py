"""
Prediction routes - handles all ML inference.
"""
from datetime import datetime
from fastapi.responses import Response
from api.utils.pdf_generator import PDFReportGenerator
from fastapi import APIRouter, HTTPException
from api.schemas import PredictionInput, PredictionOutput
from config import MODELS_DIR
import numpy as np
import pandas as pd
import joblib

router = APIRouter()

MODEL_NAMES = ["DecisionTree", "RandomForest", "ExtraTrees", "GradientBoosting", "XGBoost", "RLT", "NeuralRLT"]

# Load models and scaler (ONCE when server starts)
MODELS = {}
try:
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    feature_info = joblib.load(MODELS_DIR / "feature_info.joblib")
    for name in MODEL_NAMES:
        try:
            MODELS[name] = joblib.load(MODELS_DIR / f"{name.lower()}.joblib")
        except Exception as e:  # one broken model (e.g. TensorFlow missing) must not take down the others
            print(f"⚠️  Could not load {name}: {e}")
    print(f"✅ {len(MODELS)}/{len(MODEL_NAMES)} models loaded: {list(MODELS)}")

except Exception as e:
    print(f"❌ Error loading scaler/feature info: {e}. Run `python train_and_save.py` first.")


def _scaled_features(data: PredictionInput) -> np.ndarray:
    """Input features in the training column order, standardised with the training scaler."""
    features = pd.DataFrame([data.model_dump()], columns=feature_info["feature_names"])
    return scaler.transform(features)


def _compare(features_scaled: np.ndarray) -> dict:
    """Predictions of every model plus the majority-vote consensus."""
    results = {}
    for name, model in MODELS.items():
        prediction = int(model.predict(features_scaled)[0])
        probabilities = model.predict_proba(features_scaled)[0]

        results[name] = {
            "diagnosis": "Malignant" if prediction == 1 else "Benign",
            "confidence": float(max(probabilities)) * 100,
            "malignant_probability": float(probabilities[1]) * 100
        }

    n_malignant = sum(r["diagnosis"] == "Malignant" for r in results.values())
    return {
        "comparison": results,
        "consensus": "Malignant" if n_malignant > len(results) / 2 else "Benign"
    }


@router.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput, model_name: str = "RandomForest"):
    """
    Predict breast cancer diagnosis from 30 tumor features.
    """
    if model_name not in MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{model_name}' not found. Available: {list(MODELS.keys())}"
        )

    features_scaled = _scaled_features(data)

    # Predict
    model = MODELS[model_name]
    prediction = int(model.predict(features_scaled)[0])
    probabilities = model.predict_proba(features_scaled)[0]
    
    diagnosis = "Malignant" if prediction == 1 else "Benign"
    confidence = float(max(probabilities)) * 100
    
    # Risk level
    malignant_prob = float(probabilities[1]) * 100
    if malignant_prob < 30:
        risk_level = "Low"
    elif malignant_prob < 70:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    return PredictionOutput(
        model_name=model_name,
        diagnosis=diagnosis,
        confidence=confidence,
        probabilities={
            "Benign": float(probabilities[0]) * 100,
            "Malignant": float(probabilities[1]) * 100
        },
        risk_level=risk_level
    )

@router.get("/models")
async def list_models():
    """List all available models."""
    return {
        "available_models": list(MODELS.keys()),
        "total": len(MODELS),
        "default": "RandomForest"
    }

@router.post("/compare")
async def compare_all_models(data: PredictionInput):
    """Compare predictions from all models."""
    return _compare(_scaled_features(data))

@router.get("/health")
async def health_check():
    """Check if models are loaded."""
    return {
        "status": "healthy" if MODELS else "error",
        "models_loaded": len(MODELS),
        "models": list(MODELS.keys())
    }

@router.post("/export-pdf")
async def generate_pdf_report(data: PredictionInput, patient_id: str = "Unknown", date: str = None):
    """
    Generate a PDF report with all model predictions.
    """
    comparison_results = _compare(_scaled_features(data))

    # Prepare patient data
    patient_data = {
        "patient_id": patient_id,
        "date": date or datetime.now().strftime('%Y-%m-%d'),
        "measurements": data.model_dump()
    }

    # Generate PDF
    pdf_generator = PDFReportGenerator()
    pdf_bytes = pdf_generator.generate_report(patient_data, comparison_results)

    # Return PDF as download
    filename = f"breast_cancer_report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )

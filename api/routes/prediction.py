"""
Prediction routes - handles all ML inference.
"""
from datetime import datetime
from fastapi.responses import Response
from api.utils.pdf_generator import PDFReportGenerator
from fastapi import APIRouter, HTTPException
from api.schemas import PredictionInput, PredictionOutput
import numpy as np
import joblib
from pathlib import Path

router = APIRouter()

# Load models and scaler (ONCE when server starts)
MODELS_DIR = Path("models")

try:
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    feature_info = joblib.load(MODELS_DIR / "feature_info.joblib")
    
    MODELS = {
        "DecisionTree": joblib.load(MODELS_DIR / "decisiontree.joblib"),
        "RandomForest": joblib.load(MODELS_DIR / "randomforest.joblib"),
        "ExtraTrees": joblib.load(MODELS_DIR / "extratrees.joblib"),
        "GradientBoosting": joblib.load(MODELS_DIR / "gradientboosting.joblib"),
        "XGBoost": joblib.load(MODELS_DIR / "xgboost.joblib"),
        "RLT": joblib.load(MODELS_DIR / "rlt.joblib"),  # 🆕 NEW!
    }
    print("✅ All 6 models loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    MODELS = {}

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
    
    # Convert input to numpy array
    features = np.array([[
        data.radius_mean, data.texture_mean, data.perimeter_mean, data.area_mean, 
        data.smoothness_mean, data.compactness_mean, data.concavity_mean, 
        data.concave_points_mean, data.symmetry_mean, data.fractal_dimension_mean,
        data.radius_se, data.texture_se, data.perimeter_se, data.area_se, 
        data.smoothness_se, data.compactness_se, data.concavity_se, 
        data.concave_points_se, data.symmetry_se, data.fractal_dimension_se,
        data.radius_worst, data.texture_worst, data.perimeter_worst, data.area_worst, 
        data.smoothness_worst, data.compactness_worst, data.concavity_worst, 
        data.concave_points_worst, data.symmetry_worst, data.fractal_dimension_worst,
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
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
    """Compare predictions from all 6 models."""
    features = np.array([[
        data.radius_mean, data.texture_mean, data.perimeter_mean, data.area_mean, 
        data.smoothness_mean, data.compactness_mean, data.concavity_mean, 
        data.concave_points_mean, data.symmetry_mean, data.fractal_dimension_mean,
        data.radius_se, data.texture_se, data.perimeter_se, data.area_se, 
        data.smoothness_se, data.compactness_se, data.concavity_se, 
        data.concave_points_se, data.symmetry_se, data.fractal_dimension_se,
        data.radius_worst, data.texture_worst, data.perimeter_worst, data.area_worst, 
        data.smoothness_worst, data.compactness_worst, data.concavity_worst, 
        data.concave_points_worst, data.symmetry_worst, data.fractal_dimension_worst,
    ]])
    
    features_scaled = scaler.transform(features)
    
    results = {}
    for name, model in MODELS.items():
        prediction = int(model.predict(features_scaled)[0])
        probabilities = model.predict_proba(features_scaled)[0]
        
        results[name] = {
            "diagnosis": "Malignant" if prediction == 1 else "Benign",
            "confidence": float(max(probabilities)) * 100,
            "malignant_probability": float(probabilities[1]) * 100
        }
    
    return {
        "comparison": results,
        "consensus": "Malignant" if sum(r["diagnosis"] == "Malignant" for r in results.values()) > 3 else "Benign"
    }

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
    # Collect measurements
    measurements = {
        'radius_mean': data.radius_mean,
        'texture_mean': data.texture_mean,
        'perimeter_mean': data.perimeter_mean,
        'area_mean': data.area_mean,
        'smoothness_mean': data.smoothness_mean,
        'compactness_mean': data.compactness_mean,
        'concavity_mean': data.concavity_mean,
        'concave_points_mean': data.concave_points_mean,
        'symmetry_mean': data.symmetry_mean,
        'fractal_dimension_mean': data.fractal_dimension_mean,
        'radius_se': data.radius_se,
        'texture_se': data.texture_se,
        'perimeter_se': data.perimeter_se,
        'area_se': data.area_se,
        'smoothness_se': data.smoothness_se,
        'compactness_se': data.compactness_se,
        'concavity_se': data.concavity_se,
        'concave_points_se': data.concave_points_se,
        'symmetry_se': data.symmetry_se,
        'fractal_dimension_se': data.fractal_dimension_se,
        'radius_worst': data.radius_worst,
        'texture_worst': data.texture_worst,
        'perimeter_worst': data.perimeter_worst,
        'area_worst': data.area_worst,
        'smoothness_worst': data.smoothness_worst,
        'compactness_worst': data.compactness_worst,
        'concavity_worst': data.concavity_worst,
        'concave_points_worst': data.concave_points_worst,
        'symmetry_worst': data.symmetry_worst,
        'fractal_dimension_worst': data.fractal_dimension_worst,
    }
    
    # Get comparison results
    features = np.array([[
        data.radius_mean, data.texture_mean, data.perimeter_mean, data.area_mean,
        data.smoothness_mean, data.compactness_mean, data.concavity_mean,
        data.concave_points_mean, data.symmetry_mean, data.fractal_dimension_mean,
        data.radius_se, data.texture_se, data.perimeter_se, data.area_se,
        data.smoothness_se, data.compactness_se, data.concavity_se,
        data.concave_points_se, data.symmetry_se, data.fractal_dimension_se,
        data.radius_worst, data.texture_worst, data.perimeter_worst, data.area_worst,
        data.smoothness_worst, data.compactness_worst, data.concavity_worst,
        data.concave_points_worst, data.symmetry_worst, data.fractal_dimension_worst,
    ]])
    
    features_scaled = scaler.transform(features)
    
    results = {}
    for name, model in MODELS.items():
        prediction = int(model.predict(features_scaled)[0])
        probabilities = model.predict_proba(features_scaled)[0]
        
        results[name] = {
            "diagnosis": "Malignant" if prediction == 1 else "Benign",
            "confidence": float(max(probabilities)) * 100,
            "malignant_probability": float(probabilities[1]) * 100
        }
    
    consensus = "Malignant" if sum(r["diagnosis"] == "Malignant" for r in results.values()) > 3 else "Benign"
    
    comparison_results = {
        "consensus": consensus,
        "comparison": results
    }
    
    # Prepare patient data
    patient_data = {
        "patient_id": patient_id,
        "date": date or datetime.now().strftime('%Y-%m-%d'),
        "measurements": measurements
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

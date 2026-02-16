"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Dict

class PredictionInput(BaseModel):
    """Single prediction input with 30 features."""
    radius_mean: float = Field(..., ge=0)
    texture_mean: float = Field(..., ge=0)
    perimeter_mean: float = Field(..., ge=0)
    area_mean: float = Field(..., ge=0)
    smoothness_mean: float = Field(..., ge=0, le=1)
    compactness_mean: float = Field(..., ge=0)
    concavity_mean: float = Field(..., ge=0)
    concave_points_mean: float = Field(..., ge=0)
    symmetry_mean: float = Field(..., ge=0)
    fractal_dimension_mean: float = Field(..., ge=0)
    
    radius_se: float = Field(..., ge=0)
    texture_se: float = Field(..., ge=0)
    perimeter_se: float = Field(..., ge=0)
    area_se: float = Field(..., ge=0)
    smoothness_se: float = Field(..., ge=0)
    compactness_se: float = Field(..., ge=0)
    concavity_se: float = Field(..., ge=0)
    concave_points_se: float = Field(..., ge=0)
    symmetry_se: float = Field(..., ge=0)
    fractal_dimension_se: float = Field(..., ge=0)
    
    radius_worst: float = Field(..., ge=0)
    texture_worst: float = Field(..., ge=0)
    perimeter_worst: float = Field(..., ge=0)
    area_worst: float = Field(..., ge=0)
    smoothness_worst: float = Field(..., ge=0, le=1)
    compactness_worst: float = Field(..., ge=0)
    concavity_worst: float = Field(..., ge=0)
    concave_points_worst: float = Field(..., ge=0)
    symmetry_worst: float = Field(..., ge=0)
    fractal_dimension_worst: float = Field(..., ge=0)

class PredictionOutput(BaseModel):
    """Prediction result."""
    model_name: str
    diagnosis: str  # "Benign" or "Malignant"
    confidence: float  # 0-100
    probabilities: Dict[str, float]
    risk_level: str  # "Low", "Medium", "High"

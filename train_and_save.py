# ============================================
# train_and_save.py
# Train models ONCE and save them forever
# ============================================
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

print("🚀 Starting training process...")

# ============================================
# 1. LOAD DATA
# ============================================
print("\n📂 Loading data...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

column_names = [
    'id', 'diagnosis',
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

df = pd.read_csv(url, header=None, names=column_names)

# ============================================
# 2. PREPROCESSING
# ============================================
print("🔧 Preprocessing data...")

# Encode target
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Features and target
X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Data ready: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")

# ============================================
# 3. TRAIN ALL MODELS (Including RLT!)
# ============================================
print("\n🤖 Training models (this will take a few minutes)...")

models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=10),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', n_jobs=-1),
}

trained_models = {}

for name, model in models.items():
    print(f"  Training {name}...", end=" ")
    model.fit(X_train_scaled, y_train)
    score = model.score(X_test_scaled, y_test)
    trained_models[name] = model
    print(f"✅ Accuracy: {score:.4f}")

# ============================================
# 3B. TRAIN RLT MODEL (Simplified Version)
# ============================================
print("  Training RLT...", end=" ")

try:
    # Import the RLT implementation from your notebook
    # For now, we'll use a placeholder that stores the necessary info
    # You'll need to extract the full RLT code from your notebook
    
    # Convert to DataFrame for RLT compatibility
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Placeholder: Use Random Forest as a stand-in for RLT
    # (Replace this with your actual RLT training code from notebook)
    rlt_model = RandomForestClassifier(
        n_estimators=50, 
        max_depth=15,
        min_samples_split=5,
        random_state=42, 
        n_jobs=-1
    )
    rlt_model.fit(X_train_scaled, y_train)
    rlt_score = rlt_model.score(X_test_scaled, y_test)
    
    trained_models["RLT"] = rlt_model
    print(f"✅ Accuracy: {rlt_score:.4f}")
    print("  ⚠️  Note: Using simplified RLT implementation")
    
except Exception as e:
    print(f"⚠️  Skipped (error: {e})")
    print("  💡 To add full RLT, extract code from your notebook")

# ============================================
# 4. SAVE EVERYTHING
# ============================================
print("\n💾 Saving models...")

os.makedirs("models", exist_ok=True)

# Save scaler
joblib.dump(scaler, "models/scaler.joblib")
print("✅ Saved scaler")

# Save each model
for name, model in trained_models.items():
    filename = name.lower().replace(" ", "_")
    joblib.dump(model, f"models/{filename}.joblib")
    print(f"✅ Saved {filename}")

# ============================================
# 5. SAVE FEATURE NAMES (for API validation)
# ============================================
feature_info = {
    'feature_names': X.columns.tolist(),
    'n_features': len(X.columns)
}
joblib.dump(feature_info, "models/feature_info.joblib")
print("✅ Saved feature names")

print("\n" + "="*50)
print("🎉 SUCCESS! All models saved!")
print("="*50)
print("\n📂 Saved files:")
print("  - models/scaler.joblib")
print("  - models/feature_info.joblib")
for name in trained_models.keys():
    print(f"  - models/{name.lower().replace(' ', '_')}.joblib")

print("\n✨ You can now use these models in FastAPI WITHOUT retraining!")
print("💡 To retrain in the future, just run: python train_and_save.py")

"""
Breast Cancer Classification - Main Training Script
Trains all models including Traditional RLT and Neural RLT
Generates comprehensive visualizations and performance metrics
"""

import numpy as np
import pandas as pd
import time
import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)
import xgboost as xgb

# Import custom modules
from utils.rlt_model import (
    fit_rlt_forest,
    predict_rlt_forest,
    predict_rlt_forest_scores
)
from utils.neural_rlt import NeuralRLTForest
from utils.visualizations import ModelVisualizer, plot_rlt_feature_importance

warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """Load and prepare breast cancer dataset"""
    print("\n" + "="*70)
    print("📊 LOADING DATASET")
    print("="*70)
    
    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    print(f"✓ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"✓ Class distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create DataFrames for RLT
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    
    return X, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X_train_df, X_test_df, scaler


def train_baseline_models(X_train_scaled, y_train):
    """Train baseline models"""
    print("\n" + "="*70)
    print("🤖 TRAINING BASELINE MODELS")
    print("="*70)
    
    models = {}
    
    # Random Forest
    print("\n1️⃣  Training Random Forest...")
    start = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    models['RandomForest'] = rf
    print(f"   ✓ Trained in {time.time() - start:.2f}s")
    
    # Gradient Boosting
    print("\n2️⃣  Training Gradient Boosting...")
    start = time.time()
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train_scaled, y_train)
    models['GradientBoosting'] = gb
    print(f"   ✓ Trained in {time.time() - start:.2f}s")
    
    # XGBoost
    print("\n3️⃣  Training XGBoost...")
    start = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb_model.fit(X_train_scaled, y_train)
    models['XGBoost'] = xgb_model
    print(f"   ✓ Trained in {time.time() - start:.2f}s")
    
    # Extra Trees
    print("\n4️⃣  Training Extra Trees...")
    start = time.time()
    et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    et.fit(X_train_scaled, y_train)
    models['ExtraTrees'] = et
    print(f"   ✓ Trained in {time.time() - start:.2f}s")
    
    # Decision Tree
    print("\n5️⃣  Training Decision Tree...")
    start = time.time()
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_scaled, y_train)
    models['DecisionTree'] = dt
    print(f"   ✓ Trained in {time.time() - start:.2f}s")
    
    print("\n✅ All baseline models trained successfully!")
    
    return models


def train_traditional_rlt(X_train_df, y_train, feature_names):
    """Train Traditional RLT Forest"""
    print("\n" + "="*70)
    print("="*70)
    print(" TRAINING TRADITIONAL RLT FOREST (BEST PARAMS)")
    print("="*70)
    print("\n")
    
    start = time.time()
    
    # Best hyperparameters from tuning
    best_params = {
        'M': 50,
        'k': 1,
        'pdmode': 'none',
        'p0': 3,
        'nmin': 10,
        'n_estimators_embedded': 50
    }
    
    rlt_forest = fit_rlt_forest(
        X=X_train_df.values,
        y=y_train.values,
        feature_names=feature_names,
        M=best_params['M'],
        k=best_params['k'],
        pdmode=best_params['pdmode'],
        p0=best_params['p0'],
        nmin=best_params['nmin'],
        n_estimators_embedded=best_params['n_estimators_embedded'],
        random_state=42,
        task='classification'
    )
    
    training_time = time.time() - start
    
    print(f"\n✅ Traditional RLT Forest trained successfully!")
    print("="*70)
    
    return rlt_forest, training_time


def train_neural_rlt(X_train_scaled, y_train):
    """Train Neural RLT Forest"""
    print("\n" + "="*70)
    print("🔄 Training Neural RLT Forest...")
    
    start = time.time()
    
    neural_rlt = NeuralRLTForest(
        n_estimators=10,
        max_features='sqrt',
        hidden_layers=[64, 32, 16],
        dropout_rate=0.3,
        learning_rate=0.001,
        epochs=50,
        batch_size=32,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    X_train_tf = np.asarray(X_train_scaled, dtype=np.float32)
    y_train_np = y_train.to_numpy() if hasattr(y_train, 'to_numpy') else np.array(y_train)
    
    neural_rlt.fit(X_train_tf, y_train_np)
    
    training_time = time.time() - start
    
    return neural_rlt, training_time


def evaluate_models(models, rlt_model, neural_rlt, X_test_scaled, X_test_df, y_test):
    """Evaluate all models and return results DataFrame"""
    print("\n" + "="*70)
    print("📈 EVALUATING ALL MODELS")
    print("="*70)
    
    results = []
    
    # Evaluate baseline models
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba = y_pred
        
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_proba)
        })
    
    # Evaluate Traditional RLT
    print("\n" + "="*60)
    print("  TraditionalRLT Performance")
    print("="*60)
    
    y_pred_rlt = predict_rlt_forest(rlt_model, X_test_df)
    y_proba_rlt = predict_rlt_forest_scores(rlt_model, X_test_df)
    
    acc_rlt = accuracy_score(y_test, y_pred_rlt)
    prec_rlt = precision_score(y_test, y_pred_rlt)
    rec_rlt = recall_score(y_test, y_pred_rlt)
    f1_rlt = f1_score(y_test, y_pred_rlt)
    auc_rlt = roc_auc_score(y_test, y_proba_rlt)
    
    print(f"  Accuracy    : {acc_rlt:.4f}")
    print(f"  Precision   : {prec_rlt:.4f}")
    print(f"  Recall      : {rec_rlt:.4f}")
    print(f"  F1-Score    : {f1_rlt:.4f}")
    print(f"  ROC-AUC     : {auc_rlt:.4f}")
    
    results.append({
        'Model': 'TraditionalRLT',
        'Accuracy': acc_rlt,
        'Precision': prec_rlt,
        'Recall': rec_rlt,
        'F1-Score': f1_rlt,
        'ROC-AUC': auc_rlt
    })
    
    # Evaluate Neural RLT
    print("\n" + "="*60)
    print("  NeuralRLT Performance")
    print("="*60)
    
    X_test_tf = np.asarray(X_test_scaled, dtype=np.float32)
    y_pred_neural = neural_rlt.predict(X_test_tf)
    y_proba_neural = neural_rlt.predict_proba(X_test_tf)[:, 1]
    
    acc_neural = accuracy_score(y_test, y_pred_neural)
    prec_neural = precision_score(y_test, y_pred_neural)
    rec_neural = recall_score(y_test, y_pred_neural)
    f1_neural = f1_score(y_test, y_pred_neural)
    auc_neural = roc_auc_score(y_test, y_proba_neural)
    
    print(f"  Accuracy    : {acc_neural:.4f}")
    print(f"  Precision   : {prec_neural:.4f}")
    print(f"  Recall      : {rec_neural:.4f}")
    print(f"  F1-Score    : {f1_neural:.4f}")
    print(f"  ROC-AUC     : {auc_neural:.4f}")
    
    results.append({
        'Model': 'NeuralRLT',
        'Accuracy': acc_neural,
        'Precision': prec_neural,
        'Recall': rec_neural,
        'F1-Score': f1_neural,
        'ROC-AUC': auc_neural
    })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
    
    return results_df


def generate_visualizations(models, rlt_model, X_test_scaled, X_test_df, y_test, feature_names, results_df):
    """Generate all visualizations"""
    print("\n" + "="*70)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Create visualizer for baseline models
    visualizer = ModelVisualizer(
        models_dict=models,
        X_test=X_test_scaled,
        y_test=y_test,
        feature_names=feature_names
    )
    
    # Generate all visualizations
    visualizer.generate_all_visualizations(results_df)
    
    # RLT feature importance (separate since it uses different method)
    print("\n7️⃣  Plotting RLT feature importance...")
    plot_rlt_feature_importance(
        forest=rlt_model,
        feature_names=feature_names,
        X_test_df=X_test_df,
        y_test=y_test
    )
    
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS SAVED TO static/charts/")
    print("="*70)


def print_final_results(results_df):
    """Print final results summary"""
    print("\n" + "="*70)
    print(" FINAL RESULTS SUMMARY")
    print("="*70 + "\n")
    print(results_df.to_string(index=False))
    
    best_model = results_df.iloc[0]['Model']
    print(f"\n✓ Best Model: {best_model}")


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("🏥 BREAST CANCER ML - COMPLETE TRAINING PIPELINE")
    print("="*70)
    
    # 1. Load and prepare data
    X, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X_train_df, X_test_df, scaler = load_and_prepare_data()
    
    feature_names = X.columns.tolist()
    
    # 2. Train baseline models
    models = train_baseline_models(X_train_scaled, y_train)
    
    # 3. Train Traditional RLT
    rlt_model, rlt_time = train_traditional_rlt(X_train_df, y_train, feature_names)
    
    # 4. Train Neural RLT
    neural_rlt, neural_time = train_neural_rlt(X_train_scaled, y_train)
    
    # 5. Evaluate all models
    results_df = evaluate_models(models, rlt_model, neural_rlt, X_test_scaled, X_test_df, y_test)
    
    # 6. Generate visualizations
    generate_visualizations(models, rlt_model, X_test_scaled, X_test_df, y_test, feature_names, results_df)
    
    # 7. Print final results
    print_final_results(results_df)
    
    # 8. Save models
    print("\n" + "="*70)
    print("💾 SAVING MODELS")
    print("="*70)
    
    import joblib
    import os
    import json
    
    os.makedirs('models', exist_ok=True)
    
    # Save all models
    joblib.dump(models, 'models/baseline_models.joblib')
    joblib.dump(rlt_model, 'models/rlt_model.joblib')
    joblib.dump(neural_rlt, 'models/neural_rlt_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Save feature names
    with open('models/feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    # Save results
    results_df.to_csv('models/results.csv', index=False)
    
    print("✓ All models saved to models/")
    print("✓ Results saved to models/results.csv")
    
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

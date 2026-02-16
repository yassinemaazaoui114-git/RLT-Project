"""Data preprocessing utilities."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import TEST_SIZE, RANDOM_STATE


def preprocess_data(df):
    """Preprocess the dataset: encode, remove outliers, split, scale."""
    
    # Drop ID column
    X = df.drop(columns=["id", "diagnosis"])
    y = df["diagnosis"].map({"B": 0, "M": 1}).astype(int)
    
    # Remove outliers using IQR
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
    
    print(f"✓ Outliers removed: {len(X) - mask.sum()} samples")
    
    X = X[mask]
    y = y[mask]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Train set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

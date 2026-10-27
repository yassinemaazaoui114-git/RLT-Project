"""Data loading utilities."""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import BASE_DIR, DATA_PATH, FEATURE_COLS, RANDOM_STATE, TEST_SIZE


def load_data():
    """Load the Wisconsin Diagnostic Breast Cancer dataset (569 x 32)."""
    return pd.read_csv(Path(BASE_DIR) / DATA_PATH, header=None, names=FEATURE_COLS)


def load_split():
    """
    Return the train/test split shared by every script and model.

    Stratified split (test_size=0.2, random_state=42); target is 1 = Malignant,
    0 = Benign; features are standardised with a scaler fitted on the training set.
    Returns X_train, X_test (DataFrames of scaled features), y_train, y_test and the scaler.
    """
    df = load_data()
    X = df.drop(columns=["id", "diagnosis"])
    y = df["diagnosis"].map({"B": 0, "M": 1}).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    return X_train, X_test, y_train, y_test, scaler

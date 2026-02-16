"""Data loading utilities."""
import pandas as pd
from pathlib import Path
from config import DATA_PATH, FEATURE_COLS


def load_data():
    """Load the breast cancer dataset."""
    data_path = Path(DATA_PATH)
    df = pd.read_csv(data_path, header=None, names=FEATURE_COLS)
    
    print(f"✓ Dataset loaded: {df.shape}")
    print(f"✓ Class distribution:\n{df['diagnosis'].value_counts()}")
    
    return df

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_breast_cancer(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    path = DATA_DIR / dataset_name
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {path}")
    df = pd.read_csv(path)
    to_drop = [c for c in ["id", "Unnamed: 32"] if c in df.columns]
    df = df.drop(columns=to_drop)
    if "diagnosis" in df.columns:
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    print(y.value_counts())
    print("Shape de X:", X.shape)
    print("Shape de y:", y.shape)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Shape de X_scaled:", X_scaled.shape)
    return X_scaled, y.values

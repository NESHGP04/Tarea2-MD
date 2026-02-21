from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
N_USERS_ML100K = 943
N_ITEMS_ML100K = 1682


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


def load_movielens_100k(folder_name: str) -> tuple[csr_matrix, np.ndarray]:
    base = DATA_DIR / folder_name
    if not base.is_dir():
        raise FileNotFoundError(f"No se encontró la carpeta: {base}")
    data_path = base / "u.data"
    user_path = base / "u.user"
    if not data_path.exists() or not user_path.exists():
        raise FileNotFoundError(f"Faltan u.data o u.user en {base}")
    ratings = pd.read_csv(
        data_path, sep="\t", header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    rows = ratings["user_id"].values - 1
    cols = ratings["item_id"].values - 1
    data = ratings["rating"].values
    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(N_USERS_ML100K, N_ITEMS_ML100K),
    )
    matrix = matrix.astype(float)
    mean_rating = matrix.data.mean()
    matrix.data -= mean_rating
    users_df = pd.read_csv(
        user_path, sep="|", header=None,
        names=["user_id", "age", "gender", "occupation", "zip"],
        usecols=["user_id", "gender"],
    )
    users_df = users_df.sort_values("user_id").reset_index(drop=True)
    y = users_df["gender"].map({"M": 1, "F": 0}).values
    print(users_df["gender"].value_counts())
    print("Shape de X (matriz usuario-item):", matrix.shape)
    print("Shape de y:", y.shape)
    return matrix, y

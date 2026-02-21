from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROJECT_ROOT = DATA_DIR.parent
N_USERS_ML100K = 943
N_ITEMS_ML100K = 1682
EEG_DOWNSAMPLE_STEP = 100


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


def _eeg_ica_base(folder_name: str) -> Path:
    candidate = DATA_DIR / folder_name
    if candidate.is_dir():
        return candidate
    if folder_name == "IAC_data":
        root_candidate = PROJECT_ROOT / "IAC_data"
        if root_candidate.is_dir():
            return root_candidate
    raise FileNotFoundError(f"No se encontró la carpeta: {candidate}")


def load_eeg_ica(folder_name: str) -> tuple[np.ndarray, np.ndarray]:
    import pyedflib
    base = _eeg_ica_base(folder_name)
    info_path = base / "subject-info.csv"
    if not info_path.exists():
        raise FileNotFoundError(f"No se encontró {info_path}")
    subjects_df = pd.read_csv(info_path)
    subjects_df["Subject"] = subjects_df["Subject"].astype(str)
    y_per_subject = subjects_df["Count quality"].values
    list_X = []
    list_y = []
    for i in range(len(subjects_df)):
        subj_id = subjects_df["Subject"].iloc[i]
        edf_path = base / f"{subj_id}_2.edf"
        if not edf_path.exists():
            continue
        with pyedflib.EdfReader(str(edf_path)) as f:
            n_sigs = f.signals_in_file
            n_samples = f.getNSamples()[0]
            sigbufs = np.zeros((n_sigs, n_samples))
            for ch in range(n_sigs):
                sigbufs[ch, :] = f.readSignal(ch)
        X_subj = sigbufs.T
        step = EEG_DOWNSAMPLE_STEP
        X_subj = X_subj[::step]
        list_X.append(X_subj)
        list_y.append(np.full(X_subj.shape[0], y_per_subject[i]))
    X = np.vstack(list_X)
    y = np.concatenate(list_y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(pd.Series(y).value_counts().sort_index())
    print("Shape de X (EEG, muestras x canales):", X_scaled.shape)
    print("Shape de y:", y.shape)
    return X_scaled, y

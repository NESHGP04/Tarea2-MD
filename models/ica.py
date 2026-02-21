import numpy as np
from sklearn.decomposition import FastICA

from models.base import BaseModel
from models.datasets import load_eeg_ica


class IcaModel(BaseModel):
    def __init__(self, n_components: int = 2, random_state: int = 42):
        self._model = FastICA(
            n_components=n_components,
            random_state=random_state,
        )

    def _load_and_preprocess(self, dataset_name: str):
        return load_eeg_ica(dataset_name)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self._model.fit_transform(X)

    def get_title(self) -> str:
        return "ICA - EEG Mental Arithmetic"

    def get_colorbar_label(self) -> str:
        return "Count quality (0=B, 1=G)"

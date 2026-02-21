"""UMAP sobre Breast Cancer (mismo dataset que t-SNE)."""

import numpy as np
import umap

from models.base import BaseModel
from models.datasets import load_breast_cancer


class UmapModel(BaseModel):
    def __init__(self, n_components: int = 2, random_state: int = 42):
        self._model = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
        )

    def _load_and_preprocess(self, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
        return load_breast_cancer(dataset_name)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self._model.fit_transform(X)

    def get_title(self) -> str:
        return "UMAP - Breast Cancer Dataset"

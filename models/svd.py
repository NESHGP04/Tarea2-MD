import numpy as np
from sklearn.decomposition import TruncatedSVD

from models.base import BaseModel
from models.datasets import load_movielens_100k


class SvdModel(BaseModel):
    def __init__(self, n_components: int = 2, random_state: int = 42):
        self._model = TruncatedSVD(
            n_components=n_components,
            random_state=random_state,
        )

    def _load_and_preprocess(self, dataset_name: str):
        return load_movielens_100k(dataset_name)

    def fit_transform(self, X):
        return self._model.fit_transform(X)

    def get_title(self) -> str:
        return "SVD - MovieLens 100K"

    def get_colorbar_label(self) -> str:
        return "Gender (0=F, 1=M)"

import numpy as np

from models.base import BaseModel


class IcaModel(BaseModel):
    def _load_and_preprocess(self, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_title(self) -> str:
        return "ICA"

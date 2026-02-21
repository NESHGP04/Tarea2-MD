from abc import ABC, abstractmethod
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


class BaseModel(ABC):
    @abstractmethod
    def _load_and_preprocess(self, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_title(self) -> str:
        pass

    def run(self, dataset_name: str) -> np.ndarray:
        X, y = self._load_and_preprocess(dataset_name)
        X_transformed = self.fit_transform(X)
        print("Shape transformado:", X_transformed.shape)
        self._plot(X_transformed, y)
        return X_transformed

    def _plot(self, X_transformed: np.ndarray, y: np.ndarray) -> None:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^\w\s-]", "", self.get_title()).strip()
        name = re.sub(r"[-\s]+", "_", name)
        path = OUTPUTS_DIR / f"{name}.png"
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            X_transformed[:, 0],
            X_transformed[:, 1],
            c=y,
        )
        plt.title(self.get_title())
        plt.xlabel("Componente 1")
        plt.ylabel("Componente 2")
        plt.colorbar(scatter, label="Diagnosis (0=Benigno, 1=Maligno)")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Gráfico guardado: {path}")

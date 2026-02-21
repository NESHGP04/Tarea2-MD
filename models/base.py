from abc import ABC, abstractmethod
import numpy as np


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
        import matplotlib.pyplot as plt
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
        plt.show()

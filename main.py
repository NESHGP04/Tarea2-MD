import argparse

from models import MODEL_REGISTRY


class Main:
    def __init__(self, model_name: str, dataset_name: str):
        self._model_name = model_name.lower()
        self._dataset_name = dataset_name
        self._model_class = MODEL_REGISTRY.get(self._model_name)
        if self._model_class is None:
            raise ValueError(
                f"Modelo '{model_name}' no existe. "
                f"Disponibles: {list(MODEL_REGISTRY.keys())}"
            )

    def run(self):
        model = self._model_class()
        return model.run(self._dataset_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Modelo a usar: tsne, umap, svd, ica",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Nombre del archivo CSV en data/ (ej: tSNE_and_UMAP_data.csv)",
    )
    args = parser.parse_args()
    app = Main(model_name=args.model, dataset_name=args.dataset)
    app.run()


if __name__ == "__main__":
    main()

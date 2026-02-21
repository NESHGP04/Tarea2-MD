"""Registro de modelos: tsne, umap, svd, ica. Main elige por nombre."""

from models.base import BaseModel
from models.tsne import TsneModel
from models.umap import UmapModel
from models.svd import SvdModel
from models.ica import IcaModel

MODEL_REGISTRY = {
    "tsne": TsneModel,
    "umap": UmapModel,
    "svd": SvdModel,
    "ica": IcaModel,
}

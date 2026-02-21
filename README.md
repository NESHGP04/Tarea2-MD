# Tarea 2 Investigativa — Minería de Datos (UVG)

Proyecto que implementa cuatro modelos **no supervisados** de reducción de dimensionalidad o factorización, cada uno sobre un dataset distinto, con visualización 2D y guardado de gráficos.

**Autores:**
- Marinés García 23391
- Camila Richter 23183
- Esteban Cárcamo 23016

---

## Modelos y datasets

| Modelo | Dataset | Descripción breve |
|--------|--------|-------------------|
| **t-SNE** | Breast Cancer Wisconsin | 569 muestras, 30 variables; proyección 2D por diagnosis (B/M). |
| **UMAP** | Breast Cancer Wisconsin | Mismo dataset que t-SNE para comparar métodos. |
| **SVD** | MovieLens 100K | Matriz usuario-ítem (943×1682), centrada; proyección por género. |
| **ICA** | EEG aritmética mental | 36 sujetos, 23 canales EDF; proyección por Count quality (B/G). |

Los datos deben estar en la carpeta `data/` (o `IAC_data/` en la raíz para ICA). Ver sección [Estructura de datos](#estructura-de-datos).

---

## Requisitos

- Python 3.12 (recomendado; pyedflib puede fallar en 3.14).
- Dependencias en `requirements.txt`.

```bash
python3.12 -m venv venv
source venv/bin/activate   # en Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Uso

Desde la raíz del proyecto (con el venv activado):

```bash
# t-SNE (Breast Cancer)
python main.py --model tsne --dataset tSNE_and_UMAP_data.csv

# UMAP (Breast Cancer)
python main.py --model umap --dataset tSNE_and_UMAP_data.csv

# SVD (MovieLens 100K)
python main.py --model svd --dataset SVD_data

# ICA (EEG)
python main.py --model ica --dataset IAC_data
```

El argumento `--dataset` es el **nombre** del archivo o carpeta dentro de `data/`. Para ICA, si la carpeta está en la raíz del proyecto (`IAC_data/`), también se detecta automáticamente.

Cada ejecución imprime en terminal estadísticas del dataset y el shape de la proyección, y guarda el gráfico en `outputs/`.

---

## Estructura del proyecto

```
.
├── main.py              # Punto de entrada; clase Main y argumentos CLI
├── requirements.txt
├── README.md
├── INFORME_MODELOS.md   # Informe: dataset, resultados e interpretación por modelo
├── data/                # Datos (CSV y/o carpetas)
│   ├── tSNE_and_UMAP_data.csv
│   ├── SVD_data/        # MovieLens 100K (u.data, u.user, etc.)
│   └── (opcional) IAC_data/
├── IAC_data/            # Alternativa: EEG en la raíz
└── models/
    ├── __init__.py      # MODEL_REGISTRY (tsne, umap, svd, ica)
    ├── base.py          # BaseModel: carga → fit_transform → plot → guardar PNG
    ├── datasets.py      # load_breast_cancer, load_movielens_100k, load_eeg_ica
    ├── tsne.py          # TsneModel
    ├── umap.py          # UmapModel
    ├── svd.py           # SvdModel
    └── ica.py           # IcaModel
```

---

## Estructura de datos

- **Breast Cancer:** CSV en `data/` con columnas de características y `diagnosis` (M/B). Se eliminan `id` y `Unnamed: 32`; se escala con StandardScaler.
- **MovieLens 100K:** Carpeta `data/SVD_data/` con `u.data` (ratings) y `u.user` (género). Se construye matriz usuario-ítem y se centra restando la media.
- **EEG (ICA):** Carpeta `IAC_data/` (en `data/` o en la raíz) con `subject-info.csv` y archivos `SubjectXX_2.edf`. Se usa la grabación durante la tarea (`_2`), submuestreo y StandardScaler.

---

## Salidas

- **Terminal:** Conteos/shapes del dataset y shape de la proyección 2D.
- **Gráficos:** Se guardan en `outputs/` en PNG, con nombre derivado del título del modelo (ej. `t_SNE_Breast_Cancer_Dataset.png`, `SVD_MovieLens_100K.png`, `ICA_EEG_Mental_Arithmetic.png`).

---

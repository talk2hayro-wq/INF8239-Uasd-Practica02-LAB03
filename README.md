# INF-8239 · Unidad 01 — Práctica 02: Baseline vs SVM sobre datos reales (JAC)

Comparación de un baseline (`DummyClassifier`) contra una SVM RBF para anticipar **días de alta demanda** en los aeropuertos internacionales de República Dominicana, usando el dataset público de la Junta de Aviación Civil (JAC).

> Ejercicio académico. No es un modelo operativo de planificación aeroportuaria.

## Resultado en una tabla

Prueba temporal 2025-01-01 → 2026-06-30 (3 627 días). Clase positiva = 1 (alta demanda).

| Modelo | F1 macro | Recall clase 1 | Precisión clase 1 | ROC-AUC | FP | FN |
|---|---|---|---|---|---|---|
| Baseline (`most_frequent`) | 0.429 | 0.000 | 0.000 | — | 0 | 898 |
| SVM base (C=1, gamma=scale) | 0.762 | 0.543 | 0.737 | 0.866 | 174 | 410 |
| SVM `GridSearchCV` (C=10, gamma=0.1) | 0.760 | 0.556 | 0.714 | 0.861 | 200 | 399 |

F1 macro por validación cruzada temporal (solo entrenamiento) del modelo elegido: **0.697**. Los tres mejores del grid están empatados dentro del ruido entre folds. Detalle e interpretación en el notebook.

## Contenido

```
├── README.md
├── 02_svm_jac_pasajeros.ipynb     Notebook completo y ejecutado
├── pract02.py                     Mismo flujo como script
├── requirements.txt · pytest.ini · .gitignore
├── src/inf8239_u01/
│   ├── data.py                    download_csv, ensure_dataset, load_raw, clean_jac
│   ├── features.py                calendario, target alta_demanda, split temporal
│   ├── models.py                  build_svm (unidad 01), Dummy y SVM con Pipeline, GridSearchCV
│   └── evaluate.py                métricas y comparación con la misma partición
├── tests/                         25 pruebas (4 archivos)
├── docs/
│   ├── PROCEDENCIA_LICENCIA.md    Ficha de procedencia y licencia
│   └── DICCIONARIO_DATOS.md       Diccionario de datos
├── reports/                       Evidencias: métricas, tabla del grid, matrices, salidas de pract02 y pytest
└── data/raw/                      CSV descargado (no se versiona)
```

## Instalación (Windows 11 · PowerShell)

```powershell
cd C:\INF8239_U01
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name inf8239 --display-name "INF-8239 (.venv)"
```

Si PowerShell bloquea la activación: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

Versiones con las que se generaron las evidencias: Python 3.12, pandas 3.0, scikit-learn 1.8, NumPy 2.4.

## Descarga del dataset

`ensure_dataset` descarga el CSV a `data/raw/dataset.csv` **solo si no existe**:

```python
from src.inf8239_u01.data import URL_JAC, ensure_dataset
path = ensure_dataset(URL_JAC)
```

Fuente: `https://jac.gob.do/wp-content/uploads/2026/07/Movimiento-de-Entrada-y-Salida-de-Pasajeros-por-Aeropuertos-Junta-de-Aviacion-Civil-JAC-2018-2026.csv`

Procedencia, hash, fecha de acceso, licencia y aviso de atribución: [`docs/PROCEDENCIA_LICENCIA.md`](docs/PROCEDENCIA_LICENCIA.md). **Licencia del dataset: ODbL v1.0** (Open Data Commons Open Database License). El CSV no se sube al repositorio: se descarga siempre desde la fuente.

## Definiciones

| Elemento | Definición |
|---|---|
| **Unidad de análisis** | Un aeropuerto en un día (fila aeropuerto-día) · 19 971 filas · 7 aeropuertos · 2018-01-01 a 2026-06-30 |
| **Target** | `alta_demanda` = 1 si `entrada + salida` supera el **percentil 75 del mismo aeropuerto y año**; ≈ 25 % de positivos en cada año |
| **Variables** | `aeropuerto`, `mes`, `dia_semana` (categóricas, one-hot) y `dia` (numérica, escalada) |
| **Métrica principal** | **F1 macro**. Secundarias: recall y precisión de la clase 1, ROC-AUC, matriz de confusión (FP/FN) |
| **Partición** | Temporal única: entrenamiento 2018-2024 (16 344 filas), prueba 2025-2026 (3 627 filas), idéntica para Dummy y SVM |
| **Selección de hiperparámetros** | `GridSearchCV` (C ∈ {0.1, 1, 10} × gamma ∈ {scale, 0.01, 0.1}), `scoring="f1_macro"`, `TimeSeriesSplit(4)` **solo con entrenamiento** |

### Identificadores y fugas

- **Fuga:** `entrada`, `salida` y `total` son el resultado del día y generan el target; no entran en X. También quedan fuera `umbral_p75` (derivado del target), `fecha` (solo ordena y parte en el tiempo) y `anio` (el target ya es relativo al año).
- **Identificadores:** el dataset no tiene columna identificadora de fila. `aeropuerto` es una categoría de 7 valores conocida de antemano.
- **Preprocesamiento dentro del `Pipeline`:** imputación, escalado y one-hot se ajustan solo con los datos de entrenamiento de cada partición y de cada fold.
- **Limitación:** el umbral de cada año usa la distribución de ese año. Es una definición de la etiqueta (no una variable), pero en producción no se conocería de antemano.

Diccionario completo: [`docs/DICCIONARIO_DATOS.md`](docs/DICCIONARIO_DATOS.md).

## Ejecución

Desde la raíz del proyecto, con el entorno activado:

```powershell
python pract02.py                      # flujo completo (~2 min); escribe en reports/
python -m pytest -v                    # 25 pruebas
jupyter nbconvert --to notebook --execute --inplace 02_svm_jac_pasajeros.ipynb
```

También puede abrirse el notebook en Jupyter o VS Code con el kernel «INF-8239 (.venv)». Para ejecutar solo la parte offline, coloca el CSV en `data/raw/dataset.csv`.

## Pruebas

`python -m pytest -v` → **25 aprobadas** (salida en `reports/pytest_output.txt`).

| Archivo | Pruebas | Qué cubre |
|---|---|---|
| `test_data.py` | 9 | Rechazo de URL no HTTP y de datasets vacíos, no redescarga, `-` como cero, texto inválido, armonización 11→7, fechas y meses inválidos, BOM |
| `test_features.py` | 6 | Umbral P75 estricto por aeropuerto-año, ausencia de fugas en X, día de la semana, partición temporal sin solape |
| `test_models.py` | 8 | Pipeline con `prep` + `model`, categoría no vista, misma partición para Dummy y SVM, conteo de FP/FN, CV temporal, unidad 01 |
| `test_dataset_real.py` | 2 | Forma, aeropuertos, fechas y balance del CSV real (se omiten si no está descargado) |

## Historial de commits

Commits pequeños y con mensaje descriptivo, en este orden: estructura del proyecto → datos → pruebas de datos → target y partición → pruebas del target → modelos y evaluación → pruebas de modelos → `pract02.py` con evidencias → documentación de datos → notebook → README. Consultar con `git log --oneline`.

## Licencia y atribución

- **Dataset:** Open Data Commons Open Database License (ODbL) v1.0, con atribución a la Junta de Aviación Civil (JAC). Aviso sugerido y condiciones (atribución, share-alike, mantener abierto) en [`docs/PROCEDENCIA_LICENCIA.md`](docs/PROCEDENCIA_LICENCIA.md).
- **Código y texto del repositorio:** sin licencia declarada (trabajo académico).

## Limitaciones y siguientes pasos

- Solo 4 variables de calendario; faltan feriados, temporadas y calendario de vuelos.
- El óptimo del grid está en el borde (C=10, gamma=0.1): ampliar la búsqueda.
- Ajustar el umbral de decisión o `class_weight` si el falso negativo pesa más, decidiéndolo con validación temporal y no con el test.
- Si se publica el dataset limpio (una base de datos derivada), debe ofrecerse bajo la misma ODbL y con atribución a la JAC.

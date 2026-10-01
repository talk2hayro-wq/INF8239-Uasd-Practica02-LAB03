# INF-8239 · Unidad 01 — LAB02 y LAB03 sobre datos reales (JAC)

Comparación de un baseline (`DummyClassifier`) contra una SVM RBF para anticipar **días de alta demanda** en los aeropuertos internacionales de República Dominicana, usando el dataset público de la Junta de Aviación Civil (JAC).

> Ejercicio académico. No es un modelo operativo de planificación aeroportuaria.

**Índice.** **LAB02** (práctica 02: baseline vs SVM) ocupa las secciones hasta «Limitaciones y siguientes pasos». **LAB03** (ensambles, reducción dimensional, mediciones repetidas y frontera de Pareto) está en la sección [LAB03](#lab03--ensambles-reducción-dimensional-mediciones-repetidas-y-frontera-de-pareto) al final.

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
├── 02_svm_jac_pasajeros.ipynb     LAB02: notebook completo y ejecutado
├── pract02.py                     LAB02: mismo flujo como script
├── lab03_seleccion.py             LAB03: selección por CV temporal (no toca la prueba)
├── lab03_cierre.py                LAB03: evaluación única en la prueba
├── lab03_energia.py               LAB03: estimación de energía y CO2 (CodeCarbon)
├── requirements.txt · requirements-energia.txt · pytest.ini · .gitignore
├── src/inf8239_u01/
│   ├── data.py                    download_csv, ensure_dataset, load_raw, clean_jac
│   ├── features.py                calendario, target alta_demanda, split temporal, train_split
│   ├── models.py                  build_svm (unidad 01), Dummy y SVM con Pipeline, GridSearchCV
│   ├── evaluate.py                métricas y comparación con la misma partición
│   ├── lab03_config.py            LAB03: declaración previa (métrica, clase, regla, costos)
│   ├── catalog.py                 LAB03: catálogo de 9 configuraciones
│   ├── green.py                   LAB03: Pareto, mediciones repetidas, decisión, guarda de la prueba
│   └── explore.py                 LAB03: memoria densa, PCA y t-SNE
├── tests/                         74 pruebas (10 archivos)
├── docs/
│   ├── PROCEDENCIA_LICENCIA.md    Ficha de procedencia y licencia (ODbL)
│   ├── DICCIONARIO_DATOS.md       Diccionario de datos
│   └── LAB03_declaracion.md       LAB03: reglas fijadas antes de comparar
├── reports/                       Evidencias de LAB02 y LAB03
└── data/raw/                      CSV descargado (no se versiona)
```

## Instalación (Windows 11 · PowerShell)

```powershell
cd C:\INF8239_U01
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-energia.txt   # opcional: solo para lab03_energia.py
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
python -m pytest -v                    # 74 pruebas (LAB02 + LAB03)
jupyter nbconvert --to notebook --execute --inplace 02_svm_jac_pasajeros.ipynb
```

También puede abrirse el notebook en Jupyter o VS Code con el kernel «INF-8239 (.venv)». Para ejecutar solo la parte offline, coloca el CSV en `data/raw/dataset.csv`.

## Pruebas

`python -m pytest -v` → **74 aprobadas** (LAB02: `reports/pytest_output.txt`, 25 pruebas; completa con LAB03: `reports/lab03_pytest_output.txt`).

| Archivo | Pruebas | Qué cubre |
|---|---|---|
| `test_data.py` | 9 | Rechazo de URL no HTTP y de datasets vacíos, no redescarga, `-` como cero, texto inválido, armonización 11→7, fechas y meses inválidos, BOM |
| `test_features.py` | 6 | Umbral P75 estricto por aeropuerto-año, ausencia de fugas en X, día de la semana, partición temporal sin solape |
| `test_models.py` | 8 | Pipeline con `prep` + `model`, categoría no vista, misma partición para Dummy y SVM, conteo de FP/FN, CV temporal, unidad 01 |
| `test_dataset_real.py` | 2 | Forma, aeropuertos, fechas y balance del CSV real (se omiten si no está descargado) |
| `test_green.py` | 12 | Las dos pruebas del guion de Pareto, empates, varios costos, mediciones repetidas, regla de decisión, guarda de la prueba |
| `test_catalog.py` | 15 | ≥ 6 candidatos y 3 familias, preprocesamiento dentro del `Pipeline` y denso, sin `probability`, PCA, ajuste de cada modelo |
| `test_explore.py` | 5 | Memoria densa vs dispersa, PCA, muestra de entrenamiento, estabilidad de t-SNE y efecto de la semilla |
| `test_protocolo_lab03.py` | 5 | La selección no menciona la prueba, misma partición que LAB02, reglas declaradas, documento de declaración |
| `test_lab03_evidencia.py` | 8 | CSV con 3 repeticiones y mediana, Pareto recalculable, decisión reproducible desde el CSV, t-SNE, PCA, cierre único |
| `test_lab03_energia.py` | 4 | Ayuda del script, alcance declarado, solo entrenamiento, corrida de humo con CodeCarbon |

## Historial de commits

Commits pequeños y con mensaje descriptivo (`git log --oneline`). **LAB02:** estructura → datos → pruebas → target y partición → modelos y evaluación → `pract02.py` con evidencias → documentación de datos → notebook → README → licencia. **LAB03:** declaración previa de reglas → `green` (Pareto, mediciones, decisión) → catálogo → `explore` (PCA y t-SNE) → script de selección → **evidencia y decisión congelada** → script de cierre → **evaluación única de la prueba** → energía → README.

El orden es deliberado: la declaración precede a todo el código de modelos, y la decisión queda versionada **antes** del commit que toca la prueba.

## Licencia y atribución

- **Dataset:** Open Data Commons Open Database License (ODbL) v1.0, con atribución a la Junta de Aviación Civil (JAC). Aviso sugerido y condiciones (atribución, share-alike, mantener abierto) en [`docs/PROCEDENCIA_LICENCIA.md`](docs/PROCEDENCIA_LICENCIA.md).
- **Código y texto del repositorio:** sin licencia declarada (trabajo académico).

## Limitaciones y siguientes pasos de LAB02

- Solo 4 variables de calendario; faltan feriados, temporadas y calendario de vuelos.
- El óptimo del grid está en el borde (C=10, gamma=0.1): ampliar la búsqueda.
- Ajustar el umbral de decisión o `class_weight` si el falso negativo pesa más, decidiéndolo con validación temporal y no con el test.
- Si se publica el dataset limpio (una base de datos derivada), debe ofrecerse bajo la misma ODbL y con atribución a la JAC.

---

# LAB03 · Ensambles, reducción dimensional, mediciones repetidas y frontera de Pareto

Amplía el problema de LAB02 (¿un día será de **alta demanda** en un aeropuerto?) comparando **9 configuraciones** (SVM, Random Forest y boosting, con y sin PCA), repitiendo las mediciones de costo y decidiendo con la **frontera de Pareto**. Se reutilizan sin cambios el dataset, el target, las exclusiones y la partición temporal de LAB02.

> Ejercicio académico. Los tiempos dependen del equipo donde se ejecutan (ver «Cómo leer los tiempos»).

## Reglas declaradas antes de comparar

Quedaron versionadas en `docs/LAB03_declaracion.md` y `src/inf8239_u01/lab03_config.py`, en un commit anterior a todo el código de modelos.

| Declaración | Valor |
|---|---|
| **Métrica principal** | F1 macro, promedio de una validación cruzada temporal de 4 folds **solo sobre entrenamiento (2018-2024)** |
| **Clase prioritaria** | Alta demanda (clase 1), con su recall: el falso negativo es el error caro |
| **Costos** | Tiempo de ajuste (principal), inferencia y tamaño serializado; **3 repeticiones, mediana**, 1 hilo |
| **Regla de decisión** | Elegibles: F1 macro y recall de la clase 1 a ≤ 1 desviación estándar del mejor. Entre los elegibles en la frontera de Pareto (F1 contra tiempo de ajuste), el de menor tiempo de ajuste; desempate por tamaño |
| **Prueba 2025-2026** | No se toca hasta el cierre; una sola evaluación del modelo congelado |

## Resultados de la selección (validación cruzada temporal, sin tocar la prueba)

| Modelo | Familia | F1 macro CV | Recall clase 1 | Ajuste (s) | Inferencia (ms, 3 000 filas) | Tamaño (KB) | Frontera de Pareto | Elegible |
|---|---|---|---|---|---|---|---|---|
| `svm_c10_g01` | SVM | 0.697 ± 0.067 | 0.478 | 7.49 | 1076 | 1621 | ✔ | ✔ |
| `svm_c10` | SVM | 0.695 ± 0.064 | 0.492 | 9.20 | 962 | 1579 |  | ✔ |
| `svm_c1` **← elegido** | SVM | 0.694 ± 0.071 | 0.449 | 5.05 | 964 | 1699 | ✔ | ✔ |
| `svm_c1_pca` | SVM + PCA | 0.677 ± 0.088 | 0.404 | 4.85 | 980 | 1427 | ✔ |  |
| `boost` | Boosting | 0.674 ± 0.067 | 0.418 | 0.38 | 13 | 363 | ✔ |  |
| `logistic` | lineal | 0.534 ± 0.037 | 0.172 | 0.03 | 5 | 5 | ✔ |  |
| `rf_300` | Random Forest | 0.494 ± 0.044 | 0.083 | 1.18 | 33 | 6877 |  |  |
| `rf_100` | Random Forest | 0.493 ± 0.041 | 0.081 | 0.42 | 17 | 2286 |  |  |
| `dummy` | referencia | 0.430 ± 0.007 | 0.000 | 0.02 | 5 | 4 | ✔ |  |

Frontera de Pareto: F1 macro contra tiempo de ajuste. En `reports/green_ai_results.csv` también están las fronteras contra inferencia, contra tamaño y contra los tres costos a la vez. Figura: `reports/pareto.png`.

- **Las SVM dominan en rendimiento**, pero son las más lentas y las de inferencia más costosa (~1.3 s para 3 000 filas).
- **Boosting (`boost`) es el punto eficiente:** frente a `svm_c10_g01` (F1 macro -2.3 puntos, recall de la clase 1 -6.0 puntos; ajuste 19× más rápido, inferencia 84× más rápida, modelo 4.5× más pequeño).
- **Random Forest queda en 0.49 de F1 macro** (recall de la clase 1 ≈ 0.08) con los hiperparámetros del guion (profundidad 8, 100/300 árboles, sin ajustar). Casi nunca predice la clase 1. No investigué la causa; una hipótesis es que la profundidad limitada y `max_features="sqrt"` sobre 27 columnas *one-hot*, con clases 75/25 y umbral 0.5, lo dejan sub-ajustado. No es una conclusión sobre Random Forest en general.
- **La regresión logística** queda en 0.53: un modelo aditivo no captura las interacciones (aeropuerto × mes × día de la semana).
- **Reproducibilidad con LAB02:** el CV de `svm_c10_g01` da 0.6968, idéntico al de LAB02: la partición y el preprocesamiento son los mismos.

## Decisión cuantificada

Aplicando la regla declarada, **se elige `svm_c1`**. Frente al modelo de mayor F1 macro (`svm_c10_g01`):

| Lo que cambia al elegir `svm_c1` | Valor |
|---|---|
| F1 macro | -0.24 puntos (0.6944 frente a 0.6968) |
| Recall de la clase 1 | -2.89 puntos |
| Tiempo de ajuste | ×0.67 (33 % menos) |
| Inferencia | ×0.90 |
| Tamaño serializado | ×1.05 |

**Lectura honesta.** La regla elige un modelo casi idéntico al mejor, con un ahorro de tiempo de ajuste moderado. La razón es que la condición sobre el recall de la clase 1 (prioritaria) deja fuera a `boost`: su recall queda a 0.021 por debajo del umbral (0.439). Esa es la consecuencia de declarar que el falso negativo es el error caro.

Sensibilidad (**informativa, no cambia la decisión**): qué modelo saldría con otra tolerancia `k` en desviaciones estándar.

| k | 0.5 | 1.0 (declarada) | 1.5 | 2.0 |
|---|---|---|---|---|
| Modelo | `svm_c10_g01` | `svm_c1` | `boost` | `boost` |

Si el objetivo fuese minimizar costo de inferencia y tamaño, y aceptar ~6 puntos menos de recall en la clase prioritaria, `boost` sería la alternativa. Esa es una decisión distinta, con otra regla, y habría que declararla antes de comparar.

## Evaluación final (prueba 2025-2026, una sola vez)

Se ajustó `svm_c1` con el entrenamiento y se evaluó **una sola vez** en la prueba (3 627 días). `lab03_cierre.py` se niega a repetirse. Los otros dos modelos aparecen solo como contexto y no podían cambiar la decisión.

| Modelo | F1 macro | Recall clase 1 | Precisión clase 1 | ROC-AUC | Falsos positivos | Falsos negativos |
|---|---|---|---|---|---|---|
| **`svm_c1` (elegido)** | 0.761 | 0.543 | 0.737 | 0.866 | 174 | 410 |
| `svm_c10_g01` (SVM de LAB02, contexto) | 0.759 | 0.556 | 0.714 | 0.861 | 200 | 399 |
| `dummy` (baseline, contexto) | 0.429 | 0.000 | 0.000 | 0.500 | 0 | 898 |

- F1 macro en la prueba: **0.761**; el CV estimaba 0.694 ± 0.071. La prueba queda +6.7 puntos por encima, dentro de una desviación del CV (hipótesis, no medida: los primeros folds entrenan con pocos datos y cubren 2020-2021).
- El falso negativo sigue siendo el error caro: **410 días de alta demanda no anticipados** frente a 174 falsas alarmas.
- `svm_c1` es el SVM base de LAB02, y `svm_c10_g01` (el del grid de LAB02) reproduce su resultado de LAB02 (0.7595). **Las configuraciones nuevas (Random Forest, boosting, logística y SVM + PCA) no superan en CV a las SVM del grid de LAB02**; lo que aporta LAB03 es saber a qué costo se logra y qué alternativas más baratas existen.

## PCA: ¿aplica aquí?

- **Densidad:** `HistGradientBoosting` y PCA exigen matriz densa. La matriz codificada es de 16344 × 27: **3.53 MB densa frente a 0.85 MB dispersa** (`reports/lab03_memoria_densa.csv`). Densificar es seguro y se midió antes de hacerlo.
- **Bloque numérico:** solo tiene 1 columna (`dia`). PCA sobre una sola columna no reduce nada, por lo que se aplica a la matriz codificada completa, dentro del `Pipeline`.
- **Resultado:** 22 de 27 componentes retienen el 95.7 % de la varianza (reducción de 19 % de las columnas). Las columnas *one-hot* son casi ortogonales, así que hay poco que comprimir.
- **Efecto:** `svm_c1_pca` obtiene F1 macro 0.677 frente a 0.694 de `svm_c1` (-1.8 puntos) y tarda 4.8 s frente a 5.1 s (-4 %). PCA no compensa aquí.

## t-SNE con dos semillas

`reports/tsne_two_seeds.png`: muestra de 1 000 filas **de entrenamiento**, matriz codificada, semillas 42 y 7, coloreada por alta demanda (arriba) y por aeropuerto (abajo). Métricas en `reports/lab03_tsne_metricas.json`.

| | Valor |
|---|---|
| **Qué permanece:** pureza por aeropuerto entre los 10 vecinos más cercanos | 0.80 y 0.77 (azar: 0.15) |
| **Qué cambia:** vecinos compartidos entre los dos mapas | 60 % |
| **Qué cambia:** disparidad de Procrustes (0 = mapas idénticos) | 0.76 |
| Pureza por alta demanda entre los vecinos | 0.68 y 0.66 (azar: 0.62) |

- **Permanece:** el aeropuerto se agrupa con claridad en ambos mapas, muy por encima del azar. Las posiciones absolutas, la orientación y la cercanía entre grupos cambian.
- **Casi no se ve separación por alta demanda:** la pureza queda apenas sobre el azar. Coherente con que, con solo calendario, el problema es difícil (F1 macro ≈ 0.7).
- **Cautela:** la separación visual no demuestra causalidad ni valida un clasificador; t-SNE es exploratorio, no se puede aplicar a datos nuevos, y las distancias entre grupos no son interpretables.
- **Nota sobre `init="pca"`:** con esa inicialización (la del guion) las dos semillas dan mapas **idénticos** (disparidad 6e-32, vecinos compartidos 100 %). Por eso el análisis usa inicialización aleatoria, con la que la semilla sí importa.

## Qué se cambió respecto al guion `LAB03.py`, y por qué

| Guion | En este repositorio | Motivo |
|---|---|---|
| Modelos evaluados y ordenados por F1 **en la prueba** | Selección por **CV temporal en entrenamiento**; prueba una sola vez al cierre | Cumple «prueba sin tocar hasta el cierre» y evita elegir mirando la prueba |
| `preprocess` disperso | `OneHotEncoder(sparse_output=False)` | `HistGradientBoosting` y PCA exigen densidad; costo medido: 3.53 MB |
| `SVC(probability=True)` | `SVC` sin `probability` | ~5 veces más lento y deprecado en scikit-learn ≥ 1.9; el AUC usa `decision_function` |
| Inferencia medida 1 vez sobre la prueba | 3 repeticiones, mediana, sobre 3 000 filas **de entrenamiento** | Cumple «tres repeticiones y mediana» sin usar la prueba |
| `RandomForest(n_jobs=-1)` | `n_jobs=1` y mediciones a 1 hilo | Costo comparable entre modelos y entre equipos |
| `recall_macro` | `recall_alta` (clase 1) y F1 macro | La clase prioritaria declarada es la alta demanda |
| PCA sobre `Xtr` | PCA sobre la matriz codificada, dentro del `Pipeline` | `Xtr` tiene texto (`aeropuerto`); el bloque numérico es 1 columna |
| t-SNE sobre columnas numéricas de `X` | Matriz codificada de una muestra de entrenamiento, `init` aleatorio | `X` incluye la prueba; `mes`/`dia_semana` son categorías y se perdería el aeropuerto |
| `pareto_flags` con un costo, definida en el script | `inf8239_u01.green.pareto_flags`, con uno o varios costos | La prueba del guion la importa de ahí; se piden inferencia y tamaño |
| 6 configuraciones | 8 candidatas + baseline | Se añadieron `svm_c10_g01` (el de LAB02) y `svm_c1_pca` |

## Ejecución de LAB03

Desde la raíz del proyecto, con el entorno activado y en este orden:

```powershell
python lab03_seleccion.py     # ~3 min: CV, mediciones, PCA, t-SNE, Pareto y decisión (NO toca la prueba)
python lab03_cierre.py        # evaluación única en la prueba; se niega a repetirse
python lab03_energia.py       # opcional: estimación de energía y CO2 (requiere requirements-energia.txt)
python -m pytest -v           # 74 pruebas
```

### Cómo leer los tiempos

Los tiempos y tamaños incluidos en `reports/` se generaron en: Linux-6.18.44-fc-v50-x86_64-with-glibc2.39, Python 3.11.15, scikit-learn 1.8.0, **2 núcleo lógico**, 1 hilo. En tu equipo los valores serán distintos (y entre corridas del mismo equipo varían unos pocos por ciento).

- Las **F1 y los recalls** no dependen del equipo; los **tiempos** sí. El orden entre familias (logística < boosting ≈ Random Forest < SVM) es estable; las diferencias pequeñas, no.
- Para regenerar la evidencia en tu equipo: `python lab03_seleccion.py` y luego `python lab03_cierre.py --nueva-corrida`. Esa bandera sobrescribe la evaluación previa y solo debe usarse al rehacer **todo** el protocolo.
- Si tu ejecución elige otro modelo, vale el que salga de la regla declarada con tus mediciones.
- `reports/models/*.joblib` no se versionan (pesan ~16 MB); sus tamaños están en `green_ai_results.csv`.

## Energía y emisiones de CO2 (`lab03_energia.py`)

**El tiempo no es consumo energético.** Este script estima energía y CO2eq con CodeCarbon y declara su alcance:

| Elemento | Detalle |
|---|---|
| **Herramienta** | CodeCarbon ≥ 3.0, modo offline (sin servicios externos) |
| **Región** | `--pais DOM` (República Dominicana) por defecto; el factor de emisión es el **promedio anual de la matriz eléctrica** del país que trae CodeCarbon (≈ 581 g CO2/kWh en mi verificación), no el de la hora ni el de tu proveedor |
| **Carga** | `--repeticiones` ciclos de ajuste + inferencia por modelo dentro del medidor; se reporta **por ciclo** |
| **Cobertura** | CPU, RAM y GPU (si existe) |
| **Supuestos** | La potencia de CPU viene de contadores de hardware si están disponibles; si no, se estima con la carga y el TDP. La de RAM es una **potencia estimada por capacidad, no medida** |
| **No cubre** | Fuente de poder, refrigeración, pantalla, red, fabricación del equipo |

Salidas: `reports/lab03_energia.csv` y `reports/lab03_energia_meta.json` (herramienta, versión, región, equipo, supuestos, cobertura y advertencia).

**Advertencia de interpretación.** En mi corrida de verificación (contenedor de 1 núcleo, **no incluida** en el repositorio porque no representa tu equipo) la RAM aportó más del 70 % de la energía estimada. Como su potencia es una constante estimada, la energía resulta casi proporcional al tiempo: **no añade información que el tiempo de ajuste no dé**. Sirve para comparar modelos entre sí en el mismo equipo, no como consumo eléctrico exacto ni como huella de carbono certificada. Las corridas de pocos segundos quedan por debajo de la resolución del medidor: sube `--repeticiones`.

## Evidencia de LAB03 en `reports/`

| Archivo | Contenido |
|---|---|
| `green_ai_results.csv` | Tabla por modelo: F1 macro CV ± desv., recall, ajuste (mediana, mín., máx.), inferencia, tamaño, fronteras, elegido |
| `lab03_cv_folds.csv` | Resultado de cada fold de la CV temporal |
| `lab03_decision.json` | Decisión congelada, regla, cuantificación, sensibilidad, entorno y `prueba_tocada: false` |
| `lab03_test_final.json` | Evaluación única en la prueba |
| `pareto.png`, `tsne_two_seeds.png`, `lab03_matriz_confusion_final.png` | Figuras |
| `lab03_memoria_densa.csv`, `lab03_pca.json`, `lab03_tsne_metricas.json` | Costo de densificar, PCA y estabilidad de t-SNE |
| `lab03_entorno.txt`, `lab03_seleccion_salida.txt`, `lab03_cierre_salida.txt`, `lab03_pytest_output.txt` | Entorno y salidas de las ejecuciones |

## Limitaciones de LAB03

- **La prueba 2025-2026 ya se había visto en LAB02.** No es virgen en sentido estricto; por eso esas cifras no intervinieron en ninguna decisión de LAB03.
- **Los tiempos dependen del equipo**, y con solo 4 folds la desviación estándar es una estimación ruidosa.
- **La regla de decisión es una elección de diseño** (1 desviación estándar, recall prioritario). Con otra regla el resultado cambia, como muestra la tabla de sensibilidad.
- **Los hiperparámetros de Random Forest y boosting no se ajustaron**; la comparación es entre las configuraciones del guion, no entre los mejores posibles de cada familia.
- **Solo hay 4 variables de calendario.** Sin feriados, temporadas ni itinerarios, el techo de todos los modelos es bajo.
- **La energía y el CO2 son estimaciones** con supuestos explícitos (ver arriba).

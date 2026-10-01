# INF-8239 · Unidad 01 — LAB02 y LAB03 sobre datos reales (JAC)

Comparación de un baseline (`DummyClassifier`) contra una SVM RBF para anticipar **días de alta demanda** en los aeropuertos internacionales de República Dominicana, usando el dataset público de la Junta de Aviación Civil (JAC).

> Ejercicio académico. No es un modelo operativo de planificación aeroportuaria.

**Índice.** **LAB02** (práctica 02: baseline vs SVM) ocupa las secciones hasta «Limitaciones y siguientes pasos de LAB02». **LAB03** (ensambles, reducción dimensional, mediciones repetidas y frontera de Pareto) está en la sección [LAB03](#lab03--ensambles-reducción-dimensional-mediciones-repetidas-y-frontera-de-pareto) al final.

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
├── LAB03.py                       LAB03: todo el flujo, fiel al guion (fases: seleccion y cierre)
├── lab03_energia.py               LAB03: estimación de energía y CO2 (CodeCarbon)
├── requirements.txt · requirements-energia.txt · pytest.ini · .gitignore
├── src/inf8239_u01/
│   ├── data.py                    download_csv, ensure_dataset, load_raw, clean_jac
│   ├── features.py                calendario, target alta_demanda, split temporal, train_split
│   ├── models.py                  build_svm (unidad 01), Dummy y SVM con Pipeline, preprocesamiento (disperso o denso)
│   ├── evaluate.py                métricas y comparación con la misma partición
│   ├── lab03_config.py            LAB03: declaración previa (métrica, clase, regla, costos)
│   ├── green.py                   LAB03: Pareto, mediciones repetidas, decisión, guarda de la prueba
│   └── explore.py                 LAB03: memoria densa, PCA y t-SNE (estabilidad)
├── tests/                         63 pruebas (10 archivos)
├── docs/
│   ├── PROCEDENCIA_LICENCIA.md    Ficha de procedencia y licencia (ODbL)
│   ├── DICCIONARIO_DATOS.md       Diccionario de datos
│   └── LAB03_declaracion.md       LAB03: reglas fijadas antes de comparar (y adenda)
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
python -m pytest -v                    # 63 pruebas (LAB02 + LAB03)
jupyter nbconvert --to notebook --execute --inplace 02_svm_jac_pasajeros.ipynb
```

También puede abrirse el notebook en Jupyter o VS Code con el kernel «INF-8239 (.venv)». Para ejecutar solo la parte offline, coloca el CSV en `data/raw/dataset.csv`.

## Pruebas

`python -m pytest -v` → **63 aprobadas** (LAB02: `reports/pytest_output.txt`, 25 pruebas; suite completa con LAB03: `reports/lab03_pytest_output.txt`).

| Archivo | Pruebas | Qué cubre |
|---|---|---|
| `test_data.py` | 9 | Rechazo de URL no HTTP y de datasets vacíos, no redescarga, `-` como cero, texto inválido, armonización 11→7, fechas y meses inválidos, BOM |
| `test_features.py` | 6 | Umbral P75 estricto por aeropuerto-año, ausencia de fugas en X, día de la semana, partición temporal sin solape |
| `test_models.py` | 8 | Pipeline con `prep` + `model`, categoría no vista, misma partición para Dummy y SVM, conteo de FP/FN, CV temporal, unidad 01 |
| `test_dataset_real.py` | 2 | Forma, aeropuertos, fechas y balance del CSV real (se omiten si no está descargado) |
| `test_green.py` | 12 | Las dos pruebas de Pareto del guion, empates, varios costos, mediciones repetidas, regla de decisión y su cuantificación, guarda de la prueba |
| `test_explore.py` | 6 | Memoria densa vs dispersa, PCA, muestra de entrenamiento, estabilidad de t-SNE, efecto de la semilla y pureza por etiquetas adicionales |
| `test_protocolo_lab03.py` | 5 | `LAB03.py` no menciona la prueba antes del cierre, misma partición que LAB02, reglas declaradas, documento de declaración |
| `test_lab03_script.py` | 4 | `LAB03.py` sigue los pasos del guion en orden, conserva sus modelos y `probability=True`, cada `CAMBIO`/`ADICIÓN` está en el README, y una corrida de humo de extremo a extremo que comprueba que la prueba solo se evalúa una vez |
| `test_lab03_evidencia.py` | 7 | CSV con 3 repeticiones y mediana, Pareto recalculable, decisión reproducible desde el CSV, figuras, t-SNE, PCA, cierre único |
| `test_lab03_energia.py` | 4 | Ayuda del script, alcance declarado, solo entrenamiento, corrida de humo con CodeCarbon |

## Historial de commits

Commits pequeños y con mensaje descriptivo (`git log --oneline`). **LAB02:** estructura → datos → pruebas → target y partición → modelos y evaluación → `pract02.py` con evidencias → documentación de datos → notebook → README → licencia. **LAB03:** declaración previa de reglas → herramientas (`green`, `explore`) → pruebas → adenda para la versión fiel al guion → `LAB03.py` → **evidencia y decisión congeladas** → **evaluación única de la prueba** → energía → README.

El orden es deliberado: la declaración precede a todo el código de modelos, y la decisión queda versionada **antes** del commit que toca la prueba. Antes de `LAB03.py` hubo una primera versión modular (scripts `lab03_seleccion.py` y `lab03_cierre.py`); el commit «retirar la versión modular» la reemplaza y se conserva en el historial.

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

Amplía el problema de LAB02 (¿un día será de **alta demanda** en un aeropuerto?) comparando **7 configuraciones más un baseline** (SVM, Random Forest y boosting, con y sin PCA), repitiendo las mediciones de costo y decidiendo con la **frontera de Pareto**. Se reutilizan sin cambios el dataset, el target, las exclusiones y la partición temporal de LAB02. Todo el flujo está en un solo archivo, **`LAB03.py`**, que sigue los pasos 2 a 10 del guion entregado; lo que se separó del guion está marcado en el código y explicado más abajo.

> Ejercicio académico. Los tiempos dependen del equipo donde se ejecutan (ver «Cómo leer los tiempos»).

## Reglas declaradas antes de comparar

Quedaron versionadas en `docs/LAB03_declaracion.md` y `src/inf8239_u01/lab03_config.py`, en un commit anterior a todo el código de modelos. La adenda sobre la versión fiel al guion se versionó antes de ejecutarla.

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
| `svm_c10` | SVM | 0.695 ± 0.064 | 0.492 | 46.78 | 1032 | 1579 | ✔ | ✔ |
| `svm_c1` **← elegido** | SVM | 0.694 ± 0.071 | 0.449 | 37.21 | 1079 | 1699 | ✔ | ✔ |
| `boost` | Boosting | 0.674 ± 0.067 | 0.418 | 0.38 | 13 | 363 | ✔ |  |
| `svm_pca` | SVM + PCA | 0.674 ± 0.053 | 0.423 | 30.56 | 923 | 1430 |  |  |
| `logistic` | lineal | 0.534 ± 0.037 | 0.172 | 0.04 | 4 | 5 | ✔ |  |
| `rf_300` | Random Forest | 0.494 ± 0.044 | 0.083 | 1.13 | 31 | 6877 |  |  |
| `rf_100` | Random Forest | 0.493 ± 0.041 | 0.081 | 0.33 | 13 | 2286 |  |  |
| `dummy` | referencia | 0.430 ± 0.007 | 0.000 | 0.02 | 5 | 4 | ✔ |  |

Frontera de Pareto: F1 macro contra tiempo de ajuste; fuera de ella quedan `rf_100`, `rf_300`, `svm_pca`. En `reports/green_ai_results.csv` también están las fronteras contra inferencia, contra tamaño y contra los tres costos a la vez. Figura: `reports/pareto.png`.

- **Las dos SVM de arriba están empatadas** (F1 macro 0.694 y 0.695, con una desviación entre folds de ~0.06), pero son las más lentas: 37 y 47 s de ajuste y cerca de 1 s de inferencia por 3 000 filas.
- **Boosting (`boost`) es el punto eficiente:** frente a `svm_c10`, F1 macro -2.1 puntos y recall de la clase 1 -7.4 puntos, con un ajuste 123× más rápido, inferencia 80× más rápida y un modelo 4.3× más pequeño.
- **`svm_pca` queda dominada por `boost`:** rinde lo mismo (0.674 frente a 0.674) y tarda 80 veces más.
- **Random Forest queda en 0.49 de F1 macro** (recall de la clase 1 ≈ 0.08) con los hiperparámetros del guion (profundidad 8, 100/300 árboles, sin ajustar): casi nunca predice la clase 1. No investigué la causa; una hipótesis es que la profundidad limitada lo deja sub-ajustado. No es una conclusión sobre Random Forest en general.
- **La regresión logística** queda en 0.53: una hipótesis es que un modelo aditivo no captura las interacciones (aeropuerto × mes × día de la semana).
- **`probability=True` encarece las SVM.** Respecto de la primera versión modular de LAB03 (sin `probability`, ver el historial), el ajuste de `svm_c1` pasó de 5.1 a 37.2 s (7.4×) y el de `svm_c10` de 9.2 a 46.8 s (5.1×). El F1 y el recall no cambian: `predict` no usa las probabilidades.

## Decisión cuantificada

Aplicando la regla declarada, **se elige `svm_c1`**. Los umbrales de elegibilidad fueron F1 macro ≥ 0.631 y recall de la clase 1 ≥ 0.439; quedaron elegibles `svm_c1` y `svm_c10`; entre ellos, `svm_c1` ajusta en menos tiempo. Frente al modelo de mayor F1 macro (`svm_c10`):

| Lo que cambia al elegir `svm_c1` | Valor |
|---|---|
| F1 macro | -0.08 puntos (0.6944 frente a 0.6951) |
| Recall de la clase 1 | -4.28 puntos |
| Tiempo de ajuste | ×0.80 (20 % menos) |
| Inferencia | ×1.05 |
| Tamaño serializado | ×1.08 |

**Lectura honesta.** La diferencia de F1 macro (0.08 puntos) es unas 84 veces menor que la desviación entre folds (0.064): en rendimiento global son indistinguibles. Lo que sí cambia es el recall de la clase prioritaria (-4.3 puntos, dentro de una desviación) y un ahorro de 20 % en tiempo de ajuste. La inferencia (×1.05, dentro de la dispersión) y el tamaño (×1.08) no mejoran. La regla decide con un solo costo (el tiempo de ajuste), y aquí eso inclina la balanza hacia un modelo con menos recall; si la prioridad fuese detectar días altos, `svm_c10` sería razonable. `boost` quedó fuera por el recall de la clase 1: 0.418, a 0.021 del umbral.

Sensibilidad (**informativa, no cambia la decisión**): qué modelo saldría con otra tolerancia `k` en desviaciones estándar.

| k | 0.5 | 1.0 (declarada) | 1.5 | 2.0 |
|---|---|---|---|---|
| Modelo | `svm_c10` | `svm_c1` | `boost` | `boost` |

Con una tolerancia de 1.5 desviaciones o más, `boost` entraría como elegible y ganaría por costo. Esa sería una decisión distinta, con otra regla, y habría que declararla antes de comparar.

## Evaluación final (prueba 2025-2026, una sola vez)

Se ajustó `svm_c1` con el entrenamiento y se evaluó **una sola vez** en la prueba (3 627 días). `LAB03.py --fase cierre` lee la decisión congelada y se niega a repetirse. Los otros dos modelos aparecen solo como contexto y no podían cambiar la decisión; `svm_c10` no se evaluó en la prueba.

| Modelo | F1 macro | Recall clase 1 | Precisión clase 1 | ROC-AUC | Falsos positivos | Falsos negativos |
|---|---|---|---|---|---|---|
| **`svm_c1` (elegido)** | 0.761 | 0.543 | 0.737 | 0.866 | 174 | 410 |
| `svm_lab02` (SVM de LAB02, C=10 y gamma=0.1; contexto) | 0.759 | 0.556 | 0.714 | 0.861 | 200 | 399 |
| `dummy` (baseline, contexto) | 0.429 | 0.000 | 0.000 | 0.500 | 0 | 898 |

- F1 macro en la prueba: **0.761**; el CV estimaba 0.694 ± 0.071. La prueba queda +6.7 puntos por encima, dentro de una desviación del CV (hipótesis, no medida: los primeros folds entrenan con pocos datos y cubren 2020-2021).
- El falso negativo sigue siendo el error caro: **410 días de alta demanda no anticipados** frente a 174 falsas alarmas.
- **Consistencia con LAB02:** la SVM de LAB02 recalculada aquí da 0.7595, igual que en LAB02 (0.7595): la partición y el preprocesamiento no cambiaron.
- **Las configuraciones nuevas (Random Forest, boosting, logística y SVM + PCA) no superan en CV a las SVM de LAB02**; lo que aporta LAB03 es saber a qué costo se logra y qué alternativas más baratas existen.

## PCA: ¿aplica aquí?

- **Densidad:** `HistGradientBoosting` y PCA exigen matriz densa. La matriz codificada es de 16 344 × 27: **3.53 MB densa frente a 0.85 MB dispersa** (`reports/lab03_memoria_densa.csv`). Densificar es seguro y se midió antes de hacerlo.
- **Bloque numérico:** solo tiene 1 columna (`dia`). PCA sobre una sola columna no reduce nada, por lo que se aplica a la matriz codificada completa, dentro del `Pipeline` (`prep` → `StandardScaler` → `PCA` → SVM).
- **Resultado:** 23 de 27 componentes retienen el 96.4 % de la varianza (reducción de 15 % de las columnas). Las columnas *one-hot* casi no están correlacionadas, así que hay poco que comprimir.
- **Efecto:** `svm_pca` obtiene F1 macro 0.674 frente a 0.694 de `svm_c1` (-2.1 puntos) y recall 0.423 frente a 0.449, con un ajuste de 30.6 s frente a 37.2 s (-18 %). PCA no compensa aquí.

## t-SNE con dos semillas

`reports/tsne_two_seeds.png`: muestra de 1 000 filas **de entrenamiento**, matriz codificada y estandarizada (como en el guion), semillas 42 y 7, coloreada por alta demanda, aeropuerto y mes. Métricas en `reports/lab03_tsne_metricas.json`; «pureza» es la fracción de los 10 vecinos más cercanos de cada punto que comparte la etiqueta.

| | Mapa semilla 42 | Mapa semilla 7 | Azar |
|---|---|---|---|
| Pureza por **mes** | 1.00 | 1.00 | 0.08 |
| Pureza por día de la semana | 0.63 | 0.63 | 0.14 |
| Pureza por aeropuerto | 0.42 | 0.44 | 0.15 |
| Pureza por alta demanda | 0.67 | 0.67 | 0.62 |

Entre los dos mapas comparten el 72 % de los vecinos y la disparidad de Procrustes es 0.14 (0 = mapas idénticos).

- **Permanece:** los 12 grupos son los 12 meses (pureza 1.00 en ambos mapas), y dentro de cada grupo se agrupan además el día de la semana y, en menor medida, el aeropuerto. Una causa probable es que, al estandarizar las columnas *one-hot*, las 12 del mes pesan mucho en la distancia; no lo probé con otro escalado.
- **Cambia:** la posición y la orientación de los grupos y cerca de un 28 % de los vecinos de cada punto; las distancias entre grupos no son interpretables.
- **Casi no se ve separación por alta demanda:** la pureza queda apenas 5 puntos sobre el azar. Con solo calendario, los días altos no forman grupos propios, y aun así la SVM llega a F1 macro ≈ 0.7: un mapa 2D no conserva el límite que aprende el modelo (hipótesis, no medida).
- **Cautela:** la separación visual no demuestra causalidad ni valida un clasificador; t-SNE es exploratorio y no se puede aplicar a datos nuevos.
- **Nota sobre `init="pca"`:** con esa inicialización (la del guion) las dos semillas dan mapas **idénticos** (disparidad 3e-31, vecinos compartidos 100 %). Por eso el análisis usa inicialización aleatoria, con la que la semilla sí importa.

## Cambios respecto al guion `LAB03.py`

El guion se siguió paso a paso (2 a 10) y con los mismos nombres de modelos, columnas y archivos. **Se conservó:** `probability=True` en las SVM, los hiperparámetros de todos los modelos, las 3 repeticiones con mediana, el tamaño con `joblib`, `pareto_flags` tal cual (definida dentro del script) y las dos pruebas de Pareto en `tests/test_green.py`. Cada desviación está marcada en `LAB03.py` con la misma etiqueta que aquí.

**Cambios necesarios** (sin ellos el guion falla o rompe una regla declarada):

| Etiqueta | Guion | En `LAB03.py` | Motivo |
|---|---|---|---|
| `CAMBIO 1` | Da por hecho `Xtr`, `Xte` de un `train_test_split` aleatorio | Partición temporal de LAB02; en la selección solo existe el entrenamiento (`train_split`) | «No cambies la partición» y «prueba sin tocar hasta el cierre» |
| `CAMBIO 2` | `preprocess` con one-hot disperso | `OneHotEncoder(sparse_output=False)` (`build_preprocess(dense=True)`) | `HistGradientBoosting` y PCA exigen matriz densa (lo dice el propio guion); costo medido: 3.53 MB |
| `CAMBIO 3` | `RandomForest(n_jobs=-1)` | `N_JOBS_RF = 1` y mediciones con `threadpool_limits(1)` | La regla declarada es 1 hilo: costo comparable entre modelos y equipos. Para volver al guion basta `N_JOBS_RF = -1` |
| `CAMBIO 4` | PCA sobre `Xtr`, que tiene texto (`aeropuerto`) | `("prep", preprocess)` al inicio del `Pipeline` de PCA | PCA exige números; el bloque numérico es una columna y no se puede reducir solo |
| `CAMBIO 5` | F1 y recall calculados con la prueba; inferencia medida una vez sobre la prueba | F1 y recall por validación cruzada temporal sobre el entrenamiento (`recall_alta` además de `recall_macro`); inferencia: 3 repeticiones, mediana, sobre 3 000 filas de entrenamiento | La prueba no se toca hasta el cierre; «tres repeticiones y mediana» |
| `CAMBIO 6` | t-SNE sobre las columnas numéricas de toda `X` con `init="pca"` | Matriz codificada de una muestra de entrenamiento y `init="random"` | `X` incluye la prueba y las numéricas son 1 columna; con `init="pca"` las dos semillas dan el mismo mapa |

**Adiciones** (no estaban en el guion; no alteran sus pasos):

| Etiqueta | Qué se añadió | Motivo |
|---|---|---|
| `ADICIÓN 1` | Baseline `dummy` en el catálogo | Referencia para leer los F1; misma partición, como en LAB02 |
| `ADICIÓN 2` | Métricas de estabilidad del t-SNE (vecinos compartidos, pureza por aeropuerto, mes, día de la semana y target, Procrustes) | El guion pide describir qué permanece y qué cambia: aquí se mide |
| `ADICIÓN 3` | La figura de t-SNE tiene filas por target, aeropuerto y mes | Permite ver qué organiza el mapa |
| `ADICIÓN 4` | Fronteras de Pareto contra inferencia, tamaño y los tres costos | Los resultados esperados piden tamaño e inferencia |
| `ADICIÓN 5` | Regla de decisión declarada, cuantificación, sensibilidad a la tolerancia y congelación en `reports/lab03_decision.json` | El guion solo dibuja; se pidió «decisión cuantificada» |
| `ADICIÓN 6` | Paso 11 de cierre: evaluación única de la prueba, con guarda contra repetirla | «Conserva la prueba sin tocar hasta el cierre» |

También: el paso 8 vive en `tests/test_green.py`, y el paso 10 es un script aparte, `lab03_energia.py`, que carga los modelos de `reports/models` que deja `LAB03.py`. `LAB03.py` importa de `src/inf8239_u01` el dataset, el preprocesamiento y las funciones ya probadas (decisión, estabilidad de t-SNE, memoria), igual que el guion importa `pareto_flags` en su paso 8. La versión modular anterior de LAB03 queda solo en el historial de git.

## Ejecución de LAB03

Desde la raíz del proyecto, con el entorno activado y en este orden:

```powershell
python LAB03.py --fase seleccion   # ~9 min en el equipo de la evidencia (SVM con probability=True). NO toca la prueba
python LAB03.py --fase cierre      # evaluación única en la prueba; se niega a repetirse
python lab03_energia.py            # opcional: energía y CO2 (requiere requirements-energia.txt)
python -m pytest -v                # 63 pruebas
```

`python LAB03.py` sin argumentos ejecuta las dos fases seguidas. Separarlas permite versionar la decisión **antes** de tocar la prueba, que es lo que hace el historial de este repositorio.

### Cómo leer los tiempos

Los tiempos y tamaños incluidos en `reports/` se generaron en: Linux-6.18.44-fc-v50-x86_64-with-glibc2.39, Python 3.11.15, scikit-learn 1.8.0, **2 núcleos lógicos**, 1 hilo en las mediciones. En tu equipo los valores serán distintos (y entre corridas del mismo equipo varían unos pocos por ciento).

- Las **F1 y los recalls** no dependen del equipo; los **tiempos** sí. El orden entre familias (logística < boosting ≈ Random Forest < SVM) es estable; las diferencias pequeñas, no.
- Para rehacer **todo** el protocolo en tu equipo: `python LAB03.py --fase seleccion --nueva-corrida` y luego `python LAB03.py --fase cierre --nueva-corrida`. Las banderas sobrescriben la evaluación previa y solo deben usarse al rehacer todo.
- Si tu ejecución elige otro modelo, vale el que salga de la regla declarada con tus mediciones.
- `reports/models/*.joblib` no se versionan (pesan ~14 MB); sus tamaños están en `green_ai_results.csv`.

## Energía y emisiones de CO2 (`lab03_energia.py`)

**El tiempo no es consumo energético.** Este script estima energía y CO2eq con CodeCarbon y declara su alcance:

| Elemento | Detalle |
|---|---|
| **Herramienta** | CodeCarbon ≥ 3.0, modo offline (sin servicios externos) |
| **Región** | `--pais DOM` (República Dominicana) por defecto; el factor de emisión es el **promedio anual de la matriz eléctrica** del país que trae CodeCarbon, no el de la hora ni el de tu proveedor |
| **Carga** | `--repeticiones` ciclos de ajuste + inferencia por modelo dentro del medidor; se reporta **por ciclo** |
| **Cobertura** | CPU, RAM y GPU (si existe) |
| **Supuestos** | La potencia de CPU viene de contadores de hardware si están disponibles; si no, se estima con la carga y el TDP. La de RAM es una **potencia estimada por capacidad, no medida** |
| **No cubre** | Fuente de poder, refrigeración, pantalla, red, fabricación del equipo |

Salidas: `reports/lab03_energia.csv` y `reports/lab03_energia_meta.json` (herramienta, versión, región, equipo, supuestos, cobertura y advertencia). **No se incluye una corrida de energía en `reports/`**: en un contenedor compartido no representa tu equipo.

**Advertencia de interpretación.** Como la potencia de la RAM es una constante estimada, la energía resulta casi proporcional al tiempo: **no añade información que el tiempo de ajuste no dé**. Sirve para comparar modelos entre sí en el mismo equipo, no como consumo eléctrico exacto ni como huella de carbono certificada. Las corridas de pocos segundos quedan por debajo de la resolución del medidor: sube `--repeticiones`.

## Evidencia de LAB03 en `reports/`

| Archivo | Contenido |
|---|---|
| `green_ai_results.csv` | Tabla por modelo: F1 macro CV ± desv., recall, ajuste (mediana, mín., máx.), inferencia, tamaño, fronteras, elegido |
| `lab03_cv_folds.csv` | Resultado de cada fold de la CV temporal |
| `lab03_decision.json` | Decisión congelada, regla, cuantificación, sensibilidad, entorno y `prueba_tocada: false` |
| `lab03_test_final.json` | Evaluación única en la prueba |
| `pareto.png`, `tsne_two_seeds.png`, `lab03_matriz_confusion_final.png` | Figuras |
| `lab03_memoria_densa.csv`, `lab03_pca.json`, `lab03_tsne_metricas.json` | Costo de densificar, PCA y estabilidad de t-SNE |
| `lab03_entorno.txt`, `lab03_seleccion_salida.txt`, `lab03_pytest_output.txt` | Entorno y salidas de la selección y de pytest |

## Limitaciones de LAB03

- **La prueba 2025-2026 ya se había visto en LAB02.** No es virgen en sentido estricto; por eso esas cifras no intervinieron en ninguna decisión de LAB03.
- **Una versión previa ya se había ejecutado.** La primera versión modular de LAB03 se corrió y sus resultados se conocían antes de escribir la versión fiel al guion. Las reglas no se movieron; cambió el conjunto de modelos y el motor de las SVM (`docs/LAB03_declaracion.md`, adenda).
- **Los tiempos dependen del equipo**, y con solo 4 folds la desviación estándar es una estimación ruidosa.
- **La regla de decisión es una elección de diseño** (1 desviación estándar, recall prioritario, un solo costo principal). Con otra regla el resultado cambia, como muestra la tabla de sensibilidad.
- **Los hiperparámetros de Random Forest y boosting no se ajustaron**; la comparación es entre las configuraciones del guion, no entre los mejores posibles de cada familia.
- **Solo hay 4 variables de calendario.** Sin feriados, temporadas ni itinerarios, el techo de todos los modelos es bajo.
- **Los mapas t-SNE dependen del escalado** de la matriz codificada; con otro escalado los grupos podrían organizarse de otra forma.
- **La energía y el CO2 son estimaciones** con supuestos explícitos (ver arriba).

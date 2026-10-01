# Evidencia Green AI

## Entorno de medición (hardware log)

| Campo | Valor |
|---|---|
| Sistema operativo | Windows-11-10.0.22621-SP0 |
| CPU | Intel64 Family 6 Model 165 Stepping 5, GenuineIntel (núcleos físicos: 6, lógicos: 12) |
| RAM | 15.94 GiB |
| GPU | no utilizada |
| Python | 3.14.7 |
| Versiones | numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, joblib 1.6.0, psutil 7.2.2 |
| Fecha (UTC) | 2026-09-25T22:16:25+00:00 |

> Los tiempos y la memoria dependen de este equipo. **Vuelva a ejecutar `python -m src.green_eval` en su
> máquina** para obtener el hardware log propio; el orden relativo entre modelos suele mantenerse, los
> valores absolutos no.

## Protocolo

- Partición hold-out 80/20 estratificada (`random_state=42`); CV estratificada de 5 pliegues
  y `GridSearchCV` sobre el Pipeline completo, solo con el 80 % de entrenamiento (1600 filas).
- **Tiempo de ajuste:** mediana de 7 repeticiones con `perf_counter` (Pipeline completo, hiperparámetros ya elegidos).
- **Tiempo de inferencia:** mediana de 30 repeticiones prediciendo las 400 filas del hold-out;
  se expresa en ms por 1000 filas.
- **Memoria:** incremento del pico de RSS durante un ajuste completo, medido en un **proceso nuevo** por
  repetición (`src/mem_probe.py`, calentamiento con 200 filas) y mediana de 5 procesos. Medir dentro
  del mismo proceso subestima, porque el asignador reutiliza páginas liberadas. Sigue siendo un **estimador**
  (incluye el arranque de estructuras del Pipeline). El pico de `tracemalloc` (solo Python/NumPy) queda en
  `reports/tables/green_ai.csv`.
- **Tamaño serializado:** `joblib.dump` del Pipeline completo (escalador + PCA + modelo).
- Un solo hilo (`threadpool_limits(1)`) para que los modelos sean comparables. El tiempo de la búsqueda de
  hiperparámetros (columna `busqueda_cv_s` del CSV) se reporta aparte y no forma parte del costo de despliegue.

## Resultados

| modelo | F1 CV | +/- CV | F1 hold-out | ajuste (s, mediana) | inferencia (ms/1000 filas) | memoria pico ajuste (MiB) | tamaño serializado (KiB) | vars. | detalle |
|---|---|---|---|---|---|---|---|---|---|
| DummyClassifier | 0.1 | 0 | 0.1 | 0.002887 | 2.361 | 0.02734 | 1.97 | 20 |  |
| LogisticRegression | 0.9632 | 0.005364 | 0.9825 | 0.01735 | 2.722 | 0.01562 | 2.853 | 20 |  |
| LinearSVC | 0.8633 | 0.0194 | 0.8336 | 0.01001 | 2.718 | 0.04297 | 2.724 | 20 |  |
| SVC lineal (OvO) | 0.965 | 0.006754 | 0.975 | 0.07927 | 6.35 | 0.1445 | 42.47 | 20 | 190 vectores de soporte |
| SVC RBF | 0.9537 | 0.00576 | 0.97 | 0.0523 | 27.01 | 0.1641 | 68.75 | 20 | 317 vectores de soporte |
| PCA + SVC RBF | 0.9592 | 0.00626 | 0.9573 | 0.04912 | 26.69 | 0.1484 | 67.3 | 18 | 314 vectores de soporte |
| RandomForest | 0.8737 | 0.03217 | 0.8848 | 0.7617 | 78.69 | 11.46 | 1.373e+04 | 20 | 145102 nodos |
| HistGradientBoosting | 0.9013 | 0.02008 | 0.9277 | 0.7031 | 40.23 | 1.781 | 1220 | 20 | 200 iteraciones de boosting |

`F1 CV` = media +/- desviación en los 5 pliegues de entrenamiento (criterio de selección);
`F1 hold-out` = confirmación final sobre 400 filas nunca vistas.

Hiperparámetros elegidos por CV: `reports/tables/mejores_hiperparametros.json`.

Hiperparámetros cuyo valor óptimo cae en un **extremo de la grilla** (la grilla podría ser corta; ver
columna `params_en_borde_grilla` del CSV): svm_rbf: C, gamma; svm_rbf_pca: C, gamma; hgb: max_iter

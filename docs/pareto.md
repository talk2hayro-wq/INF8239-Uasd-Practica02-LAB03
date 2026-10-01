# Análisis de Pareto

**Rendimiento:** F1 macro medio de la validación cruzada de 5 pliegues (solo entrenamiento).
**Costos:** tiempo de ajuste, latencia de inferencia, memoria pico y tamaño serializado (`docs/green_ai.md`).
Dominancia estricta: una configuración se elimina solo si otra es igual o mejor en F1 y en todos los costos, y
estrictamente mejor en al menos uno. Figura: `reports/figures/06_pareto.png`.

## Frontera por dimensión y global

| modelo | F1 CV | F1 hold-out | frontera tiempo de ajuste | frontera inferencia | frontera memoria pico de ajuste | frontera tamaño serializado | FRONTERA GLOBAL (CV) | FRONTERA GLOBAL (hold-out) |
|---|---|---|---|---|---|---|---|---|
| DummyClassifier | 0.1000 | 0.1000 | sí | sí | no | sí | SÍ | SÍ |
| LogisticRegression | 0.9632 | 0.9825 | sí | sí | sí | sí | SÍ | SÍ |
| LinearSVC | 0.8633 | 0.8336 | sí | sí | no | sí | SÍ | SÍ |
| SVC lineal (OvO) | 0.9650 | 0.9750 | sí | sí | sí | sí | SÍ | no |
| SVC RBF | 0.9537 | 0.9700 | no | no | no | no | no | no |
| PCA + SVC RBF | 0.9592 | 0.9573 | no | no | no | no | no | no |
| RandomForest | 0.8737 | 0.8848 | no | no | no | no | no | no |
| HistGradientBoosting | 0.9013 | 0.9277 | no | no | no | no | no | no |

- **Frontera global (CV):** DummyClassifier, LogisticRegression, LinearSVC, SVC lineal (OvO).
- **Dominados globalmente:** SVC RBF, PCA + SVC RBF, RandomForest, HistGradientBoosting.
- `DummyClassifier` está en la frontera solo por ser el punto de menor costo; su F1 (0.10) lo hace inutilizable.
  Marca el extremo trivial de la frontera, no una opción.
- La columna *hold-out* repite el análisis con el F1 de las 400 filas reservadas, como prueba de sensibilidad.

## Decisión razonada (tolerancia de 0.5 puntos de F1)

Modelos útiles con F1 CV dentro de 0.5 puntos del mejor
(SVC lineal (OvO), 0.9650), ordenados por costo relativo (1.0 = el mínimo de cada dimensión
entre estos candidatos):

| modelo | F1 CV | dF1 vs mejor (pts) | ajuste (x del min.) | inferencia (x) | memoria (x) | tamaño (x) | costo relativo (media geom.) |
|---|---|---|---|---|---|---|---|
| LogisticRegression | 0.963 | -0.181 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SVC lineal (OvO) | 0.965 | 0.000 | 4.568 | 2.333 | 9.250 | 14.888 | 6.190 |

**Selección: LogisticRegression.** Frente al modelo de mayor F1 CV (SVC lineal (OvO)), cede
0.18 puntos de F1 CV (desviación entre pliegues: 0.54 puntos) y reduce el
tiempo de ajuste en 78 %, la latencia en 57 %, la memoria en
89 % y el tamaño serializado en 93 %.

## Cautelas

- La frontera se calcula sobre **medianas**; diferencias de tiempo pequeñas entre modelos de costo similar
  no son robustas frente al ruido del hardware (ver columnas IQR en `reports/tables/green_ai.csv`).
- Un modelo puede entrar a la frontera por una ventaja minima en un solo costo (por ejemplo, un ajuste algunos
  milisegundos más rápido) aunque su F1 sea muy inferior: la frontera **no es por sí sola un criterio de
  selección**. Por eso se aplica además la regla de tolerancia de F1 sobre los modelos útiles.
- Las diferencias de F1 entre los modelos lineales y el SVM RBF (< 1.5 puntos) son comparables a la
  desviación entre pliegues; no deben leerse como superioridad estadística.
- Los resultados dependen del hardware de medición y de un dataset posiblemente curado (ver ficha técnica).

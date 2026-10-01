# LAB03 · Declaración previa (antes de comparar modelos)

Este documento fija las reglas **antes** de ejecutar cualquier comparación. Se versionó en un commit propio, anterior a todo el código de modelos y a todos los resultados (ver `git log`).

## 1. Qué se reutiliza de LAB02 (sin cambios)

| Elemento | Valor |
|---|---|
| Dataset | JAC, movimiento de pasajeros por aeropuerto (`docs/PROCEDENCIA_LICENCIA.md`, ODbL) |
| Unidad de análisis | Un aeropuerto en un día · 19 971 filas |
| Target | `alta_demanda` (total > P75 del aeropuerto y año) |
| Variables | `aeropuerto`, `mes`, `dia_semana`, `dia` |
| Exclusiones | `entrada`, `salida`, `total`, `fecha`, `anio`, `umbral_p75` |
| Partición | Temporal única: entrenamiento 2018-2024 (16 344 filas), prueba 2025-2026 (3 627 filas) |

La partición **no se toca** para favorecer a ningún modelo: se reutiliza `features.temporal_split`, la misma función de LAB02.

## 2. Métrica principal y clase prioritaria

| Declaración | Valor | Por qué |
|---|---|---|
| **Métrica principal** | **F1 macro**, promedio de la validación cruzada temporal (4 folds, solo entrenamiento) | Continuidad con LAB02; no se deja engañar por el 75 % de días normales |
| **Clase prioritaria** | **Alta demanda (clase 1)** | El falso negativo (un día alto que no se anticipa) es el error caro |
| **Métrica de la clase prioritaria** | **Recall de la clase 1** (`recall_alta`) | Mide cuántos días altos se detectan |

## 3. Costos medidos

| Costo | Cómo se mide |
|---|---|
| **Tiempo de ajuste** (`fit_median_s`, costo principal) | Mediana de 3 ajustes completos sobre todo el entrenamiento, 1 hilo |
| **Inferencia** (`predict_ms`) | Mediana de 3 predicciones sobre un lote de 3 000 filas **de entrenamiento** (solo cronometraje; no se usa la prueba) |
| **Tamaño serializado** (`size_kb`) | Tamaño del `.joblib` del modelo ajustado, sin compresión |

El tiempo es contextual: depende del equipo, y **no** se presenta como consumo energético. La energía y el CO2 se estiman aparte, con CodeCarbon, y con sus supuestos (ver README).

## 4. Regla de decisión (fijada antes de ver resultados)

1. Se construye la tabla con F1 macro y recall de la clase 1 **por validación cruzada temporal** y los tres costos.
2. Se calcula la **frontera de Pareto** de F1 macro contra el tiempo de ajuste. Como complemento se calculan fronteras contra inferencia, tamaño y los tres costos a la vez.
3. **Elegibles:** modelos cuyo F1 macro esté a menos de **1 desviación estándar** (entre folds, ddof=1) del mejor F1 macro, **y** cuyo recall de la clase 1 esté a menos de 1 desviación estándar del mejor recall.
4. **Decisión:** entre los elegibles que estén en la frontera de Pareto (F1 macro contra tiempo de ajuste), se elige el de **menor tiempo de ajuste**; desempate por menor tamaño serializado.
5. **Se cuantifica** lo que se gana y lo que se pierde frente al modelo de mayor F1 macro: puntos de F1 y de recall, y razones de tiempo de ajuste, inferencia y tamaño.

La regla no se modifica después de ver resultados.

## 5. Protocolo de la prueba (2025-2026)

- La selección (`lab03_seleccion.py`) **no carga la prueba**: usa `features.train_split`, que devuelve solo el entrenamiento. Hay una prueba automática que lo vigila.
- La decisión queda congelada en `reports/lab03_decision.json` y se versiona **antes** de tocar la prueba.
- El cierre (`lab03_cierre.py`) evalúa **una sola vez** el modelo elegido y se niega a repetirse si ya existe `reports/lab03_test_final.json`.
- En el cierre se muestran, solo como contexto y sin poder cambiar la decisión, el baseline y el SVM elegido en LAB02.

## 6. Limitaciones declaradas de antemano

- **La prueba 2025-2026 ya se vio en LAB02** (Dummy, SVM base y SVM del grid). No es virgen en sentido estricto. Por eso esas cifras **no intervienen en ninguna decisión de LAB03**: los modelos se comparan solo con la validación cruzada.
- Los tiempos y tamaños dependen del equipo y de la versión de las librerías. La decisión se calcula con los costos medidos en cada ejecución.
- Con 4 folds, la desviación estándar es una estimación ruidosa.

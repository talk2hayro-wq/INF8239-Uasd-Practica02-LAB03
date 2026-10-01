"""Frontera de Pareto, mediciones repetidas y decisión cuantificada (LAB03)."""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from . import lab03_config as cfg


# --- Frontera de Pareto -------------------------------------------------------
def pareto_flags(df: pd.DataFrame, score: str = "f1_macro", cost="fit_median_s") -> list:
    """Marca True los modelos que ningún otro domina.

    Un modelo A domina a B si A no es peor en rendimiento ni en NINGÚN costo y
    es estrictamente mejor en al menos uno. `cost` puede ser una columna o una
    lista de columnas (frontera con varios costos a la vez). Dos filas idénticas
    no se dominan entre sí: ambas quedan en la frontera.
    """
    costos = [cost] if isinstance(cost, str) else list(cost)
    columnas = [score] + costos
    if df[columnas].isna().any().any():
        raise ValueError("No se puede calcular la frontera con valores ausentes")
    s = df[score].to_numpy(dtype=float)
    c = df[costos].to_numpy(dtype=float)
    banderas = []
    for i in range(len(df)):
        no_peor = (s >= s[i]) & (c <= c[i]).all(axis=1)
        mejor = (s > s[i]) | (c < c[i]).any(axis=1)
        banderas.append(not bool((no_peor & mejor).any()))
    return banderas


# --- Mediciones repetidas ------------------------------------------------------
def time_calls(fn, repetitions: int = cfg.N_REPETICIONES) -> list:
    """Cronometra `repetitions` llamadas y devuelve los tiempos en segundos."""
    tiempos = []
    for _ in range(repetitions):
        inicio = perf_counter()
        fn()
        tiempos.append(perf_counter() - inicio)
    return tiempos


def measure_model(model, X_train, y_train, X_infer, name: str,
                  models_dir="reports/models",
                  repetitions: int = cfg.N_REPETICIONES,
                  threads: int = cfg.N_HILOS) -> dict:
    """Ajusta `repetitions` veces, cronometra inferencia y mide tamaño serializado.

    - Ajuste: `repetitions` ajustes completos sobre todo X_train (modelo nuevo cada vez).
    - Inferencia: `repetitions` predicciones sobre `X_infer` (solo cronometraje).
    - Se limita a `threads` hilos para que el costo sea comparable entre modelos.
    - Se reporta la MEDIANA (y mínimo y máximo para ver la dispersión).
    """
    with threadpool_limits(limits=threads):
        tiempos_fit = []
        ajustado = None
        for _ in range(repetitions):
            ajustado = clone(model)
            inicio = perf_counter()
            ajustado.fit(X_train, y_train)
            tiempos_fit.append(perf_counter() - inicio)
        tiempos_pred = time_calls(lambda: ajustado.predict(X_infer), repetitions)

    carpeta = Path(models_dir)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{name}.joblib"
    joblib.dump(ajustado, ruta)
    return {
        "fit_median_s": float(np.median(tiempos_fit)),
        "fit_min_s": float(np.min(tiempos_fit)),
        "fit_max_s": float(np.max(tiempos_fit)),
        "predict_ms": float(np.median(tiempos_pred) * 1000),
        "size_kb": float(ruta.stat().st_size / 1024),
        "repeticiones": int(repetitions),
    }


# --- Decisión cuantificada -----------------------------------------------------
def choose_model(table: pd.DataFrame, k: float = cfg.K_DESVIACIONES,
                 score: str = "f1_macro", priority: str = "recall_alta",
                 cost: str = cfg.COSTO_PRINCIPAL, tie: str = cfg.DESEMPATE,
                 frontier: str = "is_pareto") -> dict:
    """Aplica la regla declarada en docs/LAB03_declaracion.md.

    1. Elegibles: F1 macro a <= k desviaciones del mejor F1 macro Y recall de la
       clase prioritaria a <= k desviaciones del mejor recall.
    2. Entre los elegibles en la frontera de Pareto, el de menor `cost`;
       desempate por `tie`.
    3. Cuantifica lo que se gana y se pierde frente al modelo de mayor F1 macro.
    """
    t = table.reset_index(drop=True)
    mejor_f1 = t.loc[t[score].idxmax()]
    mejor_rec = t.loc[t[priority].idxmax()]
    umbral_f1 = mejor_f1[score] - k * mejor_f1[f"{score}_std"]
    umbral_rec = mejor_rec[priority] - k * mejor_rec[f"{priority}_std"]

    elegible = (t[score] >= umbral_f1) & (t[priority] >= umbral_rec)
    candidatos = t[elegible & t[frontier]]
    nota = ""
    if candidatos.empty:
        candidatos = t[elegible]
        nota = "Ningún elegible estaba en la frontera; se eligió entre los elegibles."
    if candidatos.empty:
        candidatos = t.loc[[mejor_f1.name]]
        nota = "Nadie cumplió la regla; se usa el de mayor F1 macro."
    elegido = candidatos.sort_values([cost, tie]).iloc[0]

    def razon(columna):
        return float(elegido[columna] / mejor_f1[columna]) if mejor_f1[columna] else float("nan")

    return {
        "regla": {
            "k_desviaciones": float(k),
            "umbral_f1_macro": float(umbral_f1),
            "umbral_recall_alta": float(umbral_rec),
            "costo_principal": cost,
            "desempate": tie,
        },
        "modelo_elegido": str(elegido["model"]),
        "modelo_referencia_mayor_f1": str(mejor_f1["model"]),
        "elegibles": [str(m) for m in t.loc[elegible, "model"]],
        "candidatos_en_frontera": [str(m) for m in candidatos["model"]],
        "cuantificacion": {
            "f1_macro_elegido": float(elegido[score]),
            "f1_macro_referencia": float(mejor_f1[score]),
            "delta_f1_macro_puntos": float((elegido[score] - mejor_f1[score]) * 100),
            "delta_recall_alta_puntos": float((elegido[priority] - mejor_f1[priority]) * 100),
            "razon_tiempo_ajuste": razon("fit_median_s"),
            "razon_inferencia": razon("predict_ms"),
            "razon_tamano": razon("size_kb"),
        },
        "nota": nota,
    }


# --- Protocolo de la prueba ----------------------------------------------------
def assert_test_untouched(path=cfg.RUTA_PRUEBA_FINAL) -> None:
    """La prueba se evalúa UNA sola vez: si ya hay resultado final, se detiene."""
    if Path(path).exists():
        raise RuntimeError(
            f"La prueba ya se evaluó ({path}). Una segunda evaluación invalidaría la "
            "regla de no tocarla. Si vas a rehacer TODO el protocolo desde cero "
            "(p. ej. en otro equipo), usa --nueva-corrida de forma consciente."
        )


# --- Entorno ---------------------------------------------------------------------
def environment_info() -> dict:
    """Datos para reproducir y contextualizar los tiempos."""
    return {
        "python": sys.version.replace("\n", " "),
        "sistema": platform.platform(),
        "procesador": platform.processor() or "no informado",
        "nucleos_logicos": os.cpu_count(),
        "hilos_usados_en_mediciones": cfg.N_HILOS,
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }

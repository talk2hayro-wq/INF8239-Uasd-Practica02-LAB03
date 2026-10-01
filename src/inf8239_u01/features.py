"""Variables de calendario, target `alta_demanda` y partición temporal.

Unidad de análisis: un aeropuerto en un día (fila aeropuerto-día).
Target: alta_demanda = 1 si entrada + salida del día supera el percentil 75
del mismo aeropuerto en el mismo año; 0 en caso contrario.
"""
from __future__ import annotations

import pandas as pd

TARGET = "alta_demanda"
QUANTILE = 0.75
TEST_START_YEAR = 2025

# Variables disponibles ANTES de que ocurra el día que se quiere predecir.
FEATURES_CAT = ["aeropuerto", "mes", "dia_semana"]
FEATURES_NUM = ["dia"]
FEATURES = FEATURES_CAT + FEATURES_NUM

# Fuga: el target se calcula a partir de estas columnas (resultado del día).
LEAKAGE_COLUMNS = ["entrada", "salida", "total"]
# Excluidas por diseño: fecha/anio solo sirven para partir en el tiempo;
# umbral_p75 es un derivado del target.
EXCLUDED_COLUMNS = LEAKAGE_COLUMNS + ["fecha", "anio", "umbral_p75", TARGET]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega día de la semana (0 = lunes) derivado de la fecha."""
    out = df.copy()
    out["dia_semana"] = out["fecha"].dt.dayofweek
    return out


def build_target(df: pd.DataFrame, quantile: float = QUANTILE) -> pd.DataFrame:
    """Crea `alta_demanda` con umbral por aeropuerto y año (opción B)."""
    if not 0 < quantile < 1:
        raise ValueError("quantile debe estar entre 0 y 1")
    out = df.copy()
    out["umbral_p75"] = out.groupby(["aeropuerto", "anio"])["total"].transform(
        lambda s: s.quantile(quantile)
    )
    out[TARGET] = (out["total"] > out["umbral_p75"]).astype(int)
    return out


def temporal_split(df: pd.DataFrame, test_start_year: int = TEST_START_YEAR):
    """Partición temporal única: entrenamiento < test_start_year <= prueba.

    Devuelve X_train, X_test, y_train, y_test ordenados por fecha. Se usa la
    MISMA partición para el baseline y para la SVM.
    """
    ordenado = df.sort_values(["fecha", "aeropuerto"]).reset_index(drop=True)
    train = ordenado[ordenado["anio"] < test_start_year]
    test = ordenado[ordenado["anio"] >= test_start_year]
    if train.empty or test.empty:
        raise ValueError("La partición temporal dejó un conjunto vacío")
    if train[TARGET].nunique() < 2:
        raise ValueError("El entrenamiento no contiene las dos clases")
    return train[FEATURES], test[FEATURES], train[TARGET], test[TARGET]


def train_split(df: pd.DataFrame, test_start_year: int = TEST_START_YEAR):
    """Devuelve SOLO el entrenamiento (X_train, y_train), con la misma partición
    temporal de LAB02. La prueba no se devuelve: así la etapa de selección no
    puede tocarla ni por accidente."""
    X_train, _, y_train, _ = temporal_split(df, test_start_year)
    return X_train, y_train

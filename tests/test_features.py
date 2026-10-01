import numpy as np
import pandas as pd
import pytest

from inf8239_u01 import features


def _sintetico(anios=(2023, 2024, 2025, 2026), aeropuertos=("A", "B")):
    """Dos aeropuertos, 100 días por año, tráfico que crece cada año."""
    rng = np.random.default_rng(0)
    filas = []
    for anio in anios:
        fechas = pd.date_range(f"{anio}-01-01", periods=100)
        for aero, escala in zip(aeropuertos, (1000, 10)):
            total = rng.integers(1, 100, size=100) * escala * (1 + (anio - 2023) * 0.3)
            filas.append(pd.DataFrame({
                "fecha": fechas, "aeropuerto": aero, "entrada": total / 2,
                "salida": total / 2, "total": total, "dia": fechas.day,
                "mes": fechas.month, "anio": anio,
            }))
    return features.add_calendar_features(pd.concat(filas, ignore_index=True))


def test_build_target_uses_strict_p75_per_airport_and_year():
    df = features.build_target(_sintetico())
    tasa = df.groupby(["aeropuerto", "anio"])[features.TARGET].mean()
    assert ((tasa > 0.20) & (tasa < 0.30)).all()  # ~25 % en cada aeropuerto-año
    fila = df.iloc[0]
    assert fila[features.TARGET] == int(fila["total"] > fila["umbral_p75"])


def test_build_target_rejects_invalid_quantile():
    with pytest.raises(ValueError):
        features.build_target(_sintetico(), quantile=1.5)


def test_features_do_not_include_leakage_or_identifier_columns():
    usadas = set(features.FEATURES)
    assert usadas.isdisjoint(features.LEAKAGE_COLUMNS)
    assert usadas.isdisjoint({"fecha", "anio", "umbral_p75", features.TARGET})
    assert features.TARGET not in usadas


def test_add_calendar_features_day_of_week_is_correct():
    df = pd.DataFrame({"fecha": pd.to_datetime(["2025-01-06", "2025-01-12"])})
    assert features.add_calendar_features(df)["dia_semana"].tolist() == [0, 6]


def test_temporal_split_has_no_overlap_and_only_calendar_features():
    df = features.build_target(_sintetico())
    X_tr, X_te, y_tr, y_te = features.temporal_split(df, test_start_year=2025)
    assert list(X_tr.columns) == features.FEATURES
    assert len(X_tr) == len(y_tr) and len(X_te) == len(y_te)
    assert len(X_tr) + len(X_te) == len(df)
    ordenado = df.sort_values(["fecha", "aeropuerto"]).reset_index(drop=True)
    assert ordenado.loc[X_tr.index, "fecha"].max() < ordenado.loc[X_te.index, "fecha"].min()
    assert set(ordenado.loc[X_te.index, "anio"]) == {2025, 2026}


def test_temporal_split_rejects_empty_partition():
    df = features.build_target(_sintetico(anios=(2023, 2024)))
    with pytest.raises(ValueError):
        features.temporal_split(df, test_start_year=2025)

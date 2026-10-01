"""Prueba de integración sobre el CSV real (se omite si aún no se descargó)."""
from pathlib import Path

import pytest

from inf8239_u01 import data, features

RUTA = Path("data/raw/dataset.csv")
pytestmark = pytest.mark.skipif(not RUTA.exists(), reason="dataset.csv no descargado")


def test_real_dataset_shape_airports_and_dates():
    df = data.clean_jac(data.load_raw(RUTA))
    assert df.shape == (19971, 8)
    assert df["aeropuerto"].nunique() == 7
    assert str(df["fecha"].min().date()) == "2018-01-01"
    assert not df.duplicated(["aeropuerto", "fecha"]).any()
    assert df[["entrada", "salida", "total"]].notna().all().all()


def test_real_dataset_target_is_about_25_percent_every_year_and_split_is_temporal():
    df = features.build_target(features.add_calendar_features(data.clean_jac(data.load_raw(RUTA))))
    por_anio = df.groupby("anio")[features.TARGET].mean()
    assert ((por_anio > 0.23) & (por_anio < 0.27)).all()
    X_tr, X_te, y_tr, y_te = features.temporal_split(df)
    assert (len(X_tr), len(X_te)) == (16344, 3627)
    assert y_tr.nunique() == 2 and y_te.nunique() == 2

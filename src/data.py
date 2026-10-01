"""Carga y partición de datos.

La partición hold-out se hace UNA sola vez, antes de cualquier transformación
aprendida (escalado, PCA, imputación). Los transformadores viven en el Pipeline.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import SEED, TARGET, TEST_CSV, TEST_SIZE, TRAIN_CSV


def load_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV)


def load_test() -> pd.DataFrame:
    """test.csv de Kaggle: tiene 'id' y NO tiene etiquetas."""
    return pd.read_csv(TEST_CSV)


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c != TARGET]


def split_xy(df: pd.DataFrame):
    return df[feature_columns(df)], df[TARGET]


def holdout_split(df: pd.DataFrame | None = None):
    """Partición estratificada 80/20 reproducible -> X_tr, X_te, y_tr, y_te."""
    df = load_train() if df is None else df
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

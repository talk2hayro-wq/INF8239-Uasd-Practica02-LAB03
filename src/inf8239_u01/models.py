"""Modelos de la unidad 01 (cáncer de mama) y de la práctica 02 (JAC).

Todo el preprocesamiento vive DENTRO del Pipeline: los imputadores, el
escalado y el one-hot se ajustan solo con los datos de entrenamiento de cada
partición (o de cada fold en la validación cruzada).
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .features import FEATURES_CAT, FEATURES_NUM

RANDOM_STATE = 42
PARAM_GRID = {
    "model__C": [0.1, 1, 10],
    "model__gamma": ["scale", 0.01, 0.1],
}


def build_svm(C=1.0, gamma="scale"):
    """SVM de la unidad 01 (dataset de cáncer de mama, variables numéricas)."""
    if C <= 0:
        raise ValueError("C debe ser positivo")
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", SVC(C=C, gamma=gamma, kernel="rbf", probability=True,
                      random_state=RANDOM_STATE)),
    ])


def build_preprocess(cat_cols=None, num_cols=None) -> ColumnTransformer:
    """Imputación + one-hot para categóricas; imputación + escalado para numéricas."""
    cat_cols = list(FEATURES_CAT if cat_cols is None else cat_cols)
    num_cols = list(FEATURES_NUM if num_cols is None else num_cols)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols),
    ])


def build_dummy() -> Pipeline:
    """Baseline: predice siempre la clase más frecuente del entrenamiento."""
    return Pipeline([
        ("prep", build_preprocess()),
        ("model", DummyClassifier(strategy="most_frequent")),
    ])


def build_svm_jac(C=1.0, gamma="scale") -> Pipeline:
    """SVM RBF para la práctica 02. Sin probability=True (más rápido y no
    deprecado); el ROC-AUC se calcula con decision_function."""
    if C <= 0:
        raise ValueError("C debe ser positivo")
    return Pipeline([
        ("prep", build_preprocess()),
        ("model", SVC(C=C, gamma=gamma, kernel="rbf", random_state=RANDOM_STATE)),
    ])


def build_search(n_splits: int = 4, n_jobs: int = -1) -> GridSearchCV:
    """GridSearchCV con validación cruzada temporal dentro del entrenamiento.

    Los datos de entrenamiento deben estar ordenados por fecha (lo garantiza
    features.temporal_split). El test NO interviene en la selección.
    """
    return GridSearchCV(
        estimator=build_svm_jac(),
        param_grid=PARAM_GRID,
        scoring="f1_macro",
        cv=TimeSeriesSplit(n_splits=n_splits),
        n_jobs=n_jobs,
        return_train_score=True,
    )

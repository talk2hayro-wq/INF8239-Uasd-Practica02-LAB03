"""Catálogo de configuraciones de LAB03.

Todas comparten el mismo preprocesamiento denso dentro del Pipeline, de modo
que solo cambia el modelo. Las diferencias con el guion original están
documentadas en README.md (sección LAB03).
"""
from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from . import lab03_config as cfg
from .models import build_preprocess

FAMILIAS = {
    "dummy": "referencia",
    "logistic": "lineal",
    "svm_c1": "SVM",
    "svm_c10": "SVM",
    "svm_c10_g01": "SVM",
    "svm_c1_pca": "SVM + PCA",
    "rf_100": "Random Forest",
    "rf_300": "Random Forest",
    "boost": "Boosting",
}
REFERENCIAS = ("dummy",)


def build_catalog(random_state: int = cfg.RANDOM_STATE) -> dict:
    """Devuelve {nombre: Pipeline}. 8 candidatos + el baseline de referencia.

    - SVM sin probability=True: es ~5 veces más lento y está deprecado en
      scikit-learn >= 1.9; el AUC se calcula con decision_function.
    - Random Forest con n_jobs=1 y mediciones a 1 hilo: costo comparable.
    - svm_c10_g01 es el SVM elegido en LAB02 (continuidad).
    """
    def pipe(modelo):
        return Pipeline([("prep", build_preprocess(dense=True)), ("model", modelo)])

    return {
        "dummy": pipe(DummyClassifier(strategy="most_frequent")),
        "logistic": pipe(LogisticRegression(max_iter=2000)),
        "svm_c1": pipe(SVC(C=1, gamma="scale", random_state=random_state)),
        "svm_c10": pipe(SVC(C=10, gamma="scale", random_state=random_state)),
        "svm_c10_g01": pipe(SVC(C=10, gamma=0.1, random_state=random_state)),
        "svm_c1_pca": Pipeline([
            ("prep", build_preprocess(dense=True)),
            ("pca", PCA(n_components=cfg.PCA_VARIANZA, random_state=random_state)),
            ("model", SVC(C=1, gamma="scale", random_state=random_state)),
        ]),
        "rf_100": pipe(RandomForestClassifier(n_estimators=100, max_depth=8,
                                              random_state=random_state, n_jobs=1)),
        "rf_300": pipe(RandomForestClassifier(n_estimators=300, max_depth=8,
                                              random_state=random_state, n_jobs=1)),
        "boost": pipe(HistGradientBoostingClassifier(max_depth=6, random_state=random_state)),
    }

"""Métricas y búsqueda de hiperparámetros (siempre dentro del Pipeline)."""
from time import perf_counter

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import label_binarize

from .config import CV_FOLDS, SEED
from .pipeline import MODEL_SPECS, build_pipeline


def _scores(model, X):
    """Puntajes continuos por clase: predict_proba si existe, si no decision_function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model.decision_function(X)


def evaluate(model, X, y) -> dict:
    """Métricas de un modelo YA ajustado sobre (X, y)."""
    pred = model.predict(X)
    classes = np.sort(np.unique(y))
    y_bin = label_binarize(y, classes=classes)
    return {
        "accuracy": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "pr_auc_macro": average_precision_score(y_bin, _scores(model, X), average="macro"),
        "confusión": confusion_matrix(y, pred, labels=classes),
    }


def cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)


def tune(name: str, X_train, y_train, n_jobs: int = -1) -> dict:
    """GridSearchCV sobre el Pipeline completo (el escalado/PCA se reajustan en cada fold).

    Selección por F1 macro. No toca el hold-out.
    """
    spec = MODEL_SPECS[name]
    gs = GridSearchCV(
        build_pipeline(name), param_grid=spec.grid or {},
        scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"}, refit="f1_macro",
        cv=cv_splitter(), n_jobs=n_jobs, return_train_score=False)
    t0 = perf_counter()
    gs.fit(X_train, y_train)
    search_time = perf_counter() - t0
    i = gs.best_index_
    r = gs.cv_results_
    return {
        "best_params": gs.best_params_,
        "cv_f1_mean": float(r["mean_test_f1_macro"][i]),
        "cv_f1_std": float(r["std_test_f1_macro"][i]),
        "cv_acc_mean": float(r["mean_test_accuracy"][i]),
        "n_candidates": len(r["params"]),
        "search_time_s": search_time,
        "estimator": gs.best_estimator_,
    }

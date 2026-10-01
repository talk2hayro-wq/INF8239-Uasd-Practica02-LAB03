"""Métricas y comparación de modelos sobre UNA misma partición train/test."""
from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def evaluate(model, X, y) -> dict:
    """Métricas de un modelo ya ajustado.

    Clase positiva = 1 (alta demanda).
    FP: se predijo alta demanda y no lo fue. FN: alta demanda no anticipada.
    """
    pred = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc = None
    if hasattr(model, "decision_function"):
        auc = float(roc_auc_score(y, model.decision_function(X)))
    return {
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "f1_alta": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "precision_alta": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall_alta": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": auc,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def compare_models(models: dict, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Ajusta cada modelo con el MISMO X_train/y_train y lo evalúa con el
    MISMO X_test/y_test. Devuelve una tabla con una fila por modelo."""
    filas = {}
    for nombre, modelo in models.items():
        ajustado = clone(modelo).fit(X_train, y_train)
        filas[nombre] = evaluate(ajustado, X_test, y_test)
    return pd.DataFrame(filas).T

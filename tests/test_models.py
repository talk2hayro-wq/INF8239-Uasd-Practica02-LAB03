import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC

from inf8239_u01 import evaluate, features, models

from test_features import _sintetico


def _particion():
    df = features.build_target(_sintetico())
    return features.temporal_split(df, test_start_year=2025)


# --- Unidad 01 (cáncer de mama) -------------------------------------------
def test_svm_returns_one_prediction_per_row():
    X, y = load_breast_cancer(return_X_y=True)
    model = models.build_svm()
    model.fit(X[:450], y[:450])
    assert len(model.predict(X[450:])) == len(X[450:])


def test_svm_rejects_non_positive_c():
    with pytest.raises(ValueError):
        models.build_svm(C=0)
    with pytest.raises(ValueError):
        models.build_svm_jac(C=-1)


def test_pipeline_contains_scaler_and_model():
    assert list(models.build_svm().named_steps) == ["scale", "model"]


# --- Práctica 02 (JAC) ------------------------------------------------------
def test_preprocessing_lives_inside_the_pipeline():
    for pipe in (models.build_dummy(), models.build_svm_jac()):
        assert list(pipe.named_steps) == ["prep", "model"]
        assert isinstance(pipe.named_steps["prep"], ColumnTransformer)
    assert isinstance(models.build_dummy().named_steps["model"], DummyClassifier)
    assert isinstance(models.build_svm_jac().named_steps["model"], SVC)


def test_pipeline_fits_on_raw_columns_and_handles_unseen_category():
    X_tr, X_te, y_tr, _ = _particion()
    pipe = models.build_svm_jac().fit(X_tr, y_tr)
    X_nuevo = X_te.copy()
    X_nuevo.loc[X_nuevo.index[0], "aeropuerto"] = "Aeropuerto nuevo"
    pred = pipe.predict(X_nuevo)
    assert len(pred) == len(X_nuevo)
    assert set(pred) <= {0, 1}


def test_dummy_and_svm_are_compared_on_the_same_partition():
    X_tr, X_te, y_tr, y_te = _particion()
    tabla = evaluate.compare_models(
        {"dummy": models.build_dummy(), "svm": models.build_svm_jac()},
        X_tr, y_tr, X_te, y_te,
    )
    assert list(tabla.index) == ["dummy", "svm"]
    # Ambos evalúan exactamente las mismas filas de prueba
    assert (tabla[["tn", "fp", "fn", "tp"]].sum(axis=1) == len(y_te)).all()
    # El baseline siempre predice una sola clase
    assert tabla.loc["dummy", ["tp", "fp"]].sum() in (0, len(y_te))


def test_evaluate_counts_false_positives_and_false_negatives():
    class Fijo:
        def predict(self, X):
            return np.array([1, 1, 0, 0])

    m = evaluate.evaluate(Fijo(), pd.DataFrame({"a": range(4)}), pd.Series([1, 0, 1, 0]))
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["roc_auc"] is None


def test_build_search_uses_temporal_cv_and_train_only_grid():
    busqueda = models.build_search(n_splits=3, n_jobs=1)
    assert busqueda.scoring == "f1_macro"
    assert type(busqueda.cv).__name__ == "TimeSeriesSplit"
    assert len(busqueda.param_grid["model__C"]) * len(busqueda.param_grid["model__gamma"]) == 9
    X_tr, _, y_tr, _ = _particion()
    busqueda.fit(X_tr, y_tr)
    assert set(busqueda.best_params_) == {"model__C", "model__gamma"}

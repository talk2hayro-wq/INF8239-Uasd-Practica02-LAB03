import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

from inf8239_u01 import features, models
from inf8239_u01.catalog import FAMILIAS, REFERENCIAS, build_catalog

from test_features import _sintetico


def _particion():
    df = features.build_target(_sintetico())
    return features.temporal_split(df, test_start_year=2025)


def _modelo_final(pipe):
    return pipe.steps[-1][1]


def test_catalog_has_at_least_six_candidates_and_the_three_required_families():
    catalogo = build_catalog()
    candidatos = [n for n in catalogo if n not in REFERENCIAS]
    assert len(candidatos) >= 6
    tipos = {type(_modelo_final(catalogo[n])) for n in candidatos}
    assert {SVC, RandomForestClassifier, HistGradientBoostingClassifier} <= tipos
    assert set(catalogo) == set(FAMILIAS)


def test_every_model_keeps_preprocessing_inside_the_pipeline():
    for nombre, pipe in build_catalog().items():
        assert isinstance(pipe.named_steps["prep"], ColumnTransformer), nombre
        assert pipe.steps[0][0] == "prep", nombre


def test_dense_preprocessing_outputs_a_dense_array_for_boosting_and_pca():
    X_tr, _, _, _ = _particion()
    densa = models.build_preprocess(dense=True).fit_transform(X_tr)
    dispersa = models.build_preprocess(dense=False).fit_transform(X_tr)
    assert isinstance(densa, np.ndarray) and not issparse(densa)
    assert issparse(dispersa)  # LAB02 conserva su comportamiento original
    assert densa.shape == dispersa.shape


def test_catalog_does_not_use_probability_true():
    for nombre, pipe in build_catalog().items():
        modelo = _modelo_final(pipe)
        if isinstance(modelo, SVC):
            assert modelo.probability is False, nombre


def test_random_forest_uses_a_single_job_for_comparable_costs():
    catalogo = build_catalog()
    for nombre in ("rf_100", "rf_300"):
        assert _modelo_final(catalogo[nombre]).n_jobs == 1


def test_pca_pipeline_sits_between_preprocessing_and_model():
    pipe = build_catalog()["svm_c1_pca"]
    assert list(pipe.named_steps) == ["prep", "pca", "model"]
    assert isinstance(pipe.named_steps["pca"], PCA)
    X_tr, _, y_tr, _ = _particion()
    pipe.fit(X_tr, y_tr)
    n_codificadas = pipe.named_steps["prep"].transform(X_tr).shape[1]
    assert 1 <= pipe.named_steps["pca"].n_components_ <= n_codificadas


@pytest.mark.parametrize("nombre", list(FAMILIAS))
def test_every_catalog_model_fits_and_predicts_on_the_same_partition(nombre):
    X_tr, X_te, y_tr, _ = _particion()
    pipe = build_catalog()[nombre].fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    assert len(pred) == len(X_te)
    assert set(pred) <= {0, 1}

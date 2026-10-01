import numpy as np
import pytest

from inf8239_u01 import explore, features

from test_features import _sintetico


def _entrenamiento():
    df = features.build_target(_sintetico())
    X_tr, _, y_tr, _ = features.temporal_split(df, test_start_year=2025)
    return X_tr, y_tr


def test_memory_table_measures_dense_and_sparse_before_densifying():
    X_tr, _ = _entrenamiento()
    tabla = explore.memory_table(X_tr).set_index("representacion")
    assert set(tabla.index) == {"dispersa (CSR)", "densa (float64)"}
    assert tabla["filas"].nunique() == 1 and tabla["columnas"].nunique() == 1
    assert (tabla["MB"] > 0).all()
    assert tabla.loc["densa (float64)", "MB"] > tabla.loc["dispersa (CSR)", "MB"]


def test_pca_summary_explains_why_the_numeric_block_alone_is_not_enough():
    X_tr, _ = _entrenamiento()
    info = explore.pca_summary(X_tr, variance=0.95)
    assert info["columnas_bloque_numerico"] == 1          # solo `dia`: PCA no podría reducir nada
    assert 1 <= info["componentes_para_varianza"] <= info["columnas_codificadas"]
    assert info["varianza_explicada"] >= 0.95


def test_encoded_sample_uses_only_training_rows():
    X_tr, y_tr = _entrenamiento()
    matriz, y_muestra, aero = explore.encoded_sample(X_tr, y_tr, n=100)
    assert matriz.shape[0] == len(y_muestra) == len(aero) == 100
    assert set(aero) <= set(X_tr["aeropuerto"])


def test_tsne_stability_of_identical_maps_is_perfect():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(60, 2))
    aero = np.array(["A", "B"] * 30)
    y = np.array([0, 1, 0] * 20)
    m = explore.tsne_stability(emb, emb, aero, y, k=5)
    assert m["vecinos_compartidos"] == pytest.approx(1.0)
    assert m["disparidad_procrustes"] == pytest.approx(0.0, abs=1e-12)
    assert m["azar_aeropuerto"] == pytest.approx(0.5)


def test_tsne_seed_matters_with_random_init_but_not_with_pca_init():
    X_tr, y_tr = _entrenamiento()
    matriz, y_m, aero = explore.encoded_sample(X_tr, y_tr, n=120)
    aleatorio = explore.tsne_embeddings(matriz, seeds=(42, 7), perplexity=10, init="random")
    determinista = explore.tsne_embeddings(matriz, seeds=(42, 7), perplexity=10, init="pca")
    dif_aleatorio = explore.tsne_stability(aleatorio[42], aleatorio[7], aero, y_m, k=5)
    dif_pca = explore.tsne_stability(determinista[42], determinista[7], aero, y_m, k=5)
    assert dif_aleatorio["disparidad_procrustes"] > 1e-3   # la semilla cambia el mapa
    assert dif_pca["disparidad_procrustes"] < 1e-6         # con init="pca" los mapas son idénticos


def test_tsne_stability_reports_purity_for_extra_labels():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(60, 2))
    etiquetas = np.repeat([0, 1, 2], 20)
    m = explore.tsne_stability(emb, emb, etiquetas, etiquetas, extra={"mes": etiquetas})
    assert {"pureza_mes_mapa_a", "pureza_mes_mapa_b", "azar_mes"} <= set(m)
    assert m["azar_mes"] == pytest.approx(1 / 3)

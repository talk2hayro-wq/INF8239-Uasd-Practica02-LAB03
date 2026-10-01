"""Pruebas del protocolo de LAB03: partición intacta, prueba sin tocar, reglas declaradas."""
from pathlib import Path

import pytest

from inf8239_u01 import features, lab03_config as cfg

from test_features import _sintetico

RAIZ = Path(__file__).resolve().parents[1]
CSV_REAL = RAIZ / "data" / "raw" / "dataset.csv"


def test_selection_code_never_touches_the_test_partition():
    """La selección solo puede ver entrenamiento: ni X_test, ni y_test, ni temporal_split."""
    for archivo in ("lab03_seleccion.py", "src/inf8239_u01/explore.py",
                    "src/inf8239_u01/catalog.py"):
        texto = (RAIZ / archivo).read_text(encoding="utf-8")
        for prohibido in ("X_test", "y_test", "temporal_split"):
            assert prohibido not in texto, f"{archivo} menciona {prohibido}"
    assert "train_split" in (RAIZ / "lab03_seleccion.py").read_text(encoding="utf-8")


def test_train_split_is_exactly_the_lab02_training_partition():
    df = features.build_target(_sintetico())
    X_a, y_a = features.train_split(df, 2025)
    X_b, _, y_b, _ = features.temporal_split(df, 2025)
    assert X_a.equals(X_b) and y_a.equals(y_b)
    assert cfg.RANDOM_STATE == 42


def test_config_declares_the_agreed_rules_before_comparing():
    assert cfg.METRICA_PRINCIPAL == "f1_macro"
    assert cfg.CLASE_PRIORITARIA == 1 and cfg.METRICA_PRIORITARIA == "recall_alta"
    assert cfg.N_REPETICIONES == 3 and cfg.N_HILOS == 1
    assert cfg.COSTO_PRINCIPAL == "fit_median_s"
    assert set(cfg.COSTOS_SECUNDARIOS) == {"predict_ms", "size_kb"}
    assert cfg.K_DESVIACIONES == 1.0
    assert len(set(cfg.TSNE_SEMILLAS)) == 2            # dos semillas distintas


def test_the_declaration_document_exists_and_states_the_rules():
    texto = (RAIZ / "docs" / "LAB03_declaracion.md").read_text(encoding="utf-8")
    for frase in ("F1 macro", "Alta demanda", "frontera de Pareto", "una sola vez"):
        assert frase in texto


@pytest.mark.skipif(not CSV_REAL.exists(), reason="dataset.csv no descargado")
def test_real_partition_matches_lab02_sizes():
    from inf8239_u01 import data
    df = features.build_target(features.add_calendar_features(data.clean_jac(data.load_raw(CSV_REAL))))
    X_train, y_train = features.train_split(df)
    assert len(X_train) == 16344                       # mismo entrenamiento que LAB02
    assert X_train.shape[1] == len(features.FEATURES)
    assert y_train.nunique() == 2

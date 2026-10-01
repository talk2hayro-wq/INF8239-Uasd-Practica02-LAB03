"""Pruebas sobre la evidencia generada por LAB03 (se omiten si aún no existe)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from inf8239_u01 import lab03_config as cfg
from inf8239_u01.catalog import FAMILIAS, REFERENCIAS
from inf8239_u01.green import choose_model, pareto_flags

RAIZ = Path(__file__).resolve().parents[1]
REPORTS = RAIZ / "reports"
CSV = REPORTS / "green_ai_results.csv"
DECISION = RAIZ / cfg.RUTA_DECISION
FINAL = RAIZ / cfg.RUTA_PRUEBA_FINAL

requiere_seleccion = pytest.mark.skipif(not (CSV.exists() and DECISION.exists()),
                                        reason="falta ejecutar lab03_seleccion.py")


@pytest.fixture(scope="module")
def resultados():
    return pd.read_csv(CSV)


@requiere_seleccion
def test_csv_has_at_least_six_candidates_from_the_three_required_families(resultados):
    candidatos = resultados[~resultados["model"].isin(REFERENCIAS)]
    assert len(candidatos) >= 6
    familias = " ".join(candidatos["familia"])
    assert "SVM" in familias and "Random Forest" in familias and "Boosting" in familias
    assert set(resultados["model"]) == set(FAMILIAS)


@requiere_seleccion
def test_csv_reports_cost_columns_with_three_repetitions_and_median(resultados):
    for columna in ("fit_median_s", "fit_min_s", "fit_max_s", "predict_ms", "size_kb",
                    "f1_macro", "recall_alta"):
        assert columna in resultados.columns and resultados[columna].notna().all()
    assert (resultados["fit_min_s"] <= resultados["fit_median_s"]).all()
    assert (resultados["fit_median_s"] <= resultados["fit_max_s"]).all()
    assert (resultados["size_kb"] > 0).all() and (resultados["predict_ms"] > 0).all()


@requiere_seleccion
def test_pareto_columns_can_be_recomputed_from_the_csv(resultados):
    assert resultados["is_pareto"].tolist() == pareto_flags(resultados, "f1_macro", "fit_median_s")
    assert resultados["is_pareto_all"].tolist() == pareto_flags(
        resultados, "f1_macro", ["fit_median_s", "predict_ms", "size_kb"])
    assert resultados["is_pareto"].sum() >= 1


@requiere_seleccion
def test_decision_is_reproducible_from_the_csv_and_quantified(resultados):
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["modelo_elegido"] in set(resultados["model"])
    assert resultados["elegido"].sum() == 1
    assert resultados.loc[resultados["elegido"], "model"].iloc[0] == decision["modelo_elegido"]
    assert choose_model(resultados)["modelo_elegido"] == decision["modelo_elegido"]
    q = decision["cuantificacion"]
    for clave in ("delta_f1_macro_puntos", "delta_recall_alta_puntos", "razon_tiempo_ajuste",
                  "razon_inferencia", "razon_tamano"):
        assert clave in q
    assert decision["partida"]["prueba_tocada"] is False


@requiere_seleccion
def test_selection_reproduces_the_lab02_svm_cv_score():
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    c = decision.get("consistencia_lab02")
    if c is None:
        pytest.skip("LAB02 no disponible")
    assert abs(c["diferencia"]) < 1e-3                 # misma partición y mismo preprocesamiento


@requiere_seleccion
def test_figures_and_tsne_metrics_exist_and_the_two_seeds_really_differ():
    for figura in ("pareto.png", "tsne_two_seeds.png"):
        assert (REPORTS / figura).stat().st_size > 10_000
    m = json.loads((REPORTS / "lab03_tsne_metricas.json").read_text(encoding="utf-8"))
    assert m["init_random"]["disparidad_procrustes"] > 0.05
    assert m["init_random"]["vecinos_compartidos"] < 1.0
    assert m["init_pca"]["disparidad_procrustes"] < 1e-6   # el caso del guion: mapas idénticos
    # Lo que permanece: el aeropuerto se agrupa muy por encima del azar; el target casi no
    assert m["init_random"]["pureza_aeropuerto_mapa_a"] > 3 * m["init_random"]["azar_aeropuerto"]
    assert m["init_random"]["pureza_target_mapa_a"] < m["init_random"]["azar_target"] + 0.10


@requiere_seleccion
def test_dense_memory_and_pca_evidence_are_recorded():
    memoria = pd.read_csv(REPORTS / "lab03_memoria_densa.csv").set_index("representacion")
    assert memoria.loc["densa (float64)", "MB"] > memoria.loc["dispersa (CSR)", "MB"] > 0
    pca = json.loads((REPORTS / "lab03_pca.json").read_text(encoding="utf-8"))
    assert pca["columnas_bloque_numerico"] == 1
    assert pca["componentes_para_varianza"] < pca["columnas_codificadas"]


@pytest.mark.skipif(not (FINAL.exists() and DECISION.exists()), reason="falta ejecutar lab03_cierre.py")
def test_final_test_was_evaluated_once_with_the_frozen_model():
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    assert final["modelo_elegido"] == decision["modelo_elegido"]
    assert final["evaluaciones_de_la_prueba"] == 1
    m = final["metricas_prueba"]
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == final["filas_prueba"] == 3627
    c = final.get("consistencia_lab02")
    if c:
        assert c["f1_macro_prueba_svm_c10_g01_lab02"] == pytest.approx(
            c["f1_macro_prueba_svm_c10_g01_lab03"], abs=1e-3)

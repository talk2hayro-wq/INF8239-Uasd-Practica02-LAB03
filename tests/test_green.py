import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from inf8239_u01 import lab03_config as cfg
from inf8239_u01.green import (assert_test_untouched, choose_model, environment_info,
                               measure_model, pareto_flags, time_calls)


# --- Las dos pruebas del guion LAB03 -------------------------------------------
def test_pareto_marks_dominated_rows():
    df = pd.DataFrame({"f1_macro": [.90, .90, .88], "fit_median_s": [2., 1., 3.]})
    assert pareto_flags(df) == [False, True, False]


def test_single_model_is_pareto():
    df = pd.DataFrame({"f1_macro": [.8], "fit_median_s": [1.]})
    assert pareto_flags(df) == [True]


# --- Pareto: empates, varios costos y datos inválidos ---------------------------
def test_pareto_identical_rows_do_not_dominate_each_other():
    df = pd.DataFrame({"f1_macro": [.8, .8], "fit_median_s": [1., 1.]})
    assert pareto_flags(df) == [True, True]


def test_pareto_with_several_costs_needs_to_win_in_all_of_them():
    df = pd.DataFrame({
        "f1_macro": [.90, .90, .80],
        "fit_median_s": [1., 2., 3.],
        "size_kb": [500., 10., 400.],
    })
    # Con un solo costo, la fila 1 está dominada (más lenta que la 0)
    assert pareto_flags(df, cost="fit_median_s") == [True, False, False]
    # Con ajuste y tamaño a la vez, la 1 ya no está dominada: es mucho más pequeña
    assert pareto_flags(df, cost=["fit_median_s", "size_kb"]) == [True, True, False]


def test_pareto_rejects_missing_values():
    df = pd.DataFrame({"f1_macro": [.8, None], "fit_median_s": [1., 2.]})
    with pytest.raises(ValueError):
        pareto_flags(df)


# --- Mediciones repetidas ---------------------------------------------------------
def test_time_calls_repeats_exactly_n_times():
    llamadas = []
    tiempos = time_calls(lambda: llamadas.append(1), repetitions=3)
    assert len(llamadas) == 3 and len(tiempos) == 3
    assert all(t >= 0 for t in tiempos)


def test_measure_model_reports_median_size_and_inference(tmp_path):
    X = pd.DataFrame({"a": range(50)})
    y = pd.Series([0, 1] * 25)
    r = measure_model(DummyClassifier(), X, y, X.head(10), name="dummy",
                      models_dir=tmp_path, repetitions=3)
    assert r["repeticiones"] == 3
    assert r["fit_min_s"] <= r["fit_median_s"] <= r["fit_max_s"]
    assert r["predict_ms"] >= 0
    assert (tmp_path / "dummy.joblib").exists()
    assert r["size_kb"] > 0


# --- Decisión cuantificada ---------------------------------------------------------
def _tabla():
    t = pd.DataFrame({
        "model":            ["lento_top", "barato_casi_top", "rapido_malo", "dominado"],
        "f1_macro":         [0.720,        0.715,            0.600,         0.700],
        "f1_macro_std":     [0.020,        0.020,            0.020,         0.020],
        "recall_alta":      [0.560,        0.550,            0.300,         0.540],
        "recall_alta_std":  [0.030,        0.030,            0.030,         0.030],
        "fit_median_s":     [10.0,         1.0,              0.1,           20.0],
        "predict_ms":       [100.0,        10.0,             1.0,           200.0],
        "size_kb":          [5000.,        500.,             5.,            9000.],
    })
    t["is_pareto"] = pareto_flags(t)
    return t


def test_choose_model_picks_cheapest_eligible_on_the_frontier():
    d = choose_model(_tabla())
    # rapido_malo es el más barato, pero su F1 y su recall quedan fuera de tolerancia
    assert d["modelo_elegido"] == "barato_casi_top"
    assert d["modelo_referencia_mayor_f1"] == "lento_top"
    assert "rapido_malo" not in d["elegibles"]
    assert "dominado" in d["elegibles"] and "dominado" not in d["candidatos_en_frontera"]


def test_choose_model_quantifies_what_is_gained_and_lost():
    q = choose_model(_tabla())["cuantificacion"]
    assert q["delta_f1_macro_puntos"] == pytest.approx(-0.5)
    assert q["delta_recall_alta_puntos"] == pytest.approx(-1.0)
    assert q["razon_tiempo_ajuste"] == pytest.approx(0.1)
    assert q["razon_inferencia"] == pytest.approx(0.1)
    assert q["razon_tamano"] == pytest.approx(0.1)


def test_choose_model_breaks_ties_with_smaller_serialized_size():
    t = _tabla()
    t.loc[1, "fit_median_s"] = 10.0  # empata en tiempo con lento_top
    t["is_pareto"] = True
    d = choose_model(t)
    assert d["modelo_elegido"] == "barato_casi_top"  # mismo tiempo, pero pesa 10 veces menos


# --- Protocolo y entorno ------------------------------------------------------------
def test_assert_test_untouched_blocks_a_second_evaluation(tmp_path):
    marca = tmp_path / "test_final.json"
    assert_test_untouched(marca)  # todavía no existe: no pasa nada
    marca.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ya se evaluó"):
        assert_test_untouched(marca)


def test_environment_info_has_context_to_read_the_times():
    info = environment_info()
    for clave in ("python", "sistema", "nucleos_logicos", "scikit-learn",
                  "hilos_usados_en_mediciones"):
        assert clave in info
    assert info["hilos_usados_en_mediciones"] == cfg.N_HILOS

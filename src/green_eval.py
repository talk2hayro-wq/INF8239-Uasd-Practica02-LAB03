"""Evaluación multidimensional: desempeño predictivo + costo (Green AI).

Uso: python -m src.green_eval
Salidas: reports/tables/green_ai.csv, reports/tables/entorno.json,
         reports/tables/mejores_hiperparametros.json, docs/green_ai.md
"""
import json

import pandas as pd
from sklearn.pipeline import Pipeline

from .config import (CV_FOLDS, DOCS, FIT_REPEATS, MEM_REPEATS, PREDICT_REPEATS,
                     SEED, TABLES)
from .data import holdout_split
from .evaluate import evaluate, tune
from .green import hardware_log, measure_costs
from .pipeline import MODEL_SPECS
from .utils import md_table, write_text


def _extra(pipe: Pipeline) -> tuple[int, str]:
    """(número de variables que ve el modelo, detalle estructural)"""
    clf = pipe.steps[-1][1]
    n_in = pipe[:-1].n_features_in_ if len(pipe) > 1 else pipe.n_features_in_
    n_feat = pipe["pca"].n_components_ if "pca" in pipe.named_steps else n_in
    if hasattr(clf, "n_support_"):
        return int(n_feat), f"{int(clf.n_support_.sum())} vectores de soporte"
    if hasattr(clf, "estimators_") and hasattr(clf.estimators_[0], "tree_"):
        return int(n_feat), f"{sum(t.tree_.node_count for t in clf.estimators_)} nodos"
    if hasattr(clf, "n_iter_") and clf.__class__.__name__.startswith("HistGradient"):
        return int(n_feat), f"{int(clf.n_iter_)} iteraciones de boosting"
    return int(n_feat), ""


def edge_params(name: str, best: dict) -> list:
    """Hiperparámetros numéricos cuyo mejor valor cae en un extremo de la grilla (posible grilla corta)."""
    out = []
    for key, values in MODEL_SPECS[name].grid.items():
        nums = [v for v in values if v is not None]
        if len(nums) >= 3 and best.get(key) is not None and best[key] in (min(nums), max(nums)):
            out.append(key.replace("clf__", ""))
    return out


def main():
    X_tr, X_te, y_tr, y_te = holdout_split()
    hw = hardware_log()
    rows, params, confs = [], {}, {}
    for name, spec in MODEL_SPECS.items():
        print(f"[{name}] búsqueda de hiperparámetros y medición de costos...", flush=True)
        t = tune(name, X_tr, y_tr)
        m = evaluate(t["estimator"], X_te, y_te)
        c = measure_costs(name, t["best_params"], X_tr, y_tr, X_te)
        n_feat, extra = _extra(t["estimator"])
        params[name] = {k.replace("clf__", ""): v for k, v in t["best_params"].items()}
        confs[name] = m["confusión"].tolist()
        rows.append({
            "modelo": name, "etiqueta": spec.label, "grupo": spec.group,
            "cv_f1_mean": t["cv_f1_mean"], "cv_f1_std": t["cv_f1_std"],
            "holdout_f1_macro": m["f1_macro"], "holdout_accuracy": m["accuracy"],
            "holdout_pr_auc_macro": m["pr_auc_macro"],
            **c, "n_variables_modelo": n_feat, "detalle": extra,
            "candidatos_grid": t["n_candidates"], "busqueda_cv_s": t["search_time_s"],
            "params_en_borde_grilla": ", ".join(edge_params(name, t["best_params"])),
        })
    df = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "green_ai.csv", index=False, encoding="utf-8")
    (TABLES / "entorno.json").write_text(json.dumps(hw, indent=2, ensure_ascii=False), encoding="utf-8")
    (TABLES / "mejores_hiperparametros.json").write_text(
        json.dumps(params, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    (TABLES / "matrices_confusion_holdout.json").write_text(json.dumps(confs), encoding="utf-8")

    show = df[["etiqueta", "cv_f1_mean", "cv_f1_std", "holdout_f1_macro", "fit_median_s", "pred_ms_per_1k",
               "mem_fit_mib", "size_kb", "n_variables_modelo", "detalle"]].rename(columns={
        "etiqueta": "modelo", "cv_f1_mean": "F1 CV", "cv_f1_std": "+/- CV", "holdout_f1_macro": "F1 hold-out",
        "fit_median_s": "ajuste (s, mediana)", "pred_ms_per_1k": "inferencia (ms/1000 filas)",
        "mem_fit_mib": "memoria pico ajuste (MiB)", "size_kb": "tamaño serializado (KiB)",
        "n_variables_modelo": "vars.", "detalle": "detalle"})
    edge_txt = "; ".join(f"{r.modelo}: {r.params_en_borde_grilla}" for r in df.itertuples()
                         if r.params_en_borde_grilla) or "ninguno."
    md = f"""# Evidencia Green AI

## Entorno de medición (hardware log)

| Campo | Valor |
|---|---|
| Sistema operativo | {hw['sistema_operativo']} |
| CPU | {hw['cpu']} (núcleos físicos: {hw['nucleos_fisicos']}, lógicos: {hw['nucleos_logicos']}) |
| RAM | {hw['ram_total_gib']} GiB |
| GPU | {hw['gpu']} |
| Python | {hw['python']} |
| Versiones | {', '.join(f'{k} {v}' for k, v in hw['versiones'].items())} |
| Fecha (UTC) | {hw['fecha_utc']} |

> Los tiempos y la memoria dependen de este equipo. **Vuelva a ejecutar `python -m src.green_eval` en su
> máquina** para obtener el hardware log propio; el orden relativo entre modelos suele mantenerse, los
> valores absolutos no.

## Protocolo

- Partición hold-out 80/20 estratificada (`random_state={SEED}`); CV estratificada de {CV_FOLDS} pliegues
  y `GridSearchCV` sobre el Pipeline completo, solo con el 80 % de entrenamiento (1600 filas).
- **Tiempo de ajuste:** mediana de {FIT_REPEATS} repeticiones con `perf_counter` (Pipeline completo, hiperparámetros ya elegidos).
- **Tiempo de inferencia:** mediana de {PREDICT_REPEATS} repeticiones prediciendo las 400 filas del hold-out;
  se expresa en ms por 1000 filas.
- **Memoria:** incremento del pico de RSS durante un ajuste completo, medido en un **proceso nuevo** por
  repetición (`src/mem_probe.py`, calentamiento con 200 filas) y mediana de {MEM_REPEATS} procesos. Medir dentro
  del mismo proceso subestima, porque el asignador reutiliza páginas liberadas. Sigue siendo un **estimador**
  (incluye el arranque de estructuras del Pipeline). El pico de `tracemalloc` (solo Python/NumPy) queda en
  `reports/tables/green_ai.csv`.
- **Tamaño serializado:** `joblib.dump` del Pipeline completo (escalador + PCA + modelo).
- Un solo hilo (`threadpool_limits(1)`) para que los modelos sean comparables. El tiempo de la búsqueda de
  hiperparámetros (columna `busqueda_cv_s` del CSV) se reporta aparte y no forma parte del costo de despliegue.

## Resultados

{md_table(show, floatfmt="{:.4g}")}

`F1 CV` = media +/- desviación en los 5 pliegues de entrenamiento (criterio de selección);
`F1 hold-out` = confirmación final sobre 400 filas nunca vistas.

Hiperparámetros elegidos por CV: `reports/tables/mejores_hiperparametros.json`.

Hiperparámetros cuyo valor óptimo cae en un **extremo de la grilla** (la grilla podría ser corta; ver
columna `params_en_borde_grilla` del CSV): {edge_txt}
"""
    write_text(DOCS / "green_ai.md", md)
    print(df[["modelo", "cv_f1_mean", "holdout_f1_macro", "fit_median_s", "pred_ms_per_1k",
              "mem_fit_mib", "size_kb", "n_variables_modelo", "detalle"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

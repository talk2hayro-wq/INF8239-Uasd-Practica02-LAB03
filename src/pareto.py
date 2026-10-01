"""Análisis de la Frontera de Pareto: rendimiento (F1 macro) frente a costo (Green AI).

Definición (guía, sección 6): una configuración está DOMINADA si existe otra con F1 igual o mayor y
costo igual o menor en todas las dimensiones consideradas, con mejora estricta en al menos una.
Los no dominados forman la frontera.

Rendimiento de selección: F1 macro medio de la validación cruzada (solo entrenamiento).
El F1 del hold-out se usa como prueba de sensibilidad, no para elegir.

Uso: python -m src.pareto
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import DOCS, FIGURES, PARETO_TOLERANCE_F1, TABLES
from .utils import md_table, write_text

COSTS = {
    "fit_median_s": "Tiempo de ajuste (s, mediana)",
    "pred_ms_per_1k": "Inferencia (ms / 1000 filas)",
    "mem_fit_mib": "Memoria pico de ajuste (MiB)",
    "size_kb": "Tamaño serializado (KiB)",
}
COST_SETS = {**{v: [k] for k, v in COSTS.items()}, "GLOBAL (4 costos)": list(COSTS)}


def pareto_front(df: pd.DataFrame, perf_col: str, cost_cols: list) -> list:
    """Indices no dominados. Maximiza perf_col y minimiza cost_cols."""
    perf = df[perf_col].to_numpy(dtype=float)
    cost = df[cost_cols].to_numpy(dtype=float)
    keep = []
    for i in range(len(df)):
        dominated = False
        for j in range(len(df)):
            if i == j:
                continue
            no_worse = perf[j] >= perf[i] and np.all(cost[j] <= cost[i])
            strictly = perf[j] > perf[i] or np.any(cost[j] < cost[i])
            if no_worse and strictly:
                dominated = True
                break
        if not dominated:
            keep.append(df.index[i])
    return keep


def choose(df: pd.DataFrame, tol: float) -> tuple[pd.Series, pd.DataFrame]:
    """Decisión razonada: entre los modelos útiles (F1 dentro de `tol` del mejor, sin el dummy) elige el de
    menor costo (media geométrica de razones de costo respecto al mínimo de cada dimensión)."""
    real = df[df["modelo"] != "dummy"].copy()
    best = real["cv_f1_mean"].max()
    cand = real[real["cv_f1_mean"] >= best - tol].copy()
    ratios = np.column_stack([cand[c] / cand[c].min() for c in COSTS])
    cand["costo_relativo_gm"] = np.exp(np.log(ratios).mean(axis=1))
    return cand.sort_values("costo_relativo_gm").iloc[0], cand.sort_values("costo_relativo_gm")


def main():
    df = pd.read_csv(TABLES / "green_ai.csv").set_index("modelo", drop=False)
    flags = pd.DataFrame(index=df.index)
    for perf_col, tag in (("cv_f1_mean", "CV"), ("holdout_f1_macro", "hold-out")):
        for label, cols in COST_SETS.items():
            nd = pareto_front(df, perf_col, cols)
            flags[f"{label} [{tag}]"] = [i in nd for i in df.index]
    flags.to_csv(TABLES / "pareto_frontera.csv", encoding="utf-8")

    chosen, cand = choose(df, PARETO_TOLERANCE_F1)
    best_row = df.loc[df[df["modelo"] != "dummy"]["cv_f1_mean"].idxmax()]

    # ---------- figura ----------
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plot_df = df[df["modelo"] != "dummy"]
    for ax, (col, label) in zip(axes.ravel(), COSTS.items()):
        nd = pareto_front(df, "cv_f1_mean", [col])
        for i, r in plot_df.iterrows():
            on = i in nd
            ax.errorbar(r[col], r["cv_f1_mean"], yerr=r["cv_f1_std"], fmt="none", ecolor="#999", lw=0.8, zorder=1)
            ax.scatter(r[col], r["cv_f1_mean"], s=70, zorder=3, marker="o",
                       facecolors="#D62728" if on else "white", edgecolors="#D62728" if on else "#555")
            ax.annotate(r["etiqueta"], (r[col], r["cv_f1_mean"]), textcoords="offset points",
                        xytext=(6, 5), fontsize=8)
        front = plot_df.loc[[i for i in nd if i in plot_df.index]].sort_values(col)
        ax.step(front[col], front["cv_f1_mean"], where="post", color="#D62728", lw=1, alpha=0.7, zorder=2)
        s = df.loc[chosen["modelo"]]
        ax.scatter(s[col], s["cv_f1_mean"], s=260, marker="*", color="gold", edgecolors="k", zorder=4)
        ax.set_xscale("log"); ax.set_xlabel(label + "  (menor es mejor)"); ax.set_ylabel("F1 macro CV (mayor es mejor)")
        ax.set_ylim(0.82, 0.985); ax.grid(alpha=0.25)
    fig.suptitle("Frontera de Pareto: F1 macro (CV, media +/- desv.) frente a cada costo\n"
                 "rojo lleno = no dominado; hueco = dominado; estrella = decisión razonada. "
                 "DummyClassifier (F1=0.10) queda fuera de escala.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIGURES / "06_pareto.png", dpi=140); plt.close(fig)

    # ---------- tablas ----------
    g_cv = flags["GLOBAL (4 costos) [CV]"]; g_ho = flags["GLOBAL (4 costos) [hold-out]"]
    resumen = pd.DataFrame({
        "modelo": df["etiqueta"],
        "F1 CV": df["cv_f1_mean"], "F1 hold-out": df["holdout_f1_macro"],
        **{f"frontera {lab.split(' (')[0].lower()}": flags[f"{lab} [CV]"].map({True: "sí", False: "no"})
           for lab in COST_SETS if not lab.startswith("GLOBAL")},
        "FRONTERA GLOBAL (CV)": g_cv.map({True: "SÍ", False: "no"}),
        "FRONTERA GLOBAL (hold-out)": g_ho.map({True: "SÍ", False: "no"}),
    })
    resumen.to_csv(TABLES / "pareto_resumen.csv", index=False, encoding="utf-8")

    ref = best_row
    dec = cand.copy()
    dec_tab = pd.DataFrame({
        "modelo": dec["etiqueta"], "F1 CV": dec["cv_f1_mean"], "dF1 vs mejor (pts)": (dec["cv_f1_mean"] - ref["cv_f1_mean"]) * 100,
        "ajuste (x del min.)": dec["fit_median_s"] / dec["fit_median_s"].min(),
        "inferencia (x)": dec["pred_ms_per_1k"] / dec["pred_ms_per_1k"].min(),
        "memoria (x)": dec["mem_fit_mib"] / dec["mem_fit_mib"].min(),
        "tamaño (x)": dec["size_kb"] / dec["size_kb"].min(),
        "costo relativo (media geom.)": dec["costo_relativo_gm"],
    })
    vs = df.loc[chosen["modelo"]]
    red = lambda c: (1 - vs[c] / ref[c]) * 100

    front_names = ", ".join(df.loc[g_cv[g_cv].index, "etiqueta"])
    dom_names = ", ".join(df.loc[g_cv[~g_cv].index, "etiqueta"])
    md = f"""# Análisis de Pareto

**Rendimiento:** F1 macro medio de la validación cruzada de 5 pliegues (solo entrenamiento).
**Costos:** tiempo de ajuste, latencia de inferencia, memoria pico y tamaño serializado (`docs/green_ai.md`).
Dominancia estricta: una configuración se elimina solo si otra es igual o mejor en F1 y en todos los costos, y
estrictamente mejor en al menos uno. Figura: `reports/figures/06_pareto.png`.

## Frontera por dimensión y global

{md_table(resumen, floatfmt="{:.4f}")}

- **Frontera global (CV):** {front_names}.
- **Dominados globalmente:** {dom_names}.
- `DummyClassifier` está en la frontera solo por ser el punto de menor costo; su F1 (0.10) lo hace inutilizable.
  Marca el extremo trivial de la frontera, no una opción.
- La columna *hold-out* repite el análisis con el F1 de las 400 filas reservadas, como prueba de sensibilidad.

## Decisión razonada (tolerancia de {PARETO_TOLERANCE_F1 * 100:.1f} puntos de F1)

Modelos útiles con F1 CV dentro de {PARETO_TOLERANCE_F1 * 100:.1f} puntos del mejor
({ref['etiqueta']}, {ref['cv_f1_mean']:.4f}), ordenados por costo relativo (1.0 = el mínimo de cada dimensión
entre estos candidatos):

{md_table(dec_tab, floatfmt="{:.3f}")}

**Selección: {chosen['etiqueta']}.** Frente al modelo de mayor F1 CV ({ref['etiqueta']}), cede
{(ref['cv_f1_mean'] - vs['cv_f1_mean']) * 100:.2f} puntos de F1 CV (desviación entre pliegues: {vs['cv_f1_std'] * 100:.2f} puntos) y reduce el
tiempo de ajuste en {red('fit_median_s'):.0f} %, la latencia en {red('pred_ms_per_1k'):.0f} %, la memoria en
{red('mem_fit_mib'):.0f} % y el tamaño serializado en {red('size_kb'):.0f} %.

## Cautelas

- La frontera se calcula sobre **medianas**; diferencias de tiempo pequeñas entre modelos de costo similar
  no son robustas frente al ruido del hardware (ver columnas IQR en `reports/tables/green_ai.csv`).
- Un modelo puede entrar a la frontera por una ventaja minima en un solo costo (por ejemplo, un ajuste algunos
  milisegundos más rápido) aunque su F1 sea muy inferior: la frontera **no es por sí sola un criterio de
  selección**. Por eso se aplica además la regla de tolerancia de F1 sobre los modelos útiles.
- Las diferencias de F1 entre los modelos lineales y el SVM RBF (< 1.5 puntos) son comparables a la
  desviación entre pliegues; no deben leerse como superioridad estadística.
- Los resultados dependen del hardware de medición y de un dataset posiblemente curado (ver ficha técnica).
"""
    write_text(DOCS / "pareto.md", md)
    print(resumen.round(4).to_string(index=False))
    print("\nSeleccion:", chosen["modelo"])
    print(dec_tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

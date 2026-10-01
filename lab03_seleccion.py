"""LAB03 · Selección de modelo (NO toca la prueba 2025-2026).

Ejecución (desde la raíz del proyecto):
    python lab03_seleccion.py

Usa el mismo dataset, target, exclusiones y partición temporal de LAB02.
Las reglas están declaradas en docs/LAB03_declaracion.md y en
src/inf8239_u01/lab03_config.py, y se versionaron antes de comparar.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate

from src.inf8239_u01 import lab03_config as cfg
from src.inf8239_u01.catalog import FAMILIAS, build_catalog
from src.inf8239_u01.data import URL_JAC, clean_jac, ensure_dataset, load_raw
from src.inf8239_u01.explore import (encoded_sample, memory_table, pca_summary,
                                     tsne_embeddings, tsne_stability)
from src.inf8239_u01.features import (FEATURES, add_calendar_features, build_target,
                                      train_split)
from src.inf8239_u01.green import (choose_model, environment_info, measure_model,
                                   pareto_flags)

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

# 0. Declaración (fijada antes de comparar) -----------------------------------
print("== Declaración previa ==")
print(f"Métrica principal : {cfg.METRICA_PRINCIPAL} (CV temporal, {cfg.CV_SPLITS} folds)")
print(f"Clase prioritaria : {cfg.CLASE_PRIORITARIA} (alta_demanda) -> {cfg.METRICA_PRIORITARIA}")
print(f"Costo principal   : {cfg.COSTO_PRINCIPAL}; secundarios: {cfg.COSTOS_SECUNDARIOS}")
print(f"Regla             : elegibles a <= {cfg.K_DESVIACIONES} desv. del mejor; frontera; menor costo")

# 1. Mismo dataset, target y exclusiones que LAB02; SOLO entrenamiento ---------
df = build_target(add_calendar_features(clean_jac(load_raw(ensure_dataset(URL_JAC)))))
X_train, y_train = train_split(df)
print(f"\nEntrenamiento: {X_train.shape} | positivos: {y_train.mean():.3f} | variables: {FEATURES}")

catalogo = build_catalog()
candidatos = [n for n in catalogo if n != "dummy"]
print(f"Configuraciones: {len(catalogo)} ({len(candidatos)} candidatas + baseline)")

# 2. Validación cruzada temporal (solo entrenamiento) -------------------------
cv = TimeSeriesSplit(n_splits=cfg.CV_SPLITS)
scoring = {
    "f1_macro": "f1_macro",
    "recall_macro": "recall_macro",
    "recall_alta": make_scorer(recall_score, pos_label=cfg.CLASE_PRIORITARIA, zero_division=0),
}
folds, resumen_cv = [], {}
for nombre, modelo in catalogo.items():
    r = cross_validate(clone(modelo), X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    for i in range(cfg.CV_SPLITS):
        folds.append({"model": nombre, "fold": i + 1,
                      "f1_macro": r["test_f1_macro"][i], "recall_alta": r["test_recall_alta"][i],
                      "recall_macro": r["test_recall_macro"][i], "fit_time_s": r["fit_time"][i]})
    resumen_cv[nombre] = {
        "f1_macro": r["test_f1_macro"].mean(), "f1_macro_std": r["test_f1_macro"].std(ddof=1),
        "recall_alta": r["test_recall_alta"].mean(), "recall_alta_std": r["test_recall_alta"].std(ddof=1),
        "recall_macro": r["test_recall_macro"].mean(),
    }
    print(f"CV {nombre:12s} F1 macro {resumen_cv[nombre]['f1_macro']:.4f} "
          f"± {resumen_cv[nombre]['f1_macro_std']:.4f} | recall alta {resumen_cv[nombre]['recall_alta']:.4f}")
pd.DataFrame(folds).round(4).to_csv(REPORTS / "lab03_cv_folds.csv", index=False)

# 3. Mediciones repetidas: 3 ajustes e inferencias, mediana, tamaño ------------
X_infer = X_train.tail(cfg.N_FILAS_INFERENCIA)   # solo para cronometrar
filas = []
for nombre, modelo in catalogo.items():
    m = measure_model(modelo, X_train, y_train, X_infer, name=nombre,
                      models_dir=REPORTS / "models")
    filas.append({"model": nombre, "familia": FAMILIAS[nombre], **resumen_cv[nombre], **m})
    print(f"Medido {nombre:12s} ajuste {m['fit_median_s']:.3f} s | "
          f"inferencia {m['predict_ms']:.1f} ms | {m['size_kb']:.0f} KB")
results = pd.DataFrame(filas)

# 4. Frontera de Pareto ---------------------------------------------------------
results["is_pareto"] = pareto_flags(results, "f1_macro", "fit_median_s")
results["is_pareto_predict"] = pareto_flags(results, "f1_macro", "predict_ms")
results["is_pareto_size"] = pareto_flags(results, "f1_macro", "size_kb")
results["is_pareto_all"] = pareto_flags(results, "f1_macro", ["fit_median_s", "predict_ms", "size_kb"])

# 5. Decisión cuantificada (regla declarada) -----------------------------------
decision = choose_model(results)
# Sensibilidad (informativa): cómo cambiaría la elección con otra tolerancia.
# NO cambia la decisión: la regla declarada usa k = 1.
decision["sensibilidad_k_informativa"] = {
    str(k): choose_model(results, k=k)["modelo_elegido"] for k in (0.5, 1.0, 1.5, 2.0)
}
ruta_lab02 = REPORTS / "svm_cv_results.csv"
if ruta_lab02.exists():
    lab02 = pd.read_csv(ruta_lab02)
    fila = lab02[(lab02["param_model__C"].astype(float) == 10) &
                 (lab02["param_model__gamma"].astype(str) == "0.1")]
    if len(fila):
        cv_lab02 = float(fila["mean_test_score"].iloc[0])
        cv_lab03 = float(results.set_index("model").loc["svm_c10_g01", "f1_macro"])
        decision["consistencia_lab02"] = {
            "f1_macro_cv_svm_c10_g01_lab02": round(cv_lab02, 4),
            "f1_macro_cv_svm_c10_g01_lab03": round(cv_lab03, 4),
            "diferencia": round(cv_lab03 - cv_lab02, 4),
        }
        print("Consistencia con LAB02 (CV del mismo SVM):", decision["consistencia_lab02"])
decision["partida"] = {
    "particion": "temporal: entrenamiento 2018-2024 (la prueba 2025-2026 no se usó)",
    "filas_entrenamiento": int(len(X_train)),
    "prueba_tocada": False,
}
decision["entorno"] = environment_info()
Path(cfg.RUTA_DECISION).write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
results["elegido"] = results["model"] == decision["modelo_elegido"]

columnas = ["model", "familia", "f1_macro", "f1_macro_std", "recall_alta", "recall_alta_std",
            "recall_macro", "fit_median_s", "fit_min_s", "fit_max_s", "predict_ms", "size_kb",
            "is_pareto", "is_pareto_predict", "is_pareto_size", "is_pareto_all", "elegido"]
results = results[columnas]
results.round(5).to_csv(REPORTS / "green_ai_results.csv", index=False)
print("\n== Resultados (validación cruzada temporal) ==")
print(results.sort_values(["is_pareto", "f1_macro"], ascending=[False, False]).round(4).to_string(index=False))
print("\n== Decisión ==")
print(json.dumps({k: decision[k] for k in ("modelo_elegido", "modelo_referencia_mayor_f1",
                                           "elegibles", "cuantificacion")}, indent=2, ensure_ascii=False))

# 6. Frontera de Pareto: figura ------------------------------------------------
paneles = [("fit_median_s", "Mediana de ajuste (s)", "is_pareto"),
           ("predict_ms", f"Inferencia, {cfg.N_FILAS_INFERENCIA} filas (ms)", "is_pareto_predict"),
           ("size_kb", "Tamaño serializado (KB)", "is_pareto_size")]
fig, ejes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
for eje, (costo, etiqueta, bandera) in zip(ejes, paneles):
    color = results[bandera].map({True: "#0f7c7b", False: "#94a3b8"})
    eje.scatter(results[costo], results["f1_macro"], c=color, s=70, zorder=3)
    frontera = results[results[bandera]].sort_values(costo)
    eje.step(frontera[costo], frontera["f1_macro"], where="post", color="#0f7c7b", alpha=.5, zorder=2)
    desplazamiento = {"svm_c1": (-8, -17), "svm_c10": (6, -14), "svm_c10_g01": (6, 7),
                      "svm_c1_pca": (-52, -14)}
    for _, r in results.iterrows():
        eje.annotate(r["model"], (r[costo], r["f1_macro"]), fontsize=8,
                     xytext=desplazamiento.get(r["model"], (4, 4)), textcoords="offset points")
    elegido = results[results["elegido"]].iloc[0]
    eje.scatter([elegido[costo]], [elegido["f1_macro"]], s=260, facecolors="none",
                edgecolors="#c2410c", linewidths=2, zorder=4)
    eje.set_xscale("log")
    eje.set_xlabel(etiqueta)
    eje.grid(alpha=.3)
ejes[0].set_ylabel("F1 macro (CV temporal)")
fig.suptitle(f"Rendimiento y costo · verde = frontera de Pareto · círculo naranja = elegido "
             f"({decision['modelo_elegido']})")
fig.tight_layout()
fig.savefig(REPORTS / "pareto.png", dpi=160)
plt.close(fig)

# 7. PCA: ¿aplica aquí? ------------------------------------------------------------
memoria = memory_table(X_train)
memoria.to_csv(REPORTS / "lab03_memoria_densa.csv", index=False)
pca_info = pca_summary(X_train)
Path(REPORTS / "lab03_pca.json").write_text(json.dumps(pca_info, indent=2), encoding="utf-8")
print("\n== Costo de densificar ==")
print(memoria.to_string(index=False))
print("== PCA ==", pca_info)
svm, svm_pca = results.set_index("model").loc["svm_c1"], results.set_index("model").loc["svm_c1_pca"]
print(f"SVM sin PCA F1 {svm['f1_macro']:.4f} ({svm['fit_median_s']:.2f} s)  |  "
      f"con PCA F1 {svm_pca['f1_macro']:.4f} ({svm_pca['fit_median_s']:.2f} s)")

# 8. Dos mapas t-SNE con semillas distintas (muestra de ENTRENAMIENTO) ----------
matriz, y_muestra, aero_muestra = encoded_sample(X_train, y_train)
mapas = tsne_embeddings(matriz, init="random")          # la semilla SÍ cambia el mapa
seeds = list(mapas)
estabilidad = {"init_random": tsne_stability(mapas[seeds[0]], mapas[seeds[1]], aero_muestra, y_muestra)}
mapas_pca = tsne_embeddings(matriz, init="pca")         # caso del guion: semilla casi irrelevante
estabilidad["init_pca"] = tsne_stability(mapas_pca[seeds[0]], mapas_pca[seeds[1]], aero_muestra, y_muestra)
Path(REPORTS / "lab03_tsne_metricas.json").write_text(json.dumps(estabilidad, indent=2), encoding="utf-8")
for init, valores in estabilidad.items():
    print(f"== t-SNE {init} ==", {k: round(v, 3) for k, v in valores.items()})

codigos_aero = pd.Categorical(aero_muestra).codes
fig, ejes = plt.subplots(2, 2, figsize=(12, 10))
for col, seed in enumerate(seeds):
    emb = mapas[seed]
    ejes[0, col].scatter(emb[:, 0], emb[:, 1], c=y_muestra, s=12, cmap="viridis")
    ejes[0, col].set_title(f"t-SNE seed={seed} (init aleatorio) · color = alta_demanda")
    ejes[1, col].scatter(emb[:, 0], emb[:, 1], c=codigos_aero, s=12, cmap="tab10")
    ejes[1, col].set_title(f"t-SNE seed={seed} (init aleatorio) · color = aeropuerto")
fig.tight_layout()
fig.savefig(REPORTS / "tsne_two_seeds.png", dpi=160)
plt.close(fig)

# 9. Entorno -----------------------------------------------------------------------
entorno = environment_info()
texto = "\n".join(f"{k}: {v}" for k, v in entorno.items())
(REPORTS / "lab03_entorno.txt").write_text(texto + "\n", encoding="utf-8")
print("\n== Entorno ==\n" + texto)
print("\nEvidencias en", REPORTS.resolve(), "| la prueba 2025-2026 NO se tocó.")

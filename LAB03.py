"""LAB03 · Ensambles, reducción dimensional, mediciones repetidas y decisión por Pareto.

Versión fiel al guion LAB03.py entregado: mismos pasos 2-10, en el mismo orden y con
los mismos nombres de modelos, columnas y archivos. Lo que se separó del guion está
marcado en el código como «CAMBIO n» (necesario) o «ADICIÓN n» (agregado) y se
explica en README.md, sección «Cambios respecto al guion».

Ejecución (desde la raíz del proyecto, con el entorno activado):
    python LAB03.py --fase seleccion    # pasos 1-10. NO toca la prueba 2025-2026
    python LAB03.py --fase cierre       # paso 11: evaluación ÚNICA de la prueba
    python LAB03.py                     # las dos fases seguidas

Las reglas (métrica principal, clase prioritaria, costos y regla de decisión) están en
docs/LAB03_declaracion.md y en src/inf8239_u01/lab03_config.py, versionadas antes de
comparar. El mismo dataset, target, exclusiones y partición temporal de LAB02.
"""
import argparse
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from src.inf8239_u01 import lab03_config as cfg
from src.inf8239_u01.data import URL_JAC, clean_jac, ensure_dataset, load_raw
from src.inf8239_u01.explore import memory_table, tsne_stability
from src.inf8239_u01.features import add_calendar_features, build_target, train_split
from src.inf8239_u01.green import assert_test_untouched, choose_model, environment_info
from src.inf8239_u01.green import pareto_flags as pareto_varios_costos
from src.inf8239_u01.models import build_preprocess

parser = argparse.ArgumentParser(description="LAB03: versión fiel al guion")
parser.add_argument("--fase", choices=["todo", "seleccion", "cierre"], default="todo")
parser.add_argument("--nueva-corrida", action="store_true",
                    help="permite rehacer TODO el protocolo aunque la prueba ya se haya evaluado")
args = parser.parse_args()
SELECCION = args.fase in ("todo", "seleccion")
CIERRE = args.fase in ("todo", "cierre")

# Solo para las pruebas automáticas: corrida de humo sobre una fracción del entrenamiento
# y con otra carpeta de salida. Con la configuración normal no se usa.
HUMO = os.environ.get("LAB03_HUMO") == "1"
SALIDA = Path(os.environ.get("LAB03_SALIDA", "reports"))
SALIDA.mkdir(parents=True, exist_ok=True)
(SALIDA / "models").mkdir(exist_ok=True)
RUTA_DECISION = SALIDA / Path(cfg.RUTA_DECISION).name
RUTA_FINAL = SALIDA / Path(cfg.RUTA_PRUEBA_FINAL).name

if RUTA_FINAL.exists() and SELECCION and not args.nueva_corrida:
    sys.exit(f"La prueba ya se evaluó ({RUTA_FINAL}) y la selección quedó congelada. Si vas a "
             "rehacer TODO el protocolo (p. ej. en otro equipo), usa --nueva-corrida.")

# 0. Declaración previa (fijada antes de comparar) --------------------------------------
print("== Declaración previa ==")
print(f"Métrica principal : {cfg.METRICA_PRINCIPAL} (CV temporal, {cfg.CV_SPLITS} folds)")
print(f"Clase prioritaria : {cfg.CLASE_PRIORITARIA} (alta_demanda) -> {cfg.METRICA_PRINCIPAL} y {cfg.METRICA_PRIORITARIA}")
print(f"Costo principal   : {cfg.COSTO_PRINCIPAL}; secundarios: {cfg.COSTOS_SECUNDARIOS}")
print(f"Regla             : elegibles a <= {cfg.K_DESVIACIONES} desv. del mejor; frontera; menor costo")

# 1. Contexto que el guion da por hecho (viene de pract02.py): X, y, preprocess, Xtr, ytr ------
df = build_target(add_calendar_features(clean_jac(load_raw(ensure_dataset(URL_JAC)))))
Xtr, ytr = train_split(df)            # CAMBIO 1: solo el entrenamiento; la prueba no se carga aquí
if HUMO:
    Xtr, ytr = Xtr.iloc[::20], ytr.iloc[::20]
print(f"\nEntrenamiento: {Xtr.shape} | positivos: {ytr.mean():.3f}")
preprocess = build_preprocess(dense=True)   # CAMBIO 2: one-hot denso (sparse_output=False)

# 2. Crear el catálogo ----------------------------------------------------------------------
N_JOBS_RF = 1   # CAMBIO 3: el guion usa -1. Con 1 todos los modelos usan un hilo (regla declarada)

models = {
    "dummy": Pipeline([("prep", preprocess), ("model", DummyClassifier(strategy="most_frequent"))]),  # ADICIÓN 1: baseline
    "logistic": Pipeline([("prep", preprocess), ("model", LogisticRegression(max_iter=2000))]),
    "svm_c1": Pipeline([("prep", preprocess), ("model", SVC(C=1, probability=True, random_state=42))]),
    "svm_c10": Pipeline([("prep", preprocess), ("model", SVC(C=10, probability=True, random_state=42))]),
    "rf_100": Pipeline([("prep", preprocess), ("model", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=N_JOBS_RF))]),
    "rf_300": Pipeline([("prep", preprocess), ("model", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=N_JOBS_RF))]),
    "boost": Pipeline([("prep", preprocess), ("model", HistGradientBoostingClassifier(max_depth=6, random_state=42))]),
}
FAMILIAS = {"dummy": "referencia", "logistic": "lineal", "svm_c1": "SVM", "svm_c10": "SVM",
            "rf_100": "Random Forest", "rf_300": "Random Forest", "boost": "Boosting",
            "svm_pca": "SVM + PCA"}

# 4 (definición). SVM con PCA. Se define aquí para que el cierre pueda reconstruirlo; se mide en el paso 4
svm_pca = Pipeline([
    ("prep", preprocess),                                   # CAMBIO 4: el guion aplica PCA a Xtr con texto
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=cfg.PCA_VARIANZA, random_state=42)),
    ("model", SVC(C=1, probability=True, random_state=42)),
])
todos = {**models, "svm_pca": svm_pca}

if SELECCION:
    # 3. Medir 3 veces ---------------------------------------------------------------------
    cv = TimeSeriesSplit(n_splits=cfg.CV_SPLITS)
    scoring = {"f1_macro": "f1_macro", "recall_macro": "recall_macro",
               "recall_alta": make_scorer(recall_score, pos_label=cfg.CLASE_PRIORITARIA, zero_division=0)}
    Xinf = Xtr.tail(cfg.N_FILAS_INFERENCIA)      # CAMBIO 5: lote de ENTRENAMIENTO para cronometrar inferencia
    folds = []

    def medir(name, model):
        # CAMBIO 5: el guion puntúa con la prueba (F1 y recall). Aquí: CV temporal solo con entrenamiento
        r = cross_validate(clone(model), Xtr, ytr, cv=cv, scoring=scoring)
        for i in range(cfg.CV_SPLITS):
            folds.append({"model": name, "fold": i + 1, "f1_macro": r["test_f1_macro"][i],
                          "recall_alta": r["test_recall_alta"][i], "recall_macro": r["test_recall_macro"][i],
                          "fit_time_s": r["fit_time"][i]})
        fit_times, predict_times = [], []
        with threadpool_limits(limits=cfg.N_HILOS):          # CAMBIO 3: 1 hilo, costo comparable
            for repetition in range(cfg.N_REPETICIONES):
                start = perf_counter(); model.fit(Xtr, ytr)
                fit_times.append(perf_counter() - start)
            for repetition in range(cfg.N_REPETICIONES):
                start = perf_counter(); model.predict(Xinf)
                predict_times.append((perf_counter() - start) * 1000)
        path = SALIDA / "models" / f"{name}.joblib"
        joblib.dump(model, path)
        fila = {"model": name, "familia": FAMILIAS[name],
                "f1_macro": r["test_f1_macro"].mean(), "f1_macro_std": r["test_f1_macro"].std(ddof=1),
                "recall_alta": r["test_recall_alta"].mean(), "recall_alta_std": r["test_recall_alta"].std(ddof=1),
                "recall_macro": r["test_recall_macro"].mean(),
                "fit_median_s": np.median(fit_times), "fit_min_s": np.min(fit_times),
                "fit_max_s": np.max(fit_times), "predict_ms": np.median(predict_times),
                "size_kb": path.stat().st_size / 1024}
        print(f"{name:10s} F1 macro {fila['f1_macro']:.4f} ± {fila['f1_macro_std']:.4f} | recall alta "
              f"{fila['recall_alta']:.4f} | ajuste {fila['fit_median_s']:.2f} s | inferencia "
              f"{fila['predict_ms']:.1f} ms | {fila['size_kb']:.0f} KB", flush=True)
        return fila

    rows = [medir(name, model) for name, model in models.items()]
    results = pd.DataFrame(rows)
    results.sort_values("f1_macro", ascending=False)

    # 4. Comparar PCA con y sin PCA. PCA requiere matriz densa ----------------------------------
    # La matriz codificada es densa y pequeña: se mide antes de densificar (nota del guion)
    memoria = memory_table(Xtr)
    memoria.to_csv(SALIDA / "lab03_memoria_densa.csv", index=False)
    print("\n== Costo de densificar ==\n" + memoria.to_string(index=False))
    results = pd.concat([results, pd.DataFrame([medir("svm_pca", svm_pca)])], ignore_index=True)
    print("Componentes:", svm_pca.named_steps["pca"].n_components_)
    pca = svm_pca.named_steps["pca"]
    pca_info = {"columnas_bloque_numerico": 1,    # solo `dia`: sola, PCA no puede reducir nada
                "columnas_codificadas": int(pca.n_features_in_),
                "componentes_para_varianza": int(pca.n_components_),
                "varianza_objetivo": cfg.PCA_VARIANZA,
                "varianza_explicada": float(pca.explained_variance_ratio_.sum()),
                "reduccion_columnas": float(1 - pca.n_components_ / pca.n_features_in_)}
    (SALIDA / "lab03_pca.json").write_text(json.dumps(pca_info, indent=2), encoding="utf-8")
    pd.DataFrame(folds).round(4).to_csv(SALIDA / "lab03_cv_folds.csv", index=False)

    # 5. Crear dos mapas t-SNE ------------------------------------------------------------------
    # La separación visual no demuestra causalidad ni valida por sí sola un clasificador.
    # CAMBIO 6: el guion usa solo columnas numéricas de X (aquí es una: `dia`) y toda X.
    # Aquí: matriz codificada, solo entrenamiento. Y init="random": con init="pca" la semilla no importa.
    prep_tsne = clone(preprocess).fit(Xtr)
    sample = Xtr.sample(min(cfg.TSNE_MUESTRA, len(Xtr)), random_state=42)
    y_sample = ytr.loc[sample.index]
    scaled = StandardScaler().fit_transform(prep_tsne.transform(sample))
    mapas = {}
    for init in ("random", "pca"):
        mapas[init] = {seed: TSNE(n_components=2, perplexity=cfg.TSNE_PERPLEXITY, random_state=seed,
                                  init=init, learning_rate="auto").fit_transform(scaled)
                       for seed in cfg.TSNE_SEMILLAS}
    a, b = cfg.TSNE_SEMILLAS
    extras = {"mes": sample["mes"].to_numpy(), "dia_semana": sample["dia_semana"].to_numpy()}
    estabilidad = {f"init_{init}": tsne_stability(mapas[init][a], mapas[init][b],   # ADICIÓN 2: cuantifica qué
                                                  sample["aeropuerto"].to_numpy(), y_sample.to_numpy(),  # permanece y
                                                  extra=extras)                                         # qué cambia
                   for init in mapas}
    (SALIDA / "lab03_tsne_metricas.json").write_text(json.dumps(estabilidad, indent=2), encoding="utf-8")
    for nombre, valores in estabilidad.items():
        print(f"== t-SNE {nombre} ==", {k: round(v, 3) for k, v in valores.items()})
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))          # ADICIÓN 3: filas por aeropuerto y por mes
    capas = [("alta_demanda", pd.Categorical(y_sample).codes, "viridis"),
             ("aeropuerto", pd.Categorical(sample["aeropuerto"]).codes, "tab10"),
             ("mes", pd.Categorical(sample["mes"]).codes, "tab20")]
    for col, seed in enumerate(cfg.TSNE_SEMILLAS):
        emb = mapas["random"][seed]
        for fila, (nombre, codigos, cmap) in enumerate(capas):
            axes[fila, col].scatter(emb[:, 0], emb[:, 1], c=codigos, s=12, cmap=cmap)
            axes[fila, col].set_title(f"t-SNE seed={seed} (init aleatorio) · color = {nombre}")
    plt.tight_layout(); plt.savefig(SALIDA / "tsne_two_seeds.png", dpi=160); plt.close(fig)

    # 6. Calcular la frontera de Pareto para los modelos entrenados -----------------------------
    def pareto_flags(df, score="f1_macro", cost="fit_median_s"):
        flags = []
        for _, row in df.iterrows():
            dominated = ((df[score] >= row[score]) & (df[cost] <= row[cost]) &
                         ((df[score] > row[score]) | (df[cost] < row[cost]))).any()
            flags.append(not bool(dominated))
        return flags

    results["is_pareto"] = pareto_flags(results)
    results["is_pareto_predict"] = pareto_flags(results, cost="predict_ms")    # ADICIÓN 4: otras fronteras
    results["is_pareto_size"] = pareto_flags(results, cost="size_kb")
    results["is_pareto_all"] = pareto_varios_costos(results, "f1_macro", ["fit_median_s", "predict_ms", "size_kb"])
    results.to_csv(SALIDA / "green_ai_results.csv", index=False)
    results.sort_values(["is_pareto", "f1_macro"], ascending=[False, False])

    # 7. Dibujar y cuantificar la decisión ------------------------------------------------------
    # ADICIÓN 5: regla de decisión declarada de antemano (guion: solo dibuja)
    decision = choose_model(results)
    decision["sensibilidad_k_informativa"] = {       # informativa: NO cambia la decisión (k declarado = 1)
        str(k): choose_model(results, k=k)["modelo_elegido"] for k in (0.5, 1.0, 1.5, 2.0)}
    decision["partida"] = {"particion": "temporal: entrenamiento 2018-2024 (la prueba 2025-2026 no se usó)",
                           "filas_entrenamiento": int(len(Xtr)), "prueba_tocada": False}
    decision["entorno"] = environment_info()
    RUTA_DECISION.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    results["elegido"] = results["model"] == decision["modelo_elegido"]
    results.round(5).to_csv(SALIDA / "green_ai_results.csv", index=False)
    print("\n== Resultados (validación cruzada temporal) ==")
    print(results.sort_values(["is_pareto", "f1_macro"], ascending=[False, False]).round(4).to_string(index=False))
    print("\n== Decisión ==")
    print(json.dumps({k: decision[k] for k in ("modelo_elegido", "modelo_referencia_mayor_f1", "elegibles",
                                               "cuantificacion", "sensibilidad_k_informativa")},
                     indent=2, ensure_ascii=False))

    paneles = [("fit_median_s", "Mediana de ajuste (s)", "is_pareto"),
               ("predict_ms", f"Inferencia, {cfg.N_FILAS_INFERENCIA} filas (ms)", "is_pareto_predict"),
               ("size_kb", "Tamaño serializado (KB)", "is_pareto_size")]
    fig, ejes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
    for ax, (costo, etiqueta, bandera) in zip(ejes, paneles):
        ax.scatter(results[costo], results.f1_macro, s=70, zorder=3,
                   c=results[bandera].map({True: "#0f7c7b", False: "#94a3b8"}))
        frontera = results[results[bandera]].sort_values(costo)
        ax.step(frontera[costo], frontera.f1_macro, where="post", color="#0f7c7b", alpha=.5, zorder=2)
        desplazamiento = {"svm_c1": (-34, -16), "svm_c10": (6, 8), "svm_pca": (-8, -15)}   # evita rótulos encimados
        for _, r in results.iterrows():
            ax.annotate(r.model, (r[costo], r.f1_macro), fontsize=8,
                        xytext=desplazamiento.get(r.model, (4, 4)), textcoords="offset points")
        e = results[results.elegido].iloc[0]
        ax.scatter([e[costo]], [e.f1_macro], s=260, facecolors="none", edgecolors="#c2410c", linewidths=2, zorder=4)
        ax.set_xscale("log"); ax.set_xlabel(etiqueta); ax.grid(alpha=.3)
    ejes[0].set_ylabel("F1 macro (CV temporal)")
    fig.suptitle(f"Rendimiento y costo · verde = frontera de Pareto · círculo naranja = elegido "
                 f"({decision['modelo_elegido']})")
    plt.tight_layout(); plt.savefig(SALIDA / "pareto.png", dpi=160); plt.close(fig)

    # 8. Prueba de Pareto: tests/test_green.py (las dos pruebas del guion, más otras). Se ejecuta con
    #    python -m pytest tests/test_green.py

    # 9. Registrar el entorno --------------------------------------------------------------------
    print("\nPython", sys.version)
    print("Sistema", platform.platform())
    print("Procesador", platform.processor())
    print("scikit-learn", sklearn.__version__)
    entorno = environment_info()
    (SALIDA / "lab03_entorno.txt").write_text("\n".join(f"{k}: {v}" for k, v in entorno.items()) + "\n",
                                              encoding="utf-8")

    # 10. Medir consumo energético y emisiones de CO2 ---------------------------------------------
    # El tiempo es contextual y NO es consumo energético. El script con CodeCarbon es aparte:
    #    pip install -r requirements-energia.txt  y  python lab03_energia.py
    print("\nEnergía y CO2: python lab03_energia.py (CodeCarbon; el tiempo de ajuste no es energía).")
    print("Evidencias en", SALIDA.resolve(), "| la prueba 2025-2026 NO se tocó.")

# 11. CIERRE: evaluación única de la prueba (ADICIÓN 6) -------------------------------------------
if CIERRE:
    # Solo aquí aparece la prueba. La decisión se lee del archivo congelado, no de una variable.
    from sklearn.metrics import ConfusionMatrixDisplay

    from src.inf8239_u01.evaluate import compare_models
    from src.inf8239_u01.features import temporal_split
    from src.inf8239_u01.models import build_svm_jac

    if not RUTA_DECISION.exists():
        sys.exit("Falta la decisión congelada: ejecuta primero `python LAB03.py --fase seleccion`.")
    if not args.nueva_corrida:
        try:
            assert_test_untouched(RUTA_FINAL)
        except RuntimeError as error:
            sys.exit(str(error))
    decision = json.loads(RUTA_DECISION.read_text(encoding="utf-8"))
    if decision["partida"]["prueba_tocada"]:
        sys.exit("La decisión indica que la prueba ya se tocó durante la selección.")
    elegido = decision["modelo_elegido"]
    print(f"\nDecisión congelada: {elegido} (referencia de mayor F1 en CV: {decision['modelo_referencia_mayor_f1']})")

    Xtr, Xte, ytr, yte = temporal_split(df)       # misma función y misma partición que LAB02
    if HUMO:
        Xtr, ytr = Xtr.iloc[::20], ytr.iloc[::20]
    print(f"entrenamiento {Xtr.shape} | prueba {Xte.shape} (positivos {yte.mean():.3f})")
    svm_lab02 = build_svm_jac(C=10, gamma=0.1)    # la SVM elegida en LAB02, solo como contexto
    contexto = {"dummy": todos["dummy"], "svm_lab02": svm_lab02}
    contexto = {k: v for k, v in contexto.items() if k != elegido}
    tabla = compare_models({elegido: todos[elegido], **contexto}, Xtr, ytr, Xte, yte)
    print(tabla.round(4).to_string())

    resultados = pd.read_csv(SALIDA / "green_ai_results.csv").set_index("model")
    cv_f1, cv_std = float(resultados.loc[elegido, "f1_macro"]), float(resultados.loc[elegido, "f1_macro_std"])
    f1_prueba = float(tabla.loc[elegido, "f1_macro"])
    limpio = lambda fila: {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in fila.items()}
    final = {"modelo_elegido": elegido, "evaluaciones_de_la_prueba": 1, "filas_prueba": int(len(yte)),
             "metricas_prueba": limpio(tabla.loc[elegido]),
             "contexto_no_decisorio": {m: limpio(tabla.loc[m]) for m in contexto},
             "f1_macro_cv": round(cv_f1, 4), "f1_macro_cv_std": round(cv_std, 4),
             "brecha_prueba_menos_cv_puntos": round((f1_prueba - cv_f1) * 100, 2),
             "dentro_de_1_desviacion_del_cv": bool(abs(f1_prueba - cv_f1) <= cv_std)}
    lab02 = Path("reports/metrics.json")
    if lab02.exists() and "svm_lab02" in tabla.index and not HUMO:
        f1_lab02 = json.loads(lab02.read_text(encoding="utf-8"))["metricas_test"]["svm_grid"]["f1_macro"]
        final["consistencia_lab02"] = {"f1_macro_prueba_lab02": round(float(f1_lab02), 4),
                                       "f1_macro_prueba_recalculado": round(float(tabla.loc["svm_lab02", "f1_macro"]), 4)}
        print("Consistencia con LAB02:", final["consistencia_lab02"])
    RUTA_FINAL.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")

    modelo_final = clone(todos[elegido]).fit(Xtr, ytr)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ConfusionMatrixDisplay.from_predictions(yte, modelo_final.predict(Xte), ax=ax, colorbar=False)
    ax.set_title(f"{elegido} · prueba 2025-2026")
    plt.tight_layout(); plt.savefig(SALIDA / "lab03_matriz_confusion_final.png", dpi=140); plt.close(fig)
    print("Resultado final guardado en", RUTA_FINAL)

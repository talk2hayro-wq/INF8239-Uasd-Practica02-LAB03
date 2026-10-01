"""Práctica 02 - Baseline vs SVM sobre un dataset real (JAC, pasajeros por aeropuerto).

Ejecución (desde la raíz del proyecto):
    python pract02.py

Pregunta: ¿puede una SVM anticipar si un día será de alta demanda en un
aeropuerto mejor que un baseline, con qué costo de error?
Unidad de análisis: un aeropuerto en un día.
Target: alta_demanda (1 si entrada+salida > P75 del aeropuerto en ese año).
"""
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from src.inf8239_u01.data import URL_JAC, clean_jac, ensure_dataset, load_raw
from src.inf8239_u01.evaluate import compare_models, evaluate
from src.inf8239_u01.features import (EXCLUDED_COLUMNS, FEATURES, LEAKAGE_COLUMNS,
                                      TARGET, add_calendar_features, build_target,
                                      temporal_split)
from src.inf8239_u01.models import build_dummy, build_search, build_svm_jac

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

# 1. Descarga (solo si data/raw/dataset.csv no existe) ------------------------
path = ensure_dataset(URL_JAC)
print("Dataset:", path)

# 2. Auditoría del archivo crudo ----------------------------------------------
raw = load_raw(path)
print(raw.shape)
print(raw.dtypes)
print(raw.head())
print(raw.tail())
assert not raw.empty

guiones = {c: int((raw[c].astype(str).str.strip() == "-").sum()) for c in ("entrada", "salida")}
print("Valores '-' :", guiones)
print("Duplicados exactos:", raw.duplicated().sum())
print("Aeropuertos (nombres originales):", raw["aeropuerto_original"].nunique())

# 3. Limpieza, target y variables ---------------------------------------------
df = clean_jac(raw)
print("Aeropuertos armonizados:", df["aeropuerto"].nunique())
print("Duplicados aeropuerto-fecha:", df.duplicated(["aeropuerto", "fecha"]).sum())
df = build_target(add_calendar_features(df))

assert TARGET in df.columns
y = df[TARGET]
print(y.value_counts(dropna=False))
assert y.notna().all()
assert y.nunique() >= 2

# Fugas e identificadores: el target sale de entrada/salida/total
DROP_COLUMNS = LEAKAGE_COLUMNS  # justificación completa en README y notebook
assert set(FEATURES).isdisjoint(EXCLUDED_COLUMNS)
print("Variables usadas:", FEATURES)
print("Excluidas:", EXCLUDED_COLUMNS)

# 4. Partición temporal ÚNICA para Dummy y SVM ---------------------------------
X_train, X_test, y_train, y_test = temporal_split(df)
print("train:", X_train.shape, "positivos:", round(y_train.mean(), 3))
print("test :", X_test.shape, "positivos:", round(y_test.mean(), 3))

# 5. Baseline vs SVM base (misma partición) ------------------------------------
comparacion = compare_models(
    {"dummy": build_dummy(), "svm_base": build_svm_jac(C=1, gamma="scale")},
    X_train, y_train, X_test, y_test,
)
print(comparacion.round(4))
svm_base = build_svm_jac().fit(X_train, y_train)
print(classification_report(y_test, svm_base.predict(X_test), digits=4))

# 6. GridSearchCV con CV temporal SOLO sobre entrenamiento ---------------------
search = build_search(n_splits=4)
t0 = time.time()
search.fit(X_train, y_train)
tiempo_grid = time.time() - t0
print(search.best_params_)
print("CV f1_macro:", round(search.best_score_, 4), f"({tiempo_grid:.0f} s)")

cols = ["param_model__C", "param_model__gamma", "mean_test_score",
        "std_test_score", "mean_train_score", "rank_test_score"]
cv_results = (pd.DataFrame(search.cv_results_)[cols]
              .sort_values("rank_test_score").reset_index(drop=True))
print(cv_results.round(4))

# 7. Evaluación final en test (una sola vez, con el modelo elegido por CV) ----
mejor = search.best_estimator_
final = evaluate(mejor, X_test, y_test)
comparacion.loc["svm_grid"] = pd.Series(final)
print(comparacion.round(4))

# 8. Evidencias ----------------------------------------------------------------
cv_results.to_csv(REPORTS / "svm_cv_results.csv", index=False)
comparacion.round(4).to_csv(REPORTS / "comparacion_modelos.csv")
resumen = {
    "target": TARGET,
    "features": FEATURES,
    "train": {"filas": int(len(X_train)), "positivos": round(float(y_train.mean()), 4)},
    "test": {"filas": int(len(X_test)), "positivos": round(float(y_test.mean()), 4)},
    "mejores_hiperparametros": {k: (v if isinstance(v, str) else float(v))
                                for k, v in search.best_params_.items()},
    "cv_f1_macro": round(float(search.best_score_), 4),
    "tiempo_grid_s": round(tiempo_grid, 1),
    "metricas_test": json.loads(comparacion.round(4).to_json(orient="index")),  # NaN -> null
}
(REPORTS / "metrics.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                                      encoding="utf-8")

fig, ejes = plt.subplots(1, 2, figsize=(9, 4))
for eje, (titulo, modelo) in zip(ejes, [("SVM base", svm_base), ("SVM GridSearchCV", mejor)]):
    ConfusionMatrixDisplay.from_predictions(y_test, modelo.predict(X_test), ax=eje,
                                            colorbar=False)
    eje.set_title(f"{titulo} - prueba 2025-2026")
fig.tight_layout()
fig.savefig(REPORTS / "matriz_confusion_svm.png", dpi=120)
print("Evidencias guardadas en", REPORTS.resolve())

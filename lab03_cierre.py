"""LAB03 · Cierre: evaluación ÚNICA en la prueba 2025-2026.

Ejecución (desde la raíz del proyecto, DESPUÉS de lab03_seleccion.py):
    python lab03_cierre.py

- Lee la decisión congelada en reports/lab03_decision.json.
- Ajusta el modelo elegido con el entrenamiento y lo evalúa UNA vez en la prueba.
- Se niega a repetirse si ya existe reports/lab03_test_final.json.
- El baseline y el SVM de LAB02 se muestran solo como contexto: no pueden
  cambiar la decisión.
Para rehacer TODO el protocolo desde cero (p. ej. en otro equipo), tras volver a
correr lab03_seleccion.py: python lab03_cierre.py --nueva-corrida
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from src.inf8239_u01 import lab03_config as cfg
from src.inf8239_u01.catalog import build_catalog
from src.inf8239_u01.data import URL_JAC, clean_jac, ensure_dataset, load_raw
from src.inf8239_u01.evaluate import compare_models
from src.inf8239_u01.features import add_calendar_features, build_target, temporal_split
from src.inf8239_u01.green import assert_test_untouched

parser = argparse.ArgumentParser(description="Evaluación final única en la prueba")
parser.add_argument("--nueva-corrida", action="store_true",
                    help="permite sobrescribir una evaluación previa (rehacer todo el protocolo)")
args = parser.parse_args()

REPORTS = Path("reports")
decision_path, final_path = Path(cfg.RUTA_DECISION), Path(cfg.RUTA_PRUEBA_FINAL)
if not decision_path.exists():
    sys.exit("Falta reports/lab03_decision.json: ejecuta primero `python lab03_seleccion.py`.")
if not args.nueva_corrida:
    try:
        assert_test_untouched(final_path)
    except RuntimeError as error:
        sys.exit(str(error))

decision = json.loads(decision_path.read_text(encoding="utf-8"))
if decision["partida"]["prueba_tocada"]:
    sys.exit("La decisión indica que la prueba ya se tocó durante la selección.")
elegido = decision["modelo_elegido"]
print(f"Decisión congelada: {elegido} (referencia de mayor F1 CV: {decision['modelo_referencia_mayor_f1']})")

# Mismo dataset, target, exclusiones y partición que LAB02
df = build_target(add_calendar_features(clean_jac(load_raw(ensure_dataset(URL_JAC)))))
X_train, X_test, y_train, y_test = temporal_split(df)
print(f"train {X_train.shape} | test {X_test.shape} (positivos {y_test.mean():.3f})")

catalogo = build_catalog()
contexto = [m for m in ("dummy", "svm_c10_g01") if m != elegido]   # baseline y SVM de LAB02
modelos = {elegido: catalogo[elegido], **{m: catalogo[m] for m in contexto}}
tabla = compare_models(modelos, X_train, y_train, X_test, y_test)
print(tabla.round(4).to_string())

# Brecha entre lo estimado por validación cruzada y lo observado en la prueba
resultados = pd.read_csv(REPORTS / "green_ai_results.csv").set_index("model")
cv_f1 = float(resultados.loc[elegido, "f1_macro"])
cv_std = float(resultados.loc[elegido, "f1_macro_std"])
f1_prueba = float(tabla.loc[elegido, "f1_macro"])

final = {
    "modelo_elegido": elegido,
    "evaluaciones_de_la_prueba": 1,
    "filas_prueba": int(len(y_test)),
    "metricas_prueba": {k: (None if pd.isna(v) else round(float(v), 4))
                        for k, v in tabla.loc[elegido].items()},
    "contexto_no_decisorio": {
        m: {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in tabla.loc[m].items()}
        for m in contexto
    },
    "f1_macro_cv": round(cv_f1, 4),
    "f1_macro_cv_std": round(cv_std, 4),
    "brecha_prueba_menos_cv_puntos": round((f1_prueba - cv_f1) * 100, 2),
    "dentro_de_1_desviacion_del_cv": bool(abs(f1_prueba - cv_f1) <= cv_std),
}

# Consistencia con LAB02: el mismo SVM (C=10, gamma=0.1) debe dar el mismo F1 en la misma prueba
lab02 = REPORTS / "metrics.json"
if lab02.exists() and "svm_c10_g01" in tabla.index:
    f1_lab02 = json.loads(lab02.read_text(encoding="utf-8"))["metricas_test"]["svm_grid"]["f1_macro"]
    final["consistencia_lab02"] = {
        "f1_macro_prueba_svm_c10_g01_lab02": round(float(f1_lab02), 4),
        "f1_macro_prueba_svm_c10_g01_lab03": round(float(tabla.loc["svm_c10_g01", "f1_macro"]), 4),
    }
    print("Consistencia con LAB02:", final["consistencia_lab02"])

final_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")

# Matriz de confusión del modelo elegido
modelo_final = catalogo[elegido].fit(X_train, y_train)
fig, eje = plt.subplots(figsize=(4.6, 4.2))
ConfusionMatrixDisplay.from_predictions(y_test, modelo_final.predict(X_test), ax=eje, colorbar=False)
eje.set_title(f"{elegido} · prueba 2025-2026")
fig.tight_layout()
fig.savefig(REPORTS / "lab03_matriz_confusion_final.png", dpi=140)
print("Resultado final guardado en", final_path)

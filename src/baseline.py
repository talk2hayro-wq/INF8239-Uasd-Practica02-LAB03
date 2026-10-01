"""Baseline documentada: DummyClassifier y LogisticRegression (referencia minima).

Uso: python -m src.baseline
Salidas: reports/tables/baseline.csv, docs/baseline.md
"""
import pandas as pd
from sklearn.model_selection import cross_validate

from .config import CV_FOLDS, DOCS, SEED, TABLES
from .data import holdout_split
from .evaluate import cv_splitter, evaluate
from .pipeline import build_pipeline
from .utils import md_table, write_text

BASELINES = ["dummy", "logreg"]     # logreg con C=1 (valor por defecto, sin búsqueda)


def main():
    X_tr, X_te, y_tr, y_te = holdout_split()
    rows = []
    for name in BASELINES:
        cv = cross_validate(build_pipeline(name), X_tr, y_tr, cv=cv_splitter(),
                            scoring=["f1_macro", "accuracy"])
        model = build_pipeline(name).fit(X_tr, y_tr)
        m = evaluate(model, X_te, y_te)
        rows.append({
            "modelo": name,
            "cv_f1_macro": cv["test_f1_macro"].mean(), "cv_f1_std": cv["test_f1_macro"].std(),
            "cv_accuracy": cv["test_accuracy"].mean(),
            "holdout_f1_macro": m["f1_macro"], "holdout_accuracy": m["accuracy"],
            "holdout_pr_auc_macro": m["pr_auc_macro"],
        })
    df = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "baseline.csv", index=False, encoding="utf-8")

    dummy, lr = rows
    gain = lr["cv_f1_macro"] - dummy["cv_f1_macro"]
    md = f"""# Baseline documentada

Referencia mínima de utilidad según la guía: cualquier modelo más complejo debe superar
**de forma relevante** a estas líneas base o se rechaza en favor de la simplicidad.

Protocolo: partición hold-out estratificada 80/20 (`random_state={SEED}`); validación cruzada
estratificada de {CV_FOLDS} pliegues **solo sobre el 80 % de entrenamiento**; Pipeline
`StandardScaler -> modelo`. `LogisticRegression` usa su configuración por defecto (`C=1`), sin búsqueda.

{md_table(df)}

## Lectura

- **DummyClassifier (`most_frequent`)** obtiene F1 macro = {dummy['cv_f1_macro']:.3f} en CV: con 4 clases
  perfectamente balanceadas el azar da 0.25 de accuracy y un F1 macro aún menor, porque solo predice una clase.
  Es el suelo absoluto.
- **LogisticRegression** alcanza F1 macro = {lr['cv_f1_macro']:.3f} +/- {lr['cv_f1_std']:.3f} en CV y
  {lr['holdout_f1_macro']:.3f} en el hold-out: mejora {gain:.3f} sobre el dummy. Este es el **umbral que
  debe superar cualquier modelo complejo**.
- Como el problema está balanceado, accuracy y F1 macro casi coinciden; se mantienen ambos por consistencia.
"""
    write_text(DOCS / "baseline.md", md)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

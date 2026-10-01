# Baseline documentada

Referencia mínima de utilidad según la guía: cualquier modelo más complejo debe superar
**de forma relevante** a estas líneas base o se rechaza en favor de la simplicidad.

Protocolo: partición hold-out estratificada 80/20 (`random_state=42`); validación cruzada
estratificada de 5 pliegues **solo sobre el 80 % de entrenamiento**; Pipeline
`StandardScaler -> modelo`. `LogisticRegression` usa su configuración por defecto (`C=1`), sin búsqueda.

| modelo | cv_f1_macro | cv_f1_std | cv_accuracy | holdout_f1_macro | holdout_accuracy | holdout_pr_auc_macro |
|---|---|---|---|---|---|---|
| dummy | 0.1000 | 0.0000 | 0.2500 | 0.1000 | 0.2500 | 0.2500 |
| logreg | 0.9480 | 0.0098 | 0.9481 | 0.9650 | 0.9650 | 0.9961 |

## Lectura

- **DummyClassifier (`most_frequent`)** obtiene F1 macro = 0.100 en CV: con 4 clases
  perfectamente balanceadas el azar da 0.25 de accuracy y un F1 macro aún menor, porque solo predice una clase.
  Es el suelo absoluto.
- **LogisticRegression** alcanza F1 macro = 0.948 +/- 0.010 en CV y
  0.965 en el hold-out: mejora 0.848 sobre el dummy. Este es el **umbral que
  debe superar cualquier modelo complejo**.
- Como el problema está balanceado, accuracy y F1 macro casi coinciden; se mantienen ambos por consistencia.

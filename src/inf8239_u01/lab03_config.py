"""Declaración previa de LAB03.

Este archivo se versiona ANTES de comparar cualquier modelo. Si algo de aquí
cambia después de ver resultados, el historial de git lo dejará a la vista.
Texto completo y justificación en docs/LAB03_declaracion.md.
"""

RANDOM_STATE = 42

# --- Métrica principal y clase prioritaria ----------------------------------
METRICA_PRINCIPAL = "f1_macro"            # F1 macro, validación cruzada temporal
CLASE_PRIORITARIA = 1                     # alta_demanda
METRICA_PRIORITARIA = "recall_alta"       # recall de la clase 1

# --- Validación (solo con entrenamiento 2018-2024) --------------------------
CV_SPLITS = 4                             # TimeSeriesSplit, igual que LAB02

# --- Mediciones repetidas ----------------------------------------------------
N_REPETICIONES = 3                        # se reporta la MEDIANA
N_HILOS = 1                               # 1 hilo: costo comparable entre modelos
N_FILAS_INFERENCIA = 3000                 # lote de filas de ENTRENAMIENTO para medir inferencia

# --- Costos y regla de decisión ----------------------------------------------
COSTO_PRINCIPAL = "fit_median_s"
COSTOS_SECUNDARIOS = ("predict_ms", "size_kb")
DESEMPATE = "size_kb"
K_DESVIACIONES = 1.0                      # tolerancia: 1 desviación estándar entre folds (ddof=1)

# --- Exploración -------------------------------------------------------------
PCA_VARIANZA = 0.95
TSNE_SEMILLAS = (42, 7)
TSNE_MUESTRA = 1000
TSNE_PERPLEXITY = 30

# --- Rutas de evidencia ------------------------------------------------------
RUTA_DECISION = "reports/lab03_decision.json"
RUTA_PRUEBA_FINAL = "reports/lab03_test_final.json"

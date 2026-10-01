"""Configuración central: rutas, semillas y parámetros del experimento.

Toda fuente de aleatoriedad del proyecto se controla desde SEED.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
TRAIN_CSV = DATA_RAW / "train.csv"
TEST_CSV = DATA_RAW / "test.csv"          # sin etiquetas: NO sirve para evaluar
DOCS = ROOT / "docs"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"
MODELS_DIR = ROOT / "models"

TARGET = "price_range"
CLASS_LABELS = {0: "bajo", 1: "medio", 2: "alto", 3: "muy alto"}

SEED = 42
TEST_SIZE = 0.20          # hold-out estratificado, reservado para la evaluación final
CV_FOLDS = 5              # validación cruzada estratificada (solo sobre entrenamiento)

# --- Green AI ---
FIT_REPEATS = 7           # repeticiones para la mediana del tiempo de ajuste
PREDICT_REPEATS = 30      # repeticiones para la mediana del tiempo de inferencia
MEM_REPEATS = 5           # repeticiones para la memoria pico
PARETO_TOLERANCE_F1 = 0.005   # 0,5 puntos de F1 (criterio "decisión razonada")

# --- t-SNE ---
TSNE_SAMPLE = 1000        # tamaño de la muestra estratificada
TSNE_PERPLEXITIES = (10, 30, 50)
TSNE_MAIN_PERPLEXITY = 30

# --- Diccionario de datos (según la documentación de Kaggle del dataset) ---
FEATURE_DICT = {
    "battery_power": ("Energía total que la batería puede almacenar", "mAh", "entera"),
    "blue": ("Tiene Bluetooth", "0/1", "binaria"),
    "clock_speed": ("Velocidad a la que el microprocesador ejecuta instrucciones", "GHz", "continua"),
    "dual_sim": ("Soporta doble SIM", "0/1", "binaria"),
    "fc": ("Resolución de la cámara frontal", "Mpx", "entera"),
    "four_g": ("Tiene 4G", "0/1", "binaria"),
    "int_memory": ("Memoria interna", "GB", "entera"),
    "m_dep": ("Profundidad (grosor) del móvil", "cm", "continua"),
    "mobile_wt": ("Peso del móvil", "g", "entera"),
    "n_cores": ("Número de núcleos del procesador", "conteo", "entera"),
    "pc": ("Resolución de la cámara principal", "Mpx", "entera"),
    "px_height": ("Alto de la resolución de pantalla", "px", "entera"),
    "px_width": ("Ancho de la resolución de pantalla", "px", "entera"),
    "ram": ("Memoria RAM", "MB", "entera"),
    "sc_h": ("Alto de la pantalla", "cm", "entera"),
    "sc_w": ("Ancho de la pantalla", "cm", "entera"),
    "talk_time": ("Duración máxima de una carga en llamada", "horas", "entera"),
    "three_g": ("Tiene 3G", "0/1", "binaria"),
    "touch_screen": ("Pantalla táctil", "0/1", "binaria"),
    "wifi": ("Tiene WiFi", "0/1", "binaria"),
    TARGET: ("Rango de precio (0 bajo, 1 medio, 2 alto, 3 muy alto)", "clase", "objetivo ordinal"),
}

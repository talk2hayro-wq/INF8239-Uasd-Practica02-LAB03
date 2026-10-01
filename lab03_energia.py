"""LAB03 · Estimación de energía y emisiones de CO2eq con CodeCarbon.

Ejecución (desde la raíz del proyecto):
    pip install -r requirements-energia.txt
    python LAB03.py --fase seleccion                 # antes: deja los modelos en reports/models
    python lab03_energia.py                          # todos esos modelos, 3 repeticiones
    python lab03_energia.py --modelos svm_c1 boost --repeticiones 5

IMPORTANTE (alcance y límites):
- Es una ESTIMACIÓN, no una medición de consumo eléctrico exacto. El tiempo de
  ajuste (green_ai_results.csv) es un costo contextual y tampoco es energía.
- Herramienta: CodeCarbon, en modo offline (sin consultar servicios externos).
- Región: país indicado con --pais (por defecto DOM, República Dominicana). El
  factor de emisión es el promedio anual de la matriz eléctrica de ese país que
  trae CodeCarbon, no la intensidad del momento ni de tu proveedor.
- Cobertura: CPU y RAM (y GPU si existe). La potencia de CPU sale de contadores de
  hardware si el sistema los expone; si no, se estima con la carga de CPU y el TDP
  del procesador. La de RAM es una potencia estimada por capacidad, no medida.
  NO incluye fuente de poder, refrigeración, pantalla, red ni la fabricación
  del equipo.
- Cada modelo corre `--repeticiones` ciclos (ajuste + inferencia) dentro del medidor
  y se reporta el valor POR CICLO, porque una corrida de pocos segundos queda por
  debajo de la resolución del medidor.
- Solo usa datos de entrenamiento: la prueba 2025-2026 no se toca.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

parser = argparse.ArgumentParser(description="Estimación de energía y CO2eq con CodeCarbon")
parser.add_argument("--pais", default="DOM", help="código ISO-3 del país (def. DOM)")
parser.add_argument("--repeticiones", type=int, default=3, help="ciclos ajuste+inferencia por modelo")
parser.add_argument("--modelos", nargs="*", default=None, help="subconjunto de los modelos de reports/models")
parser.add_argument("--modo", choices=["machine", "process"], default="machine",
                    help="machine: todo el equipo (def. de CodeCarbon); process: solo este proceso")
parser.add_argument("--salida", default="reports/lab03_energia.csv", help="CSV de salida")
args = parser.parse_args()

try:
    import codecarbon
    from codecarbon import OfflineEmissionsTracker
except ImportError:
    sys.exit("Falta CodeCarbon. Instala con: pip install -r requirements-energia.txt")

import joblib
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from src.inf8239_u01 import lab03_config as cfg
from src.inf8239_u01.data import URL_JAC, clean_jac, ensure_dataset, load_raw
from src.inf8239_u01.features import add_calendar_features, build_target, train_split

# Los modelos salen de LAB03.py (una sola definición): se cargan los .joblib y se clonan SIN ajustar
modelos_dir = Path("reports/models")
rutas = {p.stem: p for p in sorted(modelos_dir.glob("*.joblib"))}
if not rutas:
    sys.exit("Faltan los modelos de reports/models: ejecuta primero `python LAB03.py --fase seleccion`.")
catalogo = {nombre: clone(joblib.load(ruta)) for nombre, ruta in rutas.items()}
FAMILIAS = pd.read_csv("reports/green_ai_results.csv").set_index("model")["familia"].to_dict()
nombres = args.modelos or list(catalogo)
desconocidos = [n for n in nombres if n not in catalogo]
if desconocidos:
    sys.exit(f"Modelos desconocidos: {desconocidos}. Disponibles: {list(catalogo)}")

df = build_target(add_calendar_features(clean_jac(load_raw(ensure_dataset(URL_JAC)))))
X_train, y_train = train_split(df)
X_infer = X_train.tail(cfg.N_FILAS_INFERENCIA)
print(f"CodeCarbon {codecarbon.__version__} | país {args.pais} | modo {args.modo} | "
      f"{args.repeticiones} ciclos por modelo | entrenamiento {X_train.shape}")

filas = []
for nombre in nombres:
    tracker = OfflineEmissionsTracker(
        country_iso_code=args.pais, project_name=f"lab03-{nombre}", tracking_mode=args.modo,
        measure_power_secs=1, save_to_file=False, log_level="error")
    tracker.start()
    try:
        with threadpool_limits(limits=cfg.N_HILOS):
            for _ in range(args.repeticiones):
                modelo = clone(catalogo[nombre])
                modelo.fit(X_train, y_train)
                modelo.predict(X_infer)
    finally:
        tracker.stop()
    d = tracker.final_emissions_data
    n = args.repeticiones
    energia = float(d.energy_consumed)            # kWh del total de ciclos
    emisiones = float(d.emissions)                # kg CO2eq del total de ciclos
    filas.append({
        "model": nombre, "familia": FAMILIAS[nombre], "ciclos": n,
        "duracion_total_s": round(float(d.duration), 2),
        "energia_wh_por_ciclo": energia * 1000 / n,
        "cpu_wh_por_ciclo": float(d.cpu_energy) * 1000 / n,
        "ram_wh_por_ciclo": float(d.ram_energy) * 1000 / n,
        "gpu_wh_por_ciclo": float(d.gpu_energy) * 1000 / n,
        "gco2e_por_ciclo": emisiones * 1000 / n,
        "factor_implicito_g_por_kwh": (emisiones / energia * 1000) if energia > 0 else float("nan"),
    })
    print(f"{nombre:12s} {filas[-1]['energia_wh_por_ciclo']:.5f} Wh/ciclo | "
          f"{filas[-1]['gco2e_por_ciclo']:.5f} gCO2e/ciclo | {d.duration:.1f} s en total")
    if float(d.duration) < 5:
        print(f"  aviso: {nombre} duró < 5 s en total; sube --repeticiones para una estimación más estable")

resultado = pd.DataFrame(filas)
salida = Path(args.salida)
salida.parent.mkdir(parents=True, exist_ok=True)
resultado.to_csv(salida, index=False)

meta = {
    "herramienta": "CodeCarbon",
    "version": codecarbon.__version__,
    "modo_de_seguimiento": args.modo,
    "region": {"pais_iso3": args.pais, "nombre": getattr(d, "country_name", None),
               "tipo_de_factor": "promedio anual de la matriz eléctrica nacional (datos incluidos en CodeCarbon)"},
    "equipo": {"cpu": getattr(d, "cpu_model", None), "nucleos": getattr(d, "cpu_count", None),
               "ram_gb": getattr(d, "ram_total_size", None), "gpu": getattr(d, "gpu_model", None) or "ninguna"},
    "fecha_de_ejecucion": date.today().isoformat(),
    "carga": f"{args.repeticiones} ciclos de ajuste + inferencia por modelo; {cfg.N_HILOS} hilo(s); "
             f"entrenamiento {X_train.shape}, lote de inferencia {cfg.N_FILAS_INFERENCIA} filas",
    "cobertura": ["CPU", "RAM", "GPU (si existe)"],
    "no_cubre": ["fuente de poder", "refrigeración", "pantalla y periféricos", "red",
                 "fabricación del equipo", "consumo base del resto del sistema en modo process"],
    "supuestos": [
        "potencia de CPU: contadores de hardware si están disponibles; si no, carga de CPU x TDP",
        "potencia de RAM estimada por capacidad, no medida",
        "factor de emisión promedio del país, no el de la hora de la ejecución",
        "en un equipo compartido, modo machine atribuye al experimento el uso de todo el equipo",
    ],
    "advertencia": "Estimación orientativa: sirve para comparar modelos entre sí en el mismo equipo, "
                   "no como consumo eléctrico exacto ni como huella de carbono certificada.",
}
Path(salida).with_name(salida.stem + "_meta.json").write_text(
    json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nResultados en {salida} y {salida.stem}_meta.json")
print(meta["advertencia"])

"""Medición del costo experimental (Green AI).

- Tiempo de ajuste e inferencia: perf_counter, N repeticiones, MEDIANA (filtra ruido del hardware).
- Memoria: incremento del pico de RSS durante UN ajuste, medido en un proceso nuevo (src/mem_probe.py;
  captura memoria nativa de libsvm/árboles) y, como referencia, pico de tracemalloc (solo Python/NumPy).
- Tamaño serializado: joblib.dump del Pipeline completo (escalador + PCA + modelo).
- Los hilos BLAS/OpenMP se limitan a 1 para que los tiempos sean comparables entre modelos.
Los tiempos y la memoria son indicadores contextuales del hardware, no constantes universales.
"""
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import tracemalloc
from datetime import datetime, timezone
from time import perf_counter

import joblib
import numpy as np
import psutil
import sklearn
from threadpoolctl import threadpool_limits

from .config import FIT_REPEATS, MEM_REPEATS, MODELS_DIR, PREDICT_REPEATS, ROOT
from .pipeline import build_pipeline


def _cpu_name() -> str:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            import subprocess
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor() or "desconocido"


def hardware_log() -> dict:
    import pandas, scipy, matplotlib
    return {
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sistema_operativo": platform.platform(),
        "cpu": _cpu_name(),
        "nucleos_fisicos": psutil.cpu_count(logical=False),
        "nucleos_logicos": psutil.cpu_count(logical=True),
        "ram_total_gib": round(psutil.virtual_memory().total / 2**30, 2),
        "gpu": "no utilizada",
        "python": platform.python_version(),
        "versiones": {"numpy": np.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__,
                      "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                      "joblib": joblib.__version__, "psutil": psutil.__version__},
    }


def peak_memory_mib(name: str, params: dict) -> float:
    """Incremento del pico de RSS (MiB) de un ajuste, medido en un proceso nuevo (ver mem_probe)."""
    out = subprocess.run([sys.executable, "-m", "src.mem_probe", name, json.dumps(params)],
                         cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])["delta_mib"]


def _median_iqr(values):
    q = np.percentile(values, [25, 75])
    return statistics.median(values), float(q[1] - q[0])


def measure_costs(name: str, params: dict, X_tr, y_tr, X_te) -> dict:
    """Costos del modelo `name` con hiperparámetros `params` (ya elegidos por CV)."""
    with threadpool_limits(limits=1):
        build_pipeline(name, params).fit(X_tr, y_tr)          # calentamiento (no se mide)

        fit_times = []
        for _ in range(FIT_REPEATS):
            pipe = build_pipeline(name, params)
            gc.collect()
            t0 = perf_counter()
            pipe.fit(X_tr, y_tr)
            fit_times.append(perf_counter() - t0)

        pipe.predict(X_te)                                     # calentamiento
        pred_times = []
        for _ in range(PREDICT_REPEATS):
            t0 = perf_counter()
            pipe.predict(X_te)
            pred_times.append(perf_counter() - t0)

        rss = [peak_memory_mib(name, params) for _ in range(MEM_REPEATS)]

        tracemalloc.start()
        build_pipeline(name, params).fit(X_tr, y_tr)
        _, py_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(pipe, path)
    size_kb = os.path.getsize(path) / 1024

    fit_med, fit_iqr = _median_iqr(fit_times)
    pred_med, pred_iqr = _median_iqr(pred_times)
    return {
        "fit_median_s": fit_med, "fit_iqr_s": fit_iqr,
        "pred_median_ms": pred_med * 1000,
        "pred_ms_per_1k": pred_med * 1000 / len(X_te) * 1000,
        "pred_iqr_ms": pred_iqr * 1000,
        "mem_fit_mib": statistics.median(rss),
        "mem_py_peak_mib": py_peak / 2**20,
        "size_kb": size_kb,
    }

"""Sonda de memoria: mide el pico de RSS de UN ajuste en un proceso nuevo.

Se ejecuta como:  python -m src.mem_probe <modelo> '<json con hiperparámetros clf__*>'

Por qué un proceso aislado: dentro de un proceso que ya ajusto otros modelos, el asignador de memoria
reutiliza páginas liberadas y el RSS casi no se mueve (subestima). En un proceso limpio, tras un
calentamiento con 200 filas (imports perezosos), el incremento del pico refleja lo que el ajuste necesita.
Pico = high-water mark del proceso: VmHWM (Linux, reiniciado antes del ajuste), peak_wset (Windows)
u ru_maxrss (macOS). Se evita ru_maxrss en Linux porque hereda el pico del proceso padre.
"""
import gc
import json
import sys

import psutil
from threadpoolctl import threadpool_limits

from .data import holdout_split
from .pipeline import build_pipeline


def _reset_peak() -> None:
    """Linux: reinicia el high-water mark (VmHWM) al RSS actual. En otros SO no hace nada."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/clear_refs", "w") as f:
                f.write("5")
        except OSError:
            pass


def peak_rss_bytes() -> int:
    """Pico de RSS del proceso actual (no hereda el del proceso padre)."""
    if sys.platform.startswith("linux"):
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    mi = psutil.Process().memory_info()
    if hasattr(mi, "peak_wset"):                      # Windows
        return mi.peak_wset
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # macOS: bytes


def main():
    name, params = sys.argv[1], json.loads(sys.argv[2])
    X_tr, _, y_tr, _ = holdout_split()
    with threadpool_limits(limits=1):
        build_pipeline(name, params).fit(X_tr.iloc[:200], y_tr.iloc[:200])   # calentamiento
        pipe = build_pipeline(name, params)
        gc.collect()
        base = psutil.Process().memory_info().rss
        _reset_peak()
        pipe.fit(X_tr, y_tr)
        peak = peak_rss_bytes()
    print(json.dumps({"delta_mib": max(peak - base, 0) / 2**20}))


if __name__ == "__main__":
    main()

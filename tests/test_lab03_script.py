"""Pruebas de LAB03.py: sigue el guion, marca sus cambios y corre de extremo a extremo (humo)."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
SCRIPT = RAIZ / "LAB03.py"
CSV_REAL = RAIZ / "data" / "raw" / "dataset.csv"
PASOS = ["# 2. Crear el catálogo", "# 3. Medir 3 veces", "# 4. Comparar PCA", "# 5. Crear dos mapas t-SNE",
         "# 6. Calcular la frontera de Pareto", "# 7. Dibujar y cuantificar la decisión",
         "# 8. Prueba de Pareto", "# 9. Registrar el entorno", "# 10. Medir consumo energético"]


def test_script_follows_the_guion_steps_in_order():
    texto = SCRIPT.read_text(encoding="utf-8")
    posiciones = [texto.index(paso) for paso in PASOS]
    assert posiciones == sorted(posiciones), "los pasos del guion deben seguir su orden"


def test_script_keeps_the_guion_models_and_probability_true():
    texto = SCRIPT.read_text(encoding="utf-8")
    for nombre in ("logistic", "svm_c1", "svm_c10", "rf_100", "rf_300", "boost", "svm_pca"):
        assert f'"{nombre}"' in texto
    assert texto.count("probability=True") == 3                   # svm_c1, svm_c10 y svm_pca, como el guion
    assert "max_depth=8" in texto and "max_depth=6" in texto


def test_every_deviation_is_marked_and_explained_in_the_readme():
    texto = SCRIPT.read_text(encoding="utf-8")
    readme = (RAIZ / "README.md").read_text(encoding="utf-8")
    marcas = set(re.findall(r"(CAMBIO \d+|ADICIÓN \d+)", texto))
    assert {f"CAMBIO {i}" for i in range(1, 7)} <= marcas
    assert {f"ADICIÓN {i}" for i in range(1, 7)} <= marcas
    for marca in sorted(marcas):
        assert marca in readme, f"README no explica {marca}"


def _correr(salida, *argumentos):
    entorno = {**os.environ, "LAB03_HUMO": "1", "LAB03_SALIDA": str(salida)}
    return subprocess.run([sys.executable, str(SCRIPT), *argumentos], cwd=RAIZ, env=entorno,
                          capture_output=True, text=True, timeout=600)


@pytest.mark.skipif(not CSV_REAL.exists(), reason="dataset.csv no descargado")
def test_smoke_run_produces_evidence_and_protects_the_test_set(tmp_path):
    r = _correr(tmp_path, "--fase", "seleccion")
    assert r.returncode == 0, r.stderr[-800:]
    resultados = pd.read_csv(tmp_path / "green_ai_results.csv")
    assert len(resultados) == 8 and resultados["elegido"].sum() == 1
    decision = json.loads((tmp_path / "lab03_decision.json").read_text(encoding="utf-8"))
    assert decision["partida"]["prueba_tocada"] is False
    assert not (tmp_path / "lab03_test_final.json").exists()      # la selección no evalúa la prueba

    r = _correr(tmp_path, "--fase", "cierre")
    assert r.returncode == 0, r.stderr[-800:]
    final = json.loads((tmp_path / "lab03_test_final.json").read_text(encoding="utf-8"))
    assert final["modelo_elegido"] == decision["modelo_elegido"] and final["filas_prueba"] == 3627

    segundo = _correr(tmp_path, "--fase", "cierre")               # una segunda evaluación se rechaza
    assert segundo.returncode != 0 and "ya se evaluó" in (segundo.stdout + segundo.stderr)
    rehacer = _correr(tmp_path, "--fase", "seleccion")            # y la selección queda congelada
    assert rehacer.returncode != 0 and "ya se evaluó" in (rehacer.stdout + rehacer.stderr)

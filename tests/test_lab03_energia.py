import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
SCRIPT = RAIZ / "lab03_energia.py"
CSV_REAL = RAIZ / "data" / "raw" / "dataset.csv"
hay_codecarbon = importlib.util.find_spec("codecarbon") is not None


def _ejecutar(*argumentos):
    return subprocess.run([sys.executable, str(SCRIPT), *argumentos], cwd=RAIZ,
                          capture_output=True, text=True, timeout=300)


def test_help_works_without_importing_codecarbon():
    r = _ejecutar("--help")
    assert r.returncode == 0
    for opcion in ("--pais", "--repeticiones", "--modelos", "--salida"):
        assert opcion in r.stdout


def test_script_declares_tool_region_assumptions_and_coverage():
    texto = SCRIPT.read_text(encoding="utf-8")
    for frase in ("CodeCarbon", "DOM", "ESTIMACIÓN", "Cobertura", "NO incluye",
                  "consumo eléctrico exacto"):
        assert frase.lower() in texto.lower(), frase
    assert "supuestos" in texto and "cobertura" in texto and "no_cubre" in texto


def test_energy_script_only_uses_training_data():
    texto = SCRIPT.read_text(encoding="utf-8")
    assert "train_split" in texto
    for prohibido in ("X_test", "y_test", "temporal_split"):
        assert prohibido not in texto


hay_modelos = any((RAIZ / "reports" / "models").glob("dummy.joblib"))


@pytest.mark.skipif(not (hay_codecarbon and CSV_REAL.exists() and hay_modelos),
                    reason="requiere codecarbon, data/raw/dataset.csv y reports/models (LAB03.py --fase seleccion)")
def test_energy_smoke_run_writes_csv_and_metadata_with_caveats(tmp_path):
    salida = tmp_path / "energia.csv"
    r = _ejecutar("--modelos", "dummy", "logistic", "--repeticiones", "1", "--salida", str(salida))
    assert r.returncode == 0, r.stderr[-800:]
    tabla = pd.read_csv(salida)
    assert list(tabla["model"]) == ["dummy", "logistic"]
    assert (tabla["energia_wh_por_ciclo"] > 0).all() and (tabla["gco2e_por_ciclo"] > 0).all()
    meta = json.loads((tmp_path / "energia_meta.json").read_text(encoding="utf-8"))
    assert meta["herramienta"] == "CodeCarbon" and meta["region"]["pais_iso3"] == "DOM"
    assert "no como consumo eléctrico exacto" in meta["advertencia"]
    assert "fuente de poder" in meta["no_cubre"]

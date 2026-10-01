import pandas as pd
import pytest

from inf8239_u01 import data


def _crudo(filas):
    return pd.DataFrame(
        filas,
        columns=["aeropuerto_original", "entrada", "salida", "dia", "mes_nombre", "anio"],
    )


def test_download_csv_rejects_non_http_url(tmp_path):
    with pytest.raises(ValueError):
        data.download_csv("C:/datos/archivo.csv", destination=tmp_path / "x.csv")


def test_download_csv_rejects_empty_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(data.pd, "read_csv", lambda url: pd.DataFrame())
    with pytest.raises(ValueError, match="vacío"):
        data.download_csv("https://ejemplo.do/a.csv", destination=tmp_path / "x.csv")


def test_ensure_dataset_does_not_redownload_existing_file(tmp_path, monkeypatch):
    destino = tmp_path / "dataset.csv"
    destino.write_text("a,b\n1,2\n", encoding="utf-8")

    def _no_debe_descargar(*args, **kwargs):
        raise AssertionError("no debía descargar")

    monkeypatch.setattr(data, "download_csv", _no_debe_descargar)
    assert data.ensure_dataset("https://ejemplo.do/a.csv", str(destino)) == destino


def test_to_number_dash_is_zero_and_strips_spaces():
    serie = pd.Series([" -   ", "1500", " 20 "])
    assert data.to_number(serie).tolist() == [0, 1500, 20]


def test_to_number_rejects_garbage_text():
    with pytest.raises(ValueError, match="no numéricos"):
        data.to_number(pd.Series(["100", "abc"]))


def test_airport_names_are_harmonized_to_seven():
    nombres = list(data.AIRPORT_MAP)
    assert len(nombres) == 11
    assert len(set(data.AIRPORT_MAP.values())) == 7
    assert data.AIRPORT_MAP["Del Cibao"] == data.AIRPORT_MAP["Santiago"]
    assert data.AIRPORT_MAP["La Isabela"] == data.AIRPORT_MAP["Santo Domingo - El Higüero"]


def test_clean_jac_builds_date_total_and_keeps_all_rows():
    crudo = _crudo([
        ["Del Cibao", "100", "50", 1, "Enero", 2018],
        ["La Romana", " -   ", " -   ", 2, "Febrero", 2018],
    ])
    limpio = data.clean_jac(crudo)
    assert len(limpio) == 2  # no se eliminan filas
    assert set(limpio["aeropuerto"]) == {"Santiago", "La Romana"}
    assert limpio.loc[limpio["aeropuerto"] == "Santiago", "total"].iloc[0] == 150
    assert limpio.loc[limpio["aeropuerto"] == "La Romana", "total"].iloc[0] == 0
    assert str(limpio["fecha"].min().date()) == "2018-01-01"


def test_clean_jac_rejects_unknown_airport_month_and_date():
    with pytest.raises(ValueError, match="sin mapeo"):
        data.clean_jac(_crudo([["Aeropuerto X", "1", "1", 1, "Enero", 2018]]))
    with pytest.raises(ValueError, match="Meses"):
        data.clean_jac(_crudo([["Del Cibao", "1", "1", 1, "Enerito", 2018]]))
    with pytest.raises(ValueError, match="fechas"):
        data.clean_jac(_crudo([["Del Cibao", "1", "1", 31, "Febrero", 2018]]))


def test_load_raw_reads_bom_and_validates_columns(tmp_path):
    bueno = tmp_path / "ok.csv"
    bueno.write_text(
        "Aerop. de Entrada,Entrada,Salida,Dia,Mes,Año\nDel Cibao,10,20,1,Enero,2018\n",
        encoding="utf-8-sig",
    )
    assert list(data.load_raw(bueno).columns) == list(data.COLUMN_MAP.values())

    malo = tmp_path / "mal.csv"
    malo.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Faltan columnas"):
        data.load_raw(malo)

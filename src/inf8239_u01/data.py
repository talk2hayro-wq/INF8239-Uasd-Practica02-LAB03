"""Descarga, lectura y limpieza del dataset JAC de movimiento de pasajeros.

Fuente: Junta de Aviación Civil (JAC), República Dominicana.
Cada fila del archivo original es un aeropuerto en un día.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

URL_JAC = (
    "https://jac.gob.do/wp-content/uploads/2026/07/"
    "Movimiento-de-Entrada-y-Salida-de-Pasajeros-por-Aeropuertos-"
    "Junta-de-Aviacion-Civil-JAC-2018-2026.csv"
)
DEFAULT_RAW = "data/raw/dataset.csv"

RAW_COLUMNS = ["Aerop. de Entrada", "Entrada", "Salida", "Dia", "Mes", "Año"]
COLUMN_MAP = {
    "Aerop. de Entrada": "aeropuerto_original",
    "Entrada": "entrada",
    "Salida": "salida",
    "Dia": "dia",
    "Mes": "mes_nombre",
    "Año": "anio",
}

# En 2018 el archivo usa otros nombres que de 2019 en adelante (no se solapan
# en años). Se unifican a los 7 aeropuertos internacionales operativos.
AIRPORT_MAP = {
    "Del Cibao": "Santiago",
    "Las Américas, JFPG": "Santo Domingo - Las Américas",
    "El Catey, Samaná": "Samaná - El Catey",
    "La Isabela": "Santo Domingo - El Higüero",
    "La Romana": "La Romana",
    "Puerto Plata": "Puerto Plata",
    "Punta Cana": "Punta Cana",
    "Santiago": "Santiago",
    "Samaná - El Catey": "Samaná - El Catey",
    "Santo Domingo - Las Américas": "Santo Domingo - Las Américas",
    "Santo Domingo - El Higüero": "Santo Domingo - El Higüero",
}

MESES = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6,
    "Julio": 7, "Agosto": 8, "Septiembre": 9, "Octubre": 10,
    "Noviembre": 11, "Diciembre": 12,
}


def download_csv(url: str, destination: str = DEFAULT_RAW) -> Path:
    """Descarga un CSV por HTTP(S) y guarda una copia validada."""
    if not url.startswith(("https://", "http://")):
        raise ValueError("La fuente debe ser una URL HTTP(S)")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(url)
    if frame.empty:
        raise ValueError("El dataset descargado está vacío")
    frame.to_csv(path, index=False)
    return path


def ensure_dataset(url: str = URL_JAC, destination: str = DEFAULT_RAW) -> Path:
    """Devuelve la ruta del CSV; lo descarga solo si todavía no existe."""
    path = Path(destination)
    if path.exists() and path.stat().st_size > 0:
        return path
    return download_csv(url, destination)


def load_raw(path: str | Path) -> pd.DataFrame:
    """Lee el CSV crudo, valida columnas y renombra a nombres estándar."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    faltantes = [c for c in RAW_COLUMNS if c not in frame.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el CSV: {faltantes}")
    if frame.empty:
        raise ValueError("El dataset está vacío")
    return frame[RAW_COLUMNS].rename(columns=COLUMN_MAP)


def to_number(series: pd.Series) -> pd.Series:
    """Convierte texto a número. El guion ('-') significa cero operaciones.

    En el archivo original el valor mínimo numérico es 1 y nunca aparece un 0
    explícito, por lo que '-' se interpreta como "sin pasajeros".
    Cualquier otro texto no numérico produce ValueError (no se oculta).
    """
    texto = series.astype("string").str.strip()
    texto = texto.replace({"-": "0"})
    numero = pd.to_numeric(texto, errors="coerce")
    invalidos = numero.isna() & texto.notna()
    if invalidos.any():
        ejemplos = texto[invalidos].unique()[:5].tolist()
        raise ValueError(f"Valores no numéricos: {ejemplos}")
    return numero


def clean_jac(raw: pd.DataFrame) -> pd.DataFrame:
    """Limpieza mínima y justificada; no elimina filas."""
    df = raw.copy()

    desconocidos = sorted(set(df["aeropuerto_original"]) - set(AIRPORT_MAP))
    if desconocidos:
        raise ValueError(f"Aeropuertos sin mapeo: {desconocidos}")
    df["aeropuerto"] = df["aeropuerto_original"].map(AIRPORT_MAP)

    for col in ("entrada", "salida"):
        df[col] = to_number(df[col])
    df["total"] = df["entrada"] + df["salida"]

    mes_num = df["mes_nombre"].map(MESES)
    if mes_num.isna().any():
        raise ValueError("Meses no reconocidos en la columna Mes")
    df["mes"] = mes_num.astype(int)

    df["fecha"] = pd.to_datetime(
        {"year": df["anio"], "month": df["mes"], "day": df["dia"]}, errors="coerce"
    )
    if df["fecha"].isna().any():
        raise ValueError("Hay fechas inválidas (año/mes/día)")

    columnas = ["fecha", "aeropuerto", "entrada", "salida", "total",
                "dia", "mes", "anio"]
    return df[columnas].sort_values(["fecha", "aeropuerto"]).reset_index(drop=True)

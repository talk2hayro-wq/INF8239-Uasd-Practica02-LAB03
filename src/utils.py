"""Utilidades pequeñas sin dependencias extra."""
import hashlib
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}", index: bool = False, widths=None) -> str:
    """DataFrame -> tabla Markdown (sin depender de tabulate).

    `widths` (opcional): guiones por columna en el separador; pandoc los usa como ancho relativo al crear Word.
    """
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["-" * w for w in (widths or [3] * len(cols))]) + "|"]
    for _, row in d.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")   # utf-8 explicito (Windows usa cp1252 por defecto)

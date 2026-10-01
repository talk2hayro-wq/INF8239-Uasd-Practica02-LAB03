from pathlib import Path
import pandas as pd


def load_local_csv(source: str | Path, destination="data/raw/dataset.csv") -> Path:
    """Lee un CSV desde una ruta local y guarda una copia validada en `destination`."""
    src = Path(source).expanduser()

    if not src.exists():
        raise FileNotFoundError(f"No existe la ruta: {src}")
    if not src.is_file():
        raise ValueError(f"La ruta no es un archivo: {src}")
    if src.suffix.lower() != ".csv":
        raise ValueError("El archivo debe tener extensión .csv")

    frame = pd.read_csv(src)
    if frame.empty:
        raise ValueError("El dataset cargado está vacío")

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


ruta = load_local_csv(r"C:\Users\Sobeit\Downloads\bonoluz20260501.csv")
print("Copia guardada en:", ruta)

df = pd.read_csv(ruta)
print(df.shape)
df.head()
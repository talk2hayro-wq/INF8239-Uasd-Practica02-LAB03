"""Ejecuta todo el flujo en orden: python -m src.run_all

Auditoría -> baseline -> t-SNE -> evaluación Green AI -> Pareto.
La evaluación Green AI (búsqueda de hiperparámetros + mediciones) es la parte más lenta.
"""
from . import audit, baseline, green_eval, pareto, tsne_viz


def main():
    for etapa in (audit, baseline, tsne_viz, green_eval, pareto):
        print(f"\n=== {etapa.__name__} ===", flush=True)
        etapa.main()


if __name__ == "__main__":
    main()

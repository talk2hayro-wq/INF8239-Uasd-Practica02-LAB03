"""Diagnóstico visual con t-SNE (exploratorio, NO evidencia estadística).

Reproducibilidad (guía, sección 5): se documentan semilla, perplexity y tamaño de la
muestra estratificada. El escalador se ajusta solo con la partición de entrenamiento.

Uso: python -m src.tsne_viz
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (CLASS_LABELS, DOCS, FIGURES, SEED, TSNE_MAIN_PERPLEXITY,
                     TSNE_PERPLEXITIES, TSNE_SAMPLE)
from .data import holdout_split
from .utils import write_text

WARNING = ("ADVERTENCIA: t-SNE es una herramienta visual exploratoria. Los ejes no tienen significado,\n"
           "las distancias entre grupos lejanos no son una medida de similitud y la imagen NO prueba\n"
           "que existan clústeres. Depende de la semilla y de la perplexity.")
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]


def embed(Xs: np.ndarray, perplexity: int) -> np.ndarray:
    return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                max_iter=1000, random_state=SEED).fit_transform(Xs)


def scatter(ax, Z, y, title):
    for k in sorted(np.unique(y)):
        m = y == k
        ax.scatter(Z[m, 0], Z[m, 1], s=9, alpha=0.75, color=COLORS[k],
                   label=f"{k} ({CLASS_LABELS[k]})")
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])


def main():
    X_tr, _, y_tr, _ = holdout_split()
    scaler = StandardScaler().fit(X_tr)                     # solo entrenamiento
    Xs_all = scaler.transform(X_tr)
    idx, _ = train_test_split(np.arange(len(y_tr)), train_size=TSNE_SAMPLE,
                              stratify=y_tr, random_state=SEED)
    Xs, ys = Xs_all[idx], y_tr.to_numpy()[idx]
    meta = f"semilla={SEED}, muestra estratificada n={TSNE_SAMPLE} (de {len(y_tr)}), scikit-learn {sklearn.__version__}"

    FIGURES.mkdir(parents=True, exist_ok=True)
    embeddings = {p: embed(Xs, p) for p in TSNE_PERPLEXITIES}

    fig, ax = plt.subplots(figsize=(7.5, 7))
    scatter(ax, embeddings[TSNE_MAIN_PERPLEXITY], ys,
            f"t-SNE de las 20 variables estandarizadas\nperplexity={TSNE_MAIN_PERPLEXITY}, {meta}")
    ax.legend(title="price_range", markerscale=2, loc="upper right", fontsize=8)
    fig.text(0.5, 0.01, WARNING, ha="center", va="bottom", fontsize=8, color="#B00020")
    fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(FIGURES / "03_tsne_perplexity30.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(TSNE_PERPLEXITIES), figsize=(15, 5.2))
    for a, p in zip(axes, TSNE_PERPLEXITIES):
        scatter(a, embeddings[p], ys, f"perplexity={p}")
    axes[0].legend(title="price_range", markerscale=2, fontsize=8)
    fig.suptitle(f"Sensibilidad a la perplexity ({meta})", fontsize=11)
    fig.text(0.5, 0.01, WARNING, ha="center", va="bottom", fontsize=8, color="#B00020")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95)); fig.savefig(FIGURES / "04_tsne_sensibilidad.png", dpi=130)
    plt.close(fig)

    # Complemento: t-SNE con las 4 variables más correlacionadas con el objetivo.
    # La selección usa solo la partición de entrenamiento (uso exploratorio, no predictivo).
    top4 = X_tr.corrwith(y_tr).abs().sort_values(ascending=False).index[:4].tolist()
    sc4 = StandardScaler().fit(X_tr[top4])
    Z4 = embed(sc4.transform(X_tr[top4])[idx], TSNE_MAIN_PERPLEXITY)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    scatter(ax, Z4, ys, f"t-SNE con solo {', '.join(top4)}\nperplexity={TSNE_MAIN_PERPLEXITY}, semilla={SEED}, n={TSNE_SAMPLE}")
    ax.legend(title="price_range", markerscale=2, loc="upper right", fontsize=8)
    fig.text(0.5, 0.01, WARNING, ha="center", va="bottom", fontsize=8, color="#B00020")
    fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(FIGURES / "05_tsne_top4_variables.png", dpi=140)
    plt.close(fig)

    write_text(DOCS / "tsne.md", f"""# Diagnóstico visual: t-SNE

| Parámetro | Valor |
|---|---|
| Semilla (`random_state`) | {SEED} |
| Perplexity principal / explorada | {TSNE_MAIN_PERPLEXITY} / {', '.join(map(str, TSNE_PERPLEXITIES))} |
| Muestra | {TSNE_SAMPLE} filas estratificadas (250 por clase) tomadas de la partición de entrenamiento |
| Preprocesamiento | `StandardScaler` ajustado solo con entrenamiento |
| Inicialización / iteraciones | `init="pca"`, `max_iter=1000`, `learning_rate="auto"` |
| Versión scikit-learn | {sklearn.__version__} |

Figuras: `reports/figures/03_tsne_perplexity30.png` (20 variables), `04_tsne_sensibilidad.png` (perplexity 10/30/50) y
`05_tsne_top4_variables.png` (solo las 4 variables más correlacionadas con el objetivo en entrenamiento: {', '.join(top4)}).

> **Advertencia de interpretación.** t-SNE es una técnica exploratoria no lineal que prioriza la estructura
> local. Sus ejes carecen de significado, la distancia entre grupos lejanos no es una medida de similitud
> y la imagen **no es una prueba estadística** de que existan clústeres. Cambiar semilla o perplexity
> modifica la forma. Solo genera hipótesis; la evidencia de separabilidad la dan las métricas validadas.

## Lectura prudente

- Con las **20 variables** (figuras 03 y 04) las cuatro clases aparecen mezcladas a simple vista, para las
  tres perplexities exploradas.
- Con solo **4 variables** (figura 05) aparece un gradiente de clases, aunque con zonas solapadas.
- Aun así, la `LogisticRegression` validada con las 20 variables logra un F1 macro alto (ver `docs/baseline.md`).
  Es decir, **las clases sí son en buena medida separables** aunque el t-SNE de las 20 variables no lo muestre:
  la señal se concentra en pocas variables (sobre todo `ram`) y el resto diluye las vecindades euclidianas.
- Moraleja metodológica: ni la presencia ni la ausencia de "clústeres" en t-SNE prueban algo. La decisión
  se apoya en métricas validadas con Pipeline, no en la imagen.
""")
    print("t-SNE listo")


if __name__ == "__main__":
    main()

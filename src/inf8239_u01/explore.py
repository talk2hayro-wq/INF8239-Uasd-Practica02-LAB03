"""Exploración de LAB03: costo de densificar, PCA justificado y t-SNE.

Todo se calcula SOLO con datos de entrenamiento.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from . import lab03_config as cfg
from .features import FEATURES_NUM
from .models import build_preprocess


def memory_table(X_train: pd.DataFrame) -> pd.DataFrame:
    """Mide cuánta memoria ocupa la matriz codificada densa frente a la dispersa.

    Sirve para justificar (o no) densificar: la nota del guion pide medirlo
    antes de convertir a denso.
    """
    dispersa = build_preprocess(dense=False).fit_transform(X_train)
    if not issparse(dispersa):
        dispersa = csr_matrix(dispersa)
    densa = build_preprocess(dense=True).fit_transform(X_train)
    bytes_dispersa = dispersa.data.nbytes + dispersa.indices.nbytes + dispersa.indptr.nbytes
    filas = [
        {"representacion": "dispersa (CSR)", "filas": dispersa.shape[0],
         "columnas": dispersa.shape[1], "MB": bytes_dispersa / 1e6},
        {"representacion": "densa (float64)", "filas": densa.shape[0],
         "columnas": densa.shape[1], "MB": densa.nbytes / 1e6},
    ]
    tabla = pd.DataFrame(filas)
    tabla["MB"] = tabla["MB"].round(3)
    return tabla


def pca_summary(X_train: pd.DataFrame, variance: float = cfg.PCA_VARIANZA) -> dict:
    """Evalúa si PCA tiene sentido aquí y cuánto reduce.

    - Bloque numérico: solo las columnas de FEATURES_NUM (aquí, `dia`). Con una
      sola columna PCA no puede reducir nada, por eso se aplica a la matriz
      codificada completa (categóricas en one-hot + numéricas escaladas).
    - Esa matriz es densa y pequeña (ver memory_table), así que PCA es viable.
    """
    matriz = build_preprocess(dense=True).fit_transform(X_train)
    pca = PCA(n_components=variance, random_state=cfg.RANDOM_STATE).fit(matriz)
    return {
        "columnas_bloque_numerico": len(FEATURES_NUM),
        "columnas_codificadas": int(matriz.shape[1]),
        "componentes_para_varianza": int(pca.n_components_),
        "varianza_objetivo": float(variance),
        "varianza_explicada": float(pca.explained_variance_ratio_.sum()),
        "reduccion_columnas": float(1 - pca.n_components_ / matriz.shape[1]),
    }


def encoded_sample(X_train: pd.DataFrame, y_train: pd.Series,
                   n: int = cfg.TSNE_MUESTRA, random_state: int = cfg.RANDOM_STATE):
    """Muestra de entrenamiento ya codificada (preprocesador ajustado con el entrenamiento)."""
    prep = build_preprocess(dense=True).fit(X_train)
    muestra = X_train.sample(min(n, len(X_train)), random_state=random_state)
    return prep.transform(muestra), y_train.loc[muestra.index].to_numpy(), \
        muestra["aeropuerto"].to_numpy()


def tsne_embeddings(matriz, seeds=cfg.TSNE_SEMILLAS,
                    perplexity: float = cfg.TSNE_PERPLEXITY, init: str = "random") -> dict:
    """Un mapa t-SNE por semilla (mismos datos, distinta semilla).

    Con init="pca" la posición inicial es determinista y la semilla casi no
    interviene: dos semillas dan mapas idénticos. Por eso el análisis principal
    usa init="random" (la semilla sí cambia el resultado) y se mide aparte el
    caso init="pca".
    """
    return {
        seed: TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                   init=init, learning_rate="auto").fit_transform(matriz)
        for seed in seeds
    }


def tsne_stability(emb_a, emb_b, aeropuerto, y, k: int = 10) -> dict:
    """Cuantifica qué permanece y qué cambia entre dos mapas t-SNE.

    - vecinos_compartidos: de los k vecinos de cada punto en un mapa, qué
      fracción sigue siendo vecino en el otro (estructura local que permanece).
    - pureza_*: fracción de los k vecinos que comparten aeropuerto / target.
      Se compara con el nivel esperado por azar.
    - disparidad_procrustes: cuánto difieren los mapas tras rotarlos y
      escalarlos al máximo parecido (0 = idénticos). Captura lo que cambia.
    """
    def vecinos(emb):
        return NearestNeighbors(n_neighbors=k + 1).fit(emb).kneighbors(
            emb, return_distance=False)[:, 1:]

    va, vb = vecinos(emb_a), vecinos(emb_b)
    compartidos = np.mean([len(set(va[i]) & set(vb[i])) / k for i in range(len(va))])

    def pureza(vec, etiquetas):
        return float(np.mean(etiquetas[vec] == etiquetas[:, None]))

    def azar(etiquetas):
        _, conteos = np.unique(etiquetas, return_counts=True)
        return float(np.sum((conteos / conteos.sum()) ** 2))

    aeropuerto, y = np.asarray(aeropuerto), np.asarray(y)
    return {
        "vecinos_compartidos": float(compartidos),
        "pureza_aeropuerto_mapa_a": pureza(va, aeropuerto),
        "pureza_aeropuerto_mapa_b": pureza(vb, aeropuerto),
        "azar_aeropuerto": azar(aeropuerto),
        "pureza_target_mapa_a": pureza(va, y),
        "pureza_target_mapa_b": pureza(vb, y),
        "azar_target": azar(y),
        "disparidad_procrustes": float(procrustes(emb_a, emb_b)[2]),
    }

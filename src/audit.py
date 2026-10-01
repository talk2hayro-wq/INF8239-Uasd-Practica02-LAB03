"""Auditoría de calidad y generación de la Ficha Técnica de Datos.

Uso:  python -m src.audit
Salidas: docs/ficha_tecnica.md, reports/tables/audit_*.csv, reports/figures/01_*.png, 02_*.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .config import (CLASS_LABELS, DOCS, FEATURE_DICT, FIGURES, SEED, TABLES,
                     TARGET, TEST_CSV, TEST_SIZE, TRAIN_CSV)
from .data import feature_columns, load_test, load_train
from .utils import md_table, sha256, write_text

KAGGLE_URL = "https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification/"


def quality_tables(df: pd.DataFrame, te: pd.DataFrame):
    feats = feature_columns(df)
    rows = []
    for c in feats + [TARGET]:
        x = df[c]
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
        desc, unit, kind = FEATURE_DICT[c]
        rows.append({"variable": c, "descripción": desc, "unidad": unit, "tipo": kind,
                     "min": x.min(), "max": x.max(), "media": x.mean(),
                     "ceros": int((x == 0).sum()), "outliers_IQR": n_out,
                     "nulos": int(x.isna().sum())})
    dic = pd.DataFrame(rows)

    # Uniformidad (estadístico KS contra U[min,max]) en variables no binarias
    ks_rows = []
    for c in feats:
        if FEATURE_DICT[c][2] == "binaria":
            continue
        x = df[c].to_numpy(dtype=float)
        u = (x - x.min()) / (x.max() - x.min())
        ks_rows.append({"variable": c, "KS_vs_uniforme": stats.kstest(u, "uniform").statistic})
    ks = pd.DataFrame(ks_rows)

    # Deriva train vs test (KS de dos muestras)
    drift = pd.DataFrame([{"variable": c,
                           "KS_train_vs_test": stats.ks_2samp(df[c], te[c]).statistic,
                           "p_valor": stats.ks_2samp(df[c], te[c]).pvalue} for c in feats])

    corr = df.corr()
    target_corr = corr[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False)
    pairs = []
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            pairs.append((a, b, corr.loc[a, b]))
    top_pairs = pd.DataFrame(sorted(pairs, key=lambda t: -abs(t[2]))[:5],
                             columns=["var_a", "var_b", "r_pearson"])
    return dic, ks, drift, target_corr, top_pairs, corr


def make_figures(df, target_corr, corr):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    counts = df[TARGET].value_counts().sort_index()
    ax[0].bar([f"{k} ({CLASS_LABELS[k]})" for k in counts.index], counts.values, color="#4C78A8")
    ax[0].set_title("Balance de clases (train.csv)"); ax[0].set_ylabel("Observaciones")
    tc = target_corr.iloc[::-1]
    ax[1].barh(tc.index, tc.values, color=["#E45756" if v < 0 else "#4C78A8" for v in tc.values])
    ax[1].set_title(f"Correlación de Pearson con {TARGET}"); ax[1].set_xlabel("r")
    fig.tight_layout(); fig.savefig(FIGURES / "01_balance_y_correlacion.png", dpi=140); plt.close(fig)

    feats = feature_columns(df)
    fig, axes = plt.subplots(4, 5, figsize=(15, 9))
    for a, c in zip(axes.ravel(), feats):
        a.hist(df[c], bins=25, color="#72B7B2"); a.set_title(c, fontsize=9)
        a.tick_params(labelsize=7)
    fig.suptitle("Distribución de las 20 variables predictoras", y=1.0)
    fig.tight_layout(); fig.savefig(FIGURES / "02_distribuciones.png", dpi=120); plt.close(fig)


def build_markdown(df, te, dic, ks, drift, target_corr, top_pairs) -> str:
    feats = feature_columns(df)
    n, p = df.shape[0], len(feats)
    counts = df[TARGET].value_counts().sort_index()
    n_ks_low = int((ks["KS_vs_uniforme"] < 0.06).sum())
    zeros_scw = int((df["sc_w"] == 0).sum()); zeros_pxh = int((df["px_height"] == 0).sum())
    test_id_ok = te["id"].is_unique
    schema_ok = list(te.columns) == ["id"] + feats
    drift_sig = drift[drift["p_valor"] < 0.01]
    dic_show = dic[["variable", "descripción", "unidad", "tipo", "min", "max", "media", "ceros", "outliers_IQR"]].copy()
    for col in ("min", "max", "media"):
        dic_show[col] = dic_show[col].map(lambda v: f"{v:.2f}".rstrip("0").rstrip("."))

    return f"""# Ficha Técnica de Datos: Mobile Price Classification

*Generada automáticamente por `python -m src.audit`; las cifras salen de los datos, no se escriben a mano.*

## 1. Procedencia

| Campo | Detalle |
|---|---|
| Nombre | Mobile Price Classification |
| Autor en Kaggle | Abhishek Sharma (usuario `iabhishekofficial`) |
| Fuente | {KAGGLE_URL} |
| Descripción pública | "Classify Mobile Price Range" |
| Archivos usados | `train.csv` ({n} filas, {p + 1} columnas) y `test.csv` ({len(te)} filas, {len(te.columns)} columnas) |
| SHA-256 `train.csv` | `{sha256(TRAIN_CSV)}` |
| SHA-256 `test.csv` | `{sha256(TEST_CSV)}` |
| Descarga | Manual por el estudiante desde la página indicada |

Los hashes permiten comprobar que los archivos locales son idénticos a los usados en este informe.

## 2. Licencia y condiciones de uso

**Estado: NO VERIFICADA.** Kaggle bloquea la lectura automatizada de su página, de modo que no fue posible
leer el campo *License* de la ficha. **Acción pendiente:** abrir la página de la sección 1, revisar la
etiqueta de licencia en la barra lateral de "Metadata" y copiar aquí su texto exacto.

Mientras se confirma, se aplican estas salvaguardas conservadoras:

- Uso restringido a fines **académicos y no comerciales**, con atribución al autor y a Kaggle.
- Los CSV **no se versionan en Git** (`data/raw/` está en `.gitignore`); se documentan por hash y se
  descargan desde la fuente original.
- Si el archivo se obtuvo desde una competencia de Kaggle y no desde la página del dataset, aplican
  las reglas de esa competencia.

## 3. Unidad de análisis, objetivo y momento de disponibilidad

- **Unidad de análisis:** un modelo de teléfono móvil descrito por sus especificaciones técnicas.
- **Variable objetivo:** `{TARGET}` con 4 clases ordinales (0 bajo, 1 medio, 2 alto, 3 muy alto). Se trata
  como clasificación **multiclase nominal**, lo que ignora el orden entre clases (ver amenazas).
- **Momento de disponibilidad:** las 20 variables son especificaciones de hardware conocidas al diseñar el
  producto y previas a la fijación del precio. **No se identificó ninguna variable posterior a la decisión**
  ni derivada del precio, por lo que no hay fuga por variable objetivo (tipo `duration` en Bank Marketing).
  La fuga que sí puede ocurrir es de **preprocesamiento** (escalado/PCA ajustados con todo el dataset) y se
  neutraliza con `Pipeline` (ver `src/pipeline.py` y `tests/test_pipeline.py`).

## 4. Diccionario de datos

{md_table(dic_show, widths=[13, 34, 7, 9, 6, 6, 7, 6, 10])}

Las unidades y descripciones siguen la documentación pública del dataset en Kaggle y deben contrastarse con
la página de origen.

## 5. Auditoría de calidad

### 5.1 Integridad básica

| Verificación | Resultado |
|---|---|
| Dimensiones `train.csv` | {n} filas x {p + 1} columnas |
| Valores nulos (train / test) | {int(df.isna().sum().sum())} / {int(te.isna().sum().sum())} |
| Filas duplicadas en train | {int(df.duplicated().sum())} |
| Tipos | {int((df.dtypes == 'int64').sum())} enteras, {int((df.dtypes == 'float64').sum())} decimales |
| Esquema `test.csv` = `id` + 20 predictoras | {'Sí' if schema_ok else 'NO'} |
| `id` único en test | {'Sí' if test_id_ok else 'NO'} |
| Etiquetas en `test.csv` | **No**: no se puede usar para evaluar |

### 5.2 Balance de clases

Clases 0/1/2/3: {', '.join(f'{int(v)}' for v in counts.values)} observaciones ({(counts / n * 100).round(1).iloc[0]} % cada una).
El problema está **perfectamente balanceado**, así que `accuracy` y `F1 macro` son prácticamente equivalentes
y `DummyClassifier` obtiene 0.25 por construcción. Se reporta igualmente F1 macro y PR-AUC macro por consistencia
con la guía.

### 5.3 Valores no plausibles y atípicos

- `sc_w` (ancho de pantalla en cm) es **0 en {zeros_scw} filas** ({zeros_scw / n * 100:.1f} %): una pantalla de ancho
  cero no existe; es un valor centinela o dato faltante disfrazado.
- `px_height` es 0 en {zeros_pxh} filas: resolución imposible.
- `fc` (cámara frontal) y `pc` (principal) con 0 Mpx pueden ser legítimos (equipo sin cámara).
- **Decisión:** los datos crudos no se modifican. No se imputa `sc_w`/`px_height` porque cualquier imputación
  con estadísticos del conjunto completo sería fuga; si se imputara, debería hacerse dentro del `Pipeline`.
  El impacto se declara como limitación.
- Atípicos por regla de IQR (1.5 x IQR): ver columna `outliers_IQR` del diccionario. Se conservan.

### 5.4 Relación con el objetivo y colinealidad

Correlación de Pearson con `{TARGET}` (5 mayores en valor absoluto):

{md_table(target_corr.head(5).rename('r_pearson').reset_index().rename(columns={'index': 'variable'}))}

`ram` domina claramente el problema (no es fuga: es un dato de diseño, pero condiciona toda la comparación).
Pares de predictoras más correlacionados:

{md_table(top_pairs)}

### 5.5 Forma de las distribuciones

{n_ks_low} de {len(ks)} variables no binarias tienen un estadístico KS < 0.06 frente a una distribución uniforme
entre su mínimo y su máximo (ver `reports/tables/audit_uniformidad.csv` y la figura 02). Una uniformidad así es poco
habitual en datos de mercado reales y **podría indicar datos curados o sintéticos**, lo que limitaría la
validez externa; es una observación, no una prueba del origen.

### 5.6 Deriva entre `train.csv` y `test.csv`

Prueba KS de dos muestras por variable: {len(drift_sig)} de {len(drift)} variables con p < 0.01
({', '.join(drift_sig['variable']) if len(drift_sig) else 'ninguna'}).
Estadístico KS máximo: {drift['KS_train_vs_test'].max():.3f}. Con las etiquetas ausentes solo se puede verificar
que las distribuciones de entrada sean comparables.

## 6. Partición y decisiones de tratamiento

- `test.csv` **no se usa** para entrenar ni evaluar (sin etiquetas); solo para verificar esquema y deriva.
- Sobre `train.csv` se hace **una única** partición hold-out estratificada {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}
  con `random_state={SEED}`: {int(n * (1 - TEST_SIZE))} filas para entrenamiento/validación cruzada y
  {int(n * TEST_SIZE)} filas reservadas como simulacro de producción.
- Ningún transformador con estado se ajusta antes de la partición.

## 7. Amenazas a la validez

| Tipo | Amenaza |
|---|---|
| Interna | Fuga de preprocesamiento (mitigada con Pipeline); sc_w = 0 sin imputar; una sola partición hold-out. |
| Externa | Dataset posiblemente curado/sintético y de un único dominio; 2000 filas; resultados no generalizan a precios reales. |
| Constructo | `price_range` es una categoría ordinal discretizada, tratada como nominal; tiempo y memoria son proxies de sostenibilidad, no de energía. |
| Conclusión | Con 400 observaciones de prueba, diferencias de F1 menores a ~0.01 no son concluyentes; se apoya la decisión en validación cruzada. |
"""


def main():
    df, te = load_train(), load_test()
    dic, ks, drift, target_corr, top_pairs, corr = quality_tables(df, te)
    TABLES.mkdir(parents=True, exist_ok=True)
    dic.to_csv(TABLES / "audit_diccionario.csv", index=False, encoding="utf-8")
    ks.to_csv(TABLES / "audit_uniformidad.csv", index=False, encoding="utf-8")
    drift.to_csv(TABLES / "audit_deriva_train_test.csv", index=False, encoding="utf-8")
    make_figures(df, target_corr, corr)
    write_text(DOCS / "ficha_tecnica.md", build_markdown(df, te, dic, ks, drift, target_corr, top_pairs))
    print("Ficha técnica generada en docs/ficha_tecnica.md")


if __name__ == "__main__":
    main()

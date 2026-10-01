# Diagnóstico visual: t-SNE

| Parámetro | Valor |
|---|---|
| Semilla (`random_state`) | 42 |
| Perplexity principal / explorada | 30 / 10, 30, 50 |
| Muestra | 1000 filas estratificadas (250 por clase) tomadas de la partición de entrenamiento |
| Preprocesamiento | `StandardScaler` ajustado solo con entrenamiento |
| Inicialización / iteraciones | `init="pca"`, `max_iter=1000`, `learning_rate="auto"` |
| Versión scikit-learn | 1.9.1 |

Figuras: `reports/figures/03_tsne_perplexity30.png` (20 variables), `04_tsne_sensibilidad.png` (perplexity 10/30/50) y
`05_tsne_top4_variables.png` (solo las 4 variables más correlacionadas con el objetivo en entrenamiento: ram, battery_power, px_width, px_height).

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

# Ficha Técnica de Datos: Mobile Price Classification

*Generada automáticamente por `python -m src.audit`; las cifras salen de los datos, no se escriben a mano.*

## 1. Procedencia

| Campo | Detalle |
|---|---|
| Nombre | Mobile Price Classification |
| Autor en Kaggle | Abhishek Sharma (usuario `iabhishekofficial`) |
| Fuente | https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification/ |
| Descripción pública | "Classify Mobile Price Range" |
| Archivos usados | `train.csv` (2000 filas, 21 columnas) y `test.csv` (1000 filas, 21 columnas) |
| SHA-256 `train.csv` | `f9e8cd3154b8684a1be0ff401b1fba7750710af6bd076a399c6ce7ceb979c624` |
| SHA-256 `test.csv` | `e929f5999ffeedec84abf53cc8d35faa1bbd7e8ebd61826881aa62562d37adce` |
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
- **Variable objetivo:** `price_range` con 4 clases ordinales (0 bajo, 1 medio, 2 alto, 3 muy alto). Se trata
  como clasificación **multiclase nominal**, lo que ignora el orden entre clases (ver amenazas).
- **Momento de disponibilidad:** las 20 variables son especificaciones de hardware conocidas al diseñar el
  producto y previas a la fijación del precio. **No se identificó ninguna variable posterior a la decisión**
  ni derivada del precio, por lo que no hay fuga por variable objetivo (tipo `duration` en Bank Marketing).
  La fuga que sí puede ocurrir es de **preprocesamiento** (escalado/PCA ajustados con todo el dataset) y se
  neutraliza con `Pipeline` (ver `src/pipeline.py` y `tests/test_pipeline.py`).

## 4. Diccionario de datos

| variable | descripción | unidad | tipo | min | max | media | ceros | outliers_IQR |
|-------------|----------------------------------|-------|---------|------|------|-------|------|----------|
| battery_power | Energía total que la batería puede almacenar | mAh | entera | 501 | 1998 | 1238.52 | 0 | 0 |
| blue | Tiene Bluetooth | 0/1 | binaria | 0 | 1 | 0.49 | 1010 | 0 |
| clock_speed | Velocidad a la que el microprocesador ejecuta instrucciones | GHz | continua | 0.5 | 3 | 1.52 | 0 | 0 |
| dual_sim | Soporta doble SIM | 0/1 | binaria | 0 | 1 | 0.51 | 981 | 0 |
| fc | Resolución de la cámara frontal | Mpx | entera | 0 | 19 | 4.31 | 474 | 18 |
| four_g | Tiene 4G | 0/1 | binaria | 0 | 1 | 0.52 | 957 | 0 |
| int_memory | Memoria interna | GB | entera | 2 | 64 | 32.05 | 0 | 0 |
| m_dep | Profundidad (grosor) del móvil | cm | continua | 0.1 | 1 | 0.5 | 0 | 0 |
| mobile_wt | Peso del móvil | g | entera | 80 | 200 | 140.25 | 0 | 0 |
| n_cores | Número de núcleos del procesador | conteo | entera | 1 | 8 | 4.52 | 0 | 0 |
| pc | Resolución de la cámara principal | Mpx | entera | 0 | 20 | 9.92 | 101 | 0 |
| px_height | Alto de la resolución de pantalla | px | entera | 0 | 1960 | 645.11 | 2 | 2 |
| px_width | Ancho de la resolución de pantalla | px | entera | 500 | 1998 | 1251.52 | 0 | 0 |
| ram | Memoria RAM | MB | entera | 256 | 3998 | 2124.21 | 0 | 0 |
| sc_h | Alto de la pantalla | cm | entera | 5 | 19 | 12.31 | 0 | 0 |
| sc_w | Ancho de la pantalla | cm | entera | 0 | 18 | 5.77 | 180 | 0 |
| talk_time | Duración máxima de una carga en llamada | horas | entera | 2 | 20 | 11.01 | 0 | 0 |
| three_g | Tiene 3G | 0/1 | binaria | 0 | 1 | 0.76 | 477 | 0 |
| touch_screen | Pantalla táctil | 0/1 | binaria | 0 | 1 | 0.5 | 994 | 0 |
| wifi | Tiene WiFi | 0/1 | binaria | 0 | 1 | 0.51 | 986 | 0 |
| price_range | Rango de precio (0 bajo, 1 medio, 2 alto, 3 muy alto) | clase | objetivo ordinal | 0 | 3 | 1.5 | 500 | 0 |

Las unidades y descripciones siguen la documentación pública del dataset en Kaggle y deben contrastarse con
la página de origen.

## 5. Auditoría de calidad

### 5.1 Integridad básica

| Verificación | Resultado |
|---|---|
| Dimensiones `train.csv` | 2000 filas x 21 columnas |
| Valores nulos (train / test) | 0 / 0 |
| Filas duplicadas en train | 0 |
| Tipos | 19 enteras, 2 decimales |
| Esquema `test.csv` = `id` + 20 predictoras | Sí |
| `id` único en test | Sí |
| Etiquetas en `test.csv` | **No**: no se puede usar para evaluar |

### 5.2 Balance de clases

Clases 0/1/2/3: 500, 500, 500, 500 observaciones (25.0 % cada una).
El problema está **perfectamente balanceado**, así que `accuracy` y `F1 macro` son prácticamente equivalentes
y `DummyClassifier` obtiene 0.25 por construcción. Se reporta igualmente F1 macro y PR-AUC macro por consistencia
con la guía.

### 5.3 Valores no plausibles y atípicos

- `sc_w` (ancho de pantalla en cm) es **0 en 180 filas** (9.0 %): una pantalla de ancho
  cero no existe; es un valor centinela o dato faltante disfrazado.
- `px_height` es 0 en 2 filas: resolución imposible.
- `fc` (cámara frontal) y `pc` (principal) con 0 Mpx pueden ser legítimos (equipo sin cámara).
- **Decisión:** los datos crudos no se modifican. No se imputa `sc_w`/`px_height` porque cualquier imputación
  con estadísticos del conjunto completo sería fuga; si se imputara, debería hacerse dentro del `Pipeline`.
  El impacto se declara como limitación.
- Atípicos por regla de IQR (1.5 x IQR): ver columna `outliers_IQR` del diccionario. Se conservan.

### 5.4 Relación con el objetivo y colinealidad

Correlación de Pearson con `price_range` (5 mayores en valor absoluto):

| variable | r_pearson |
|---|---|
| ram | 0.9170 |
| battery_power | 0.2007 |
| px_width | 0.1658 |
| px_height | 0.1489 |
| int_memory | 0.0444 |

`ram` domina claramente el problema (no es fuga: es un dato de diseño, pero condiciona toda la comparación).
Pares de predictoras más correlacionados:

| var_a | var_b | r_pearson |
|---|---|---|
| fc | pc | 0.6446 |
| four_g | three_g | 0.5842 |
| px_height | px_width | 0.5107 |
| sc_h | sc_w | 0.5061 |
| px_height | sc_h | 0.0596 |

### 5.5 Forma de las distribuciones

7 de 14 variables no binarias tienen un estadístico KS < 0.06 frente a una distribución uniforme
entre su mínimo y su máximo (ver `reports/tables/audit_uniformidad.csv` y la figura 02). Una uniformidad así es poco
habitual en datos de mercado reales y **podría indicar datos curados o sintéticos**, lo que limitaría la
validez externa; es una observación, no una prueba del origen.

### 5.6 Deriva entre `train.csv` y `test.csv`

Prueba KS de dos muestras por variable: 0 de 20 variables con p < 0.01
(ninguna).
Estadístico KS máximo: 0.057. Con las etiquetas ausentes solo se puede verificar
que las distribuciones de entrada sean comparables.

## 6. Partición y decisiones de tratamiento

- `test.csv` **no se usa** para entrenar ni evaluar (sin etiquetas); solo para verificar esquema y deriva.
- Sobre `train.csv` se hace **una única** partición hold-out estratificada 80/20
  con `random_state=42`: 1600 filas para entrenamiento/validación cruzada y
  400 filas reservadas como simulacro de producción.
- Ningún transformador con estado se ajusta antes de la partición.

## 7. Amenazas a la validez

| Tipo | Amenaza |
|---|---|
| Interna | Fuga de preprocesamiento (mitigada con Pipeline); sc_w = 0 sin imputar; una sola partición hold-out. |
| Externa | Dataset posiblemente curado/sintético y de un único dominio; 2000 filas; resultados no generalizan a precios reales. |
| Constructo | `price_range` es una categoría ordinal discretizada, tratada como nominal; tiempo y memoria son proxies de sostenibilidad, no de energía. |
| Conclusión | Con 400 observaciones de prueba, diferencias de F1 menores a ~0.01 no son concluyentes; se apoya la decisión en validación cruzada. |

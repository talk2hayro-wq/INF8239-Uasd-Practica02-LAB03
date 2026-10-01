# Ficha de procedencia y licencia del dataset

| Campo | Valor |
|---|---|
| **Nombre** | Movimiento de Entrada y Salida de Pasajeros por Aeropuertos, 2018-2026 |
| **Entidad productora** | Junta de Aviación Civil (JAC), República Dominicana |
| **URL de descarga** | https://jac.gob.do/wp-content/uploads/2026/07/Movimiento-de-Entrada-y-Salida-de-Pasajeros-por-Aeropuertos-Junta-de-Aviacion-Civil-JAC-2018-2026.csv |
| **Otros canales** | La JAC publica versiones anteriores (2018-2024 y 2018-2025) del mismo conjunto de datos en el portal nacional de datos abiertos, datos.gob.do |
| **Fecha de acceso** | 2026-09-29 |
| **Formato** | CSV, codificación UTF-8 con BOM, separador coma |
| **Tamaño del original** | 834 358 bytes |
| **SHA-256 del original** | `96def79bc38b61237c5bba76fd63dbc078b388c9792319f3a557388ab8cc412d` |
| **Filas / columnas** | 19 971 filas × 6 columnas |
| **Cobertura temporal** | 2018-01-01 a 2026-06-30 |
| **Cobertura geográfica** | 7 aeropuertos internacionales de República Dominicana |
| **Granularidad** | Un registro por aeropuerto y día (los aeropuertos pequeños omiten días sin operaciones) |
| **Datos personales** | No contiene datos personales: son conteos agregados diarios |

## Cómo se obtuvo

`download_csv(URL_JAC)` (módulo `src/inf8239_u01/data.py`) descarga el CSV con `pandas.read_csv` y guarda una copia en `data/raw/dataset.csv`. Esa copia se vuelve a serializar con pandas (sin BOM), por lo que su hash es distinto al del archivo original:

| Archivo | SHA-256 |
|---|---|
| Original descargado del sitio de la JAC | `96def79bc38b61237c5bba76fd63dbc078b388c9792319f3a557388ab8cc412d` |
| Copia de trabajo `data/raw/dataset.csv` (814 383 bytes) | `a5ecfb704a24bf9362c70d065701ccbc27d4e75bc9b27a3e1cdd07e213b0e712` |

> **Nota de reproducibilidad.** Las corridas de las evidencias de este repositorio se hicieron a partir de una copia local del mismo CSV, porque el entorno donde se generaron no tenía acceso a `jac.gob.do`. La copia de trabajo se produjo con el mismo procedimiento que `download_csv` (`read_csv` + `to_csv(index=False)`). En una máquina con Internet, `python pract02.py` descarga el archivo por sí mismo.

## Licencia y condiciones de uso

- **Licencia explícita: no verificada.** Ni el archivo ni las páginas consultadas de la JAC declaran una licencia. Antes de redistribuir el dataset o publicarlo fuera del curso, debe confirmarse con la JAC o en la ficha del conjunto de datos en datos.gob.do.
- **Uso en este proyecto:** exclusivamente académico (curso INF-8239), con atribución a la fuente.
- **Por eso `data/raw/` no se versiona** (ver `.gitignore`): el repositorio no redistribuye el CSV; quien lo clone lo descarga desde la fuente.
- **Atribución sugerida:** «Junta de Aviación Civil (JAC), República Dominicana. Movimiento de Entrada y Salida de Pasajeros por Aeropuertos, 2018-2026. Consultado el 2026-09-29.»

## Calidad y advertencias de la fuente

- **Los datos pueden cambiar.** Los informes estadísticos de la JAC indican que sus datos son colectados en una fecha y están sujetos a cambios; por eso se registran fecha de acceso y hash.
- **Verificación cruzada.** La suma de 2018 en el archivo (14 488 668 pasajeros) coincide con la cifra anual divulgada por la JAC para 2018 (14 487 503), con una diferencia de 0.008 %.
- **Nombres de aeropuertos cambian en 2019.** Se armonizaron 11 nombres a 7 aeropuertos (ver `docs/DICCIONARIO_DATOS.md`). Las equivalencias son coherentes con que los nombres no se solapan en años y con las rutas citadas por la JAC («Nueva York / Las Américas JFPG», «Nueva York / Del Cibao»); la de *La Isabela = El Higüero* se basa en el mismo criterio y en la continuidad de los volúmenes.
- **El guion `-` en `Entrada`/`Salida`** (736 y 586 veces) se interpretó como cero operaciones: el mínimo numérico del archivo es 1 y nunca aparece un 0 explícito.
- **Cambio de régimen 2020-2021** por la pandemia (total anual: 14.4 M en 2019, 5.5 M en 2020, 10.8 M en 2021). Es un dato real, no un error.
- **2026 es un año parcial** (hasta el 30 de junio).

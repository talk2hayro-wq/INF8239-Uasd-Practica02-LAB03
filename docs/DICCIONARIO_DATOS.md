# Diccionario de datos

Unidad de análisis: **un aeropuerto en un día** (fila aeropuerto-día). Dataset limpio: 19 971 filas, 7 aeropuertos, 2018-01-01 a 2026-06-30, sin duplicados aeropuerto-fecha.

## 1. Columnas del CSV original

| Columna original | Tipo original | Significado | Observaciones |
|---|---|---|---|
| `Aerop. de Entrada` | texto | Aeropuerto donde se registró el movimiento | 11 nombres distintos (ver sección 3) |
| `Entrada` | texto | Pasajeros que llegan al país ese día en ese aeropuerto | Contiene `" -   "` (736 veces) = sin operaciones |
| `Salida` | texto | Pasajeros que salen del país ese día en ese aeropuerto | Contiene `" -   "` (586 veces) = sin operaciones |
| `Dia` | entero | Día del mes (1-31) | |
| `Mes` | texto | Nombre del mes en español («Enero» … «Diciembre») | Se convierte a 1-12 |
| `Año` | entero | Año (2018-2026) | |

## 2. Columnas del dataset limpio y de modelado

| Columna | Tipo | Origen | Rol en el modelo | Justificación |
|---|---|---|---|---|
| `fecha` | fecha | `Año` + `Mes` + `Dia` | **No es variable** | Solo ordena y parte en el tiempo; usarla como número extrapolaría fuera del rango de entrenamiento |
| `aeropuerto` | texto (7 valores) | `Aerop. de Entrada` armonizado | **Variable** (categórica) | Se conoce antes del día a predecir. No es un identificador de fila: es una categoría con 7 valores que se repiten miles de veces |
| `entrada` | entero | `Entrada` | **Excluida (fuga)** | Es parte del resultado del día que se quiere anticipar |
| `salida` | entero | `Salida` | **Excluida (fuga)** | Ídem |
| `total` | entero | `entrada + salida` | **Excluida (fuga)** | Con ella se calcula el target |
| `dia` | entero 1-31 | `Dia` | **Variable** (numérica) | Calendario, conocido de antemano |
| `mes` | entero 1-12 | `Mes` | **Variable** (categórica) | Captura estacionalidad (temporada alta) |
| `dia_semana` | entero 0-6 (0 = lunes) | derivada de `fecha` | **Variable** (categórica) | Captura el patrón semanal de vuelos |
| `anio` | entero | `Año` | **Excluida** | El target ya es relativo al año; en el test (2025-2026) son años no vistos y un número no extrapola |
| `umbral_p75` | decimal | percentil 75 de `total` por aeropuerto y año | **Excluida** | Derivada del target |
| `alta_demanda` | entero {0, 1} | `total > umbral_p75` | **TARGET** | 1 = día de alta demanda respecto a su aeropuerto y año; ≈ 25 % de positivos |

Identificadores: el dataset **no tiene columna identificadora** de fila (no hay ID de vuelo, pasajero ni registro). La clave natural (`aeropuerto`, `fecha`) es única y no entra al modelo como identificador.

## 3. Armonización de nombres de aeropuerto

| Nombre en el CSV | Aeropuerto armonizado | Años en que aparece |
|---|---|---|
| `Del Cibao` | Santiago | 2018 |
| `Santiago` | Santiago | 2019-2026 |
| `Las Américas, JFPG` | Santo Domingo - Las Américas | 2018 |
| `Santo Domingo - Las Américas` | Santo Domingo - Las Américas | 2019-2026 |
| `El Catey, Samaná` | Samaná - El Catey | 2018 |
| `Samaná - El Catey` | Samaná - El Catey | 2019-2026 |
| `La Isabela` | Santo Domingo - El Higüero | 2018 |
| `Santo Domingo - El Higüero` | Santo Domingo - El Higüero | 2019-2026 |
| `La Romana` | La Romana | 2018-2026 |
| `Puerto Plata` | Puerto Plata | 2018-2026 |
| `Punta Cana` | Punta Cana | 2018-2026 |

## 4. Resumen por aeropuerto (dataset limpio)

| Aeropuerto | Filas | Total diario medio | Total diario máximo |
|---|---|---|---|
| Santo Domingo - Las Américas | 3 103 | 11 882 | 22 790 |
| Santo Domingo - El Higüero | 3 097 | 168 | 2 287 |
| Santiago | 3 075 | 5 052 | 8 955 |
| Punta Cana | 3 057 | 21 612 | 49 772 |
| Puerto Plata | 2 984 | 1 982 | 6 052 |
| La Romana | 2 824 | 679 | 4 528 |
| Samaná - El Catey | 1 831 | 475 | 2 176 |

Samaná - El Catey tiene menos filas porque no reporta los días sin operaciones.

## 5. Valores especiales y decisiones

| Situación | Cantidad | Decisión |
|---|---|---|
| `Entrada` = `-` | 736 | Se convierte a 0 |
| `Salida` = `-` | 586 | Se convierte a 0 |
| Días con `total` = 0 | 146 | Se conservan (son días de baja demanda reales; caen en la clase 0) |
| Filas eliminadas | 0 | La limpieza no descarta registros |

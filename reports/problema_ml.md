# Problema de machine learning — Titanic

## 1. Definición del problema (Paso 39)

| Decisión | Respuesta | Justificación |
|---|---|---|
| Tipo de problema | Clasificación binaria supervisada | El objetivo tiene dos valores conocidos (0/1) para cada pasajero etiquetado. |
| Variable objetivo | `Survived` (1 = sobrevivió) | Es lo que la aseguradora quiere estimar. |
| Unidad y momento de predicción | Un pasajero, al embarcar | Solo se usa información disponible antes del naufragio. |
| Métrica principal | ROC-AUC | No depende del umbral ni se deja engañar por el desbalance (≈ 38 % de positivos); mide qué tan bien el modelo ordena el riesgo. |
| Métricas secundarias | Precisión, recall, F1, exactitud | Describen los errores una vez elegido un umbral; la exactitud sola engaña (el ingenuo ya acierta ≈ 62 %). |
| Costo de errores | Es peor predecir "sobrevive" a quien no sobrevivió (falso positivo) | Para una aseguradora, subestimar el riesgo lleva a primas insuficientes y protocolos que dejan gente atrás. |
| Criterio de éxito | Superar el baseline de género por un margen mayor que la desviación entre pliegues | Una mejora menor que esa variabilidad no se distingue del azar. |

## 2. Baselines (Paso 42)

| Baseline | Exactitud | Precisión | Recall | F1 |
|---|---|---|---|---|
| Ingenuo (`most_frequent`) | ___ | 0.000 | 0.000 | 0.000 |
| Regla de género | ___ | ___ | ___ | ___ |

**Criterio de éxito:** superar a la regla de género (exactitud ≈ ___, F1 ≈ ___).

## 3. Selección de modelos (Paso 50)
_Se completa en la Parte 12._

## 4. Limitaciones
_Se completa en la Parte 16._
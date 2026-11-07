# Problema de machine learning — Titanic

**Autor:** Samuel Acevedo Mambuscay

---

## 1. Definición del problema (Paso 39)

| Decisión | Respuesta | Justificación |
|---|---|---|
| Tipo de problema | Clasificación binaria supervisada | El objetivo tiene dos valores conocidos (0/1) para cada pasajero etiquetado. |
| Variable objetivo | `Survived` (1 = sobrevivió) | Es lo que la aseguradora quiere estimar. |
| Unidad y momento de predicción | Un pasajero, al embarcar | Solo se usa información disponible antes del naufragio. |
| Métrica principal | ROC-AUC | No depende del umbral ni se deja engañar por el desbalance (≈ 38 % de positivos); mide qué tan bien el modelo ordena el riesgo. |
| Métricas secundarias | Precisión, recall, F1, exactitud | Describen los errores una vez elegido un umbral; la exactitud sola engaña (el modelo ingenuo ya acierta ≈ 62 %). |
| Costo de errores | Es peor predecir "sobrevive" a quien no sobrevivió (falso positivo) | Para una aseguradora, subestimar el riesgo lleva a primas insuficientes y a protocolos que dejan gente atrás; sobrestimarlo cuesta dinero, pero no vidas. |
| Criterio de éxito | Superar el baseline de género por un margen mayor que la desviación entre pliegues | Una mejora menor que esa variabilidad no se distingue del azar. |

## 2. Baselines (Paso 42)

Evaluados con validación cruzada estratificada de 5 pliegues sobre el conjunto de entrenamiento (712 pasajeros).

| Baseline | Exactitud | F1 |
|---|---|---|
| Ingenuo (siempre "no sobrevive") | 0.617 | 0.000 |
| Regla de género ("sobrevive si es mujer") | 0.789 | 0.714 |

**Criterio de éxito:** superar a la regla de género (exactitud 0.789, F1 0.714 en validación cruzada).

## 3. Selección de modelos (Pasos 47 a 50)

Resultados de la validación cruzada (5 pliegues, umbral 0.5):

| Modelo | ROC-AUC | Desv. | Exactitud | F1 | ROC-AUC entrenamiento | Brecha |
|---|---|---|---|---|---|---|
| hgb | **0.887** | 0.019 | 0.820 | 0.755 | 0.994 | 0.107 |
| forest | 0.866 | 0.016 | 0.803 | 0.736 | 0.998 | 0.133 |
| logreg | 0.858 | 0.024 | 0.792 | 0.720 | 0.865 | 0.006 |
| tree | 0.847 | 0.026 | 0.805 | 0.719 | 0.893 | 0.046 |
| knn | 0.841 | 0.017 | 0.787 | 0.710 | 0.928 | 0.087 |
| dummy | 0.500 | 0.000 | 0.617 | 0.000 | 0.500 | 0.000 |

**Finalistas:**

- **HistGradientBoosting (`hgb`):** el mejor ROC-AUC (0.887) y la mejor exactitud y F1, con desviación baja (0.019). Su brecha de 0.107 indica sobreajuste moderado, que se controla con regularización (`l2_regularization`) y profundidad limitada. Costo computacional bajo.
- **Random forest (`forest`):** segundo en ROC-AUC (0.866) y el más estable (desviación 0.016), pero con la mayor brecha (0.133): casi memoriza el entrenamiento.
- **Regresión logística (`logreg`):** algo por debajo (0.858), pero prácticamente sin sobreajuste (brecha 0.006) y la más interpretable, gracias a sus coeficientes y *odds ratios*. Es la referencia para explicar el modelo.

Todos los modelos superan con claridad al baseline ingenuo. El árbol simple y k vecinos quedaron fuera por menor ROC-AUC y, en el caso de k vecinos, mayor brecha.

**Decisiones posteriores:**

- **Hiperparámetros (Paso 57):** `GridSearchCV` eligió para `hgb` `learning_rate=0.1`, `max_depth=4`, `max_iter=200` y `l2_regularization=1.0`.
- **Umbral (Paso 53):** 0.61, que maximiza el F1 fuera de muestra (0.759 frente a 0.755 con 0.5) y sube la precisión de 0.791 a 0.840, en línea con el costo de errores definido.
- **Variables de cabina (Paso 61):** se excluyen del modelo final por el posible artefacto de registro (la cabina pudo registrarse más para los sobrevivientes), aunque mejoren la métrica.

## 4. Evaluación final y limitaciones (Pasos 62 y 63)

### Resultados en el conjunto de prueba (179 pasajeros, evaluado una sola vez)

| Modelo | Exactitud CV | F1 CV | Exactitud prueba | F1 prueba |
|---|---|---|---|---|
| Ingenuo | 0.617 | 0.000 | 0.615 | 0.000 |
| Regla de género | 0.789 | 0.714 | 0.777 | 0.692 |
| **Modelo final (hgb)** | **0.820** | **0.755** | **0.788** | 0.683 |

ROC-AUC del modelo final: **0.887 ± 0.019** en validación cruzada y **0.824** en prueba (IC 95 % por bootstrap: 0.747 – 0.894).

**Coherencia:** el modelo cumple el criterio de éxito en validación cruzada. En prueba supera a la regla de género en exactitud (+0.011), pero queda ligeramente por debajo en F1 (−0.009), porque el umbral 0.61 sacrifica recall para ganar precisión. Ambas diferencias son mucho menores que el ancho del intervalo de confianza (≈ 0.15), así que en prueba **el modelo y la regla de género no se distinguen de forma concluyente** en exactitud ni en F1. La ventaja del modelo está en ordenar el riesgo (ROC-AUC) y en la precisión. La caída de ROC-AUC entre validación cruzada y prueba es coherente con el sobreajuste moderado y con el tamaño reducido de la prueba.

### Limitaciones

- **Tamaño de muestra:** 891 pasajeros etiquetados y 179 de prueba; el intervalo de confianza es amplio y las diferencias pequeñas entre modelos no son concluyentes.
- **Un solo suceso histórico:** el modelo aprende las reglas de evacuación de 1912 y la estructura social del barco; no se generaliza a otros eventos.
- **Datos faltantes:** el 19.9 % de las edades se imputa dentro del pipeline y el 77.1 % de las cabinas es desconocido (por eso se excluyó la cabina).
- **Desempeño desigual por subgrupos:** en el Paso 54 el recall fue de 0.888 para mujeres, pero solo de 0.353 para hombres y de 0.495 en 3.ª clase. El modelo reconoce mal a los sobrevivientes de los grupos con menor tasa de supervivencia.
- **Variables sensibles:** usa `Sex` y `Pclass` (y `TTitulo`, que codifica el sexo), que dominaron las decisiones de evacuación de 1912.

### Contextos de uso

- **Aceptable:** análisis académico e histórico del naufragio, y como ejemplo de un flujo de machine learning reproducible y sin fuga de datos.
- **No aceptable:** fijar primas, aprobar o negar seguros, priorizar personas en emergencias actuales o cualquier decisión que afecte a personas reales. Que el modelo prediga bien lo que ocurrió en 1912 no significa que esas variables sean apropiadas para decidir sobre alguien hoy.
# Ficha del modelo — `titanic_model_v1`

**Autor:** Samuel Acevedo Mambuscay
**Fecha de entrenamiento:** 7 de noviembre de 2026
**Archivos:** `models/titanic_model_v1.joblib` y `models/titanic_model_v1.metadata.json`

---

## 1. Propósito y uso previsto

Modelo de clasificación binaria que estima la probabilidad de que un pasajero del Titanic sobreviviera, a partir de la información disponible al momento de embarcar.

- **Uso previsto:** académico y exploratorio. Sirve para estudiar qué características se asociaron con la supervivencia en el naufragio de 1912 y como ejemplo de un flujo de machine learning reproducible.
- **Usuarios previstos:** analistas de datos y personas que revisen el proyecto del diplomado.
- **Salida:** probabilidad de supervivencia (`proba_survive`) y predicción binaria (`pred`) con un umbral de 0.61.

## 2. Datos de entrenamiento

| Aspecto | Detalle |
|---|---|
| Fuente | Dataset del Titanic del curso (capa Bronze, `data/bronze/titanic_labeled.csv`) |
| Capa usada | Silver (`data/silver/titanic_silver.parquet`) |
| Huella de los datos | SHA-256 registrada en el campo `data_sha256` de `titanic_model_v1.metadata.json` |
| Pasajeros etiquetados | 891 (≈ 38 % sobrevivientes) |
| Evaluación | División estratificada 80 / 20: 712 de entrenamiento y 179 de prueba |
| Modelo final | Reentrenado con los 891 pasajeros etiquetados |
| Periodo | Un único evento: el naufragio del Titanic (abril de 1912) |

## 3. Variables

**Usadas:**

- Numéricas: `Age` (edad original, imputada dentro del pipeline), `SibSp`, `Parch`, `Fare`, `Familia`.
- Categóricas (*one-hot*): `Pclass`, `Sex`, `Embarked`, `TTitulo`, `Alone`.

**Excluidas:**

| Variable | Motivo |
|---|---|
| `Survived`, `Survivor` | Son el objetivo (fuga directa) |
| `PassengerId`, `Ticket`, `Name` | Identificadores: memorizan, no generalizan (el título se usa vía `TTitulo`) |
| `NSex`, `NEmbarked`, `NCabin`, `NTitulo` | Duplican variables con un orden numérico arbitrario |
| `Age` imputada en Silver, `AgeImputada` | Calculadas con todos los datos: provocarían fuga |
| `Cabin`, `TCabin`, `HasCabin` | Posible artefacto del registro (la cabina pudo registrarse más para los sobrevivientes) |
| `Person` | Redundante con `Age` y `Sex` |

## 4. Modelo y decisiones

| Aspecto | Detalle |
|---|---|
| Algoritmo | `HistGradientBoostingClassifier` (scikit-learn) |
| Hiperparámetros | `learning_rate=0.1`, `max_depth=4`, `max_iter=200`, `l2_regularization=1.0` (búsqueda con `GridSearchCV`, Paso 57) |
| Preprocesamiento | `GroupAgeImputer` (edad por sexo y clase) → mediana y estandarización para numéricas → imputación por moda y *one-hot* para categóricas |
| Umbral de decisión | 0.61 |
| Semilla | 42 |

- **Selección del algoritmo:** en validación cruzada fue el modelo con mayor ROC-AUC (0.887), por delante de random forest (0.866) y regresión logística (0.858). Su brecha entre entrenamiento y validación (0.107) indica cierto sobreajuste, aunque menor que la de random forest (0.133).
- **Umbral 0.61:** maximiza el F1 en las predicciones fuera de muestra (0.759 frente a 0.755 con 0.5) y sube la precisión de 0.791 a 0.840. Se prefirió porque el error más costoso para una aseguradora es predecir "sobrevive" a quien no sobrevivió.
- Todo el preprocesamiento está dentro del `Pipeline` y se ajusta solo con los datos de entrenamiento, por lo que no hay fuga de información del conjunto de prueba.

## 5. Desempeño

### Validación cruzada (5 pliegues estratificados, sobre el entrenamiento, umbral 0.5)

| Métrica | Valor |
|---|---|
| ROC-AUC | 0.887 ± 0.019 |
| Exactitud | 0.820 |
| Precisión | 0.796 |
| Recall | 0.722 |
| F1 | 0.755 |

### Conjunto de prueba (179 pasajeros, evaluado una sola vez, umbral 0.61)

| Métrica | Valor |
|---|---|
| **ROC-AUC** | **0.824** (IC 95 % por bootstrap: 0.747 – 0.894) |
| Exactitud | 0.788 |
| F1 | 0.683 |

### Comparación con los baselines

| Modelo | Exactitud CV | F1 CV | Exactitud prueba | F1 prueba |
|---|---|---|---|---|
| Ingenuo (siempre "no sobrevive") | 0.617 | 0.000 | 0.615 | 0.000 |
| Regla de género ("sobrevive si es mujer") | 0.789 | 0.714 | 0.777 | 0.692 |
| **Modelo (hgb)** | **0.820** | **0.755** | **0.788** | 0.683 |

**Lectura:**

- En validación cruzada el modelo supera con claridad a ambos baselines.
- En prueba supera a la regla de género en exactitud (+0.011), pero queda ligeramente por debajo en F1 (−0.009). El umbral 0.61 reduce el recall a cambio de más precisión, a propósito. Con 179 pasajeros estas diferencias son mucho menores que el ancho del intervalo de confianza (≈ 0.15), así que **en prueba el modelo y la regla de género no se distinguen de forma concluyente en exactitud ni en F1**.
- La principal ventaja del modelo está en **ordenar el riesgo** (ROC-AUC 0.824 frente a una regla binaria que solo separa en dos grupos) y en su **mayor precisión**, que es la prioridad del caso.
- La caída de ROC-AUC entre validación cruzada (0.887) y prueba (0.824) es mayor que la desviación entre pliegues, coherente con el sobreajuste moderado observado en el Paso 49 y con la variabilidad de una prueba pequeña.

### Desempeño por subgrupos (Paso 54, predicciones fuera de muestra sobre el entrenamiento, umbral 0.5)

| Subgrupo | n | Exactitud | Recall |
|---|---|---|---|
| Mujeres | 253 | 0.806 | 0.888 |
| Hombres | 459 | 0.828 | 0.353 |
| 1.ª clase | 171 | 0.789 | 0.820 |
| 2.ª clase | 150 | 0.913 | 0.881 |
| 3.ª clase | 391 | 0.798 | 0.495 |
| Edad 0–15 | 76 | 0.868 | 0.884 |
| Edad 16–39 | 378 | 0.796 | 0.688 |
| Edad 40–59 | 105 | 0.800 | 0.698 |
| Edad 60+ | 16 | 0.938 | 0.667 |

Los grupos de edad suman 575 pasajeros porque los 137 sin edad registrada no se asignan a ningún grupo.

**Lectura:**

- La exactitud es parecida en todos los grupos (0.79–0.94), pero el **recall** muestra dónde falla el modelo: identifica al 89 % de las mujeres que sobrevivieron, pero solo al **35 % de los hombres** que sobrevivieron, y a menos de la mitad de los sobrevivientes de **3.ª clase** (49.5 %).
- La exactitud de los hombres es alta (0.828) porque la mayoría no sobrevivió y el modelo tiende a predecir "no sobrevive" para ellos; los hombres que sí sobrevivieron son justo los casos que más le cuesta reconocer.
- El modelo reproduce el patrón del naufragio (mujeres, niños y clases altas sobrevivieron más) y por eso es menos sensible con los sobrevivientes que rompen ese patrón.
- El grupo de 60 años o más tiene solo 16 pasajeros: su exactitud de 0.938 no es confiable.

## 6. Limitaciones

- **Un solo evento histórico:** los patrones reflejan las reglas de evacuación de 1912 ("mujeres y niños primero") y la estructura social del barco; no se generalizan a otros naufragios ni a protocolos actuales.
- **Muestra pequeña:** 891 pasajeros etiquetados y 179 de prueba, de ahí el intervalo de confianza amplio y que la ventaja frente a la regla de género no sea concluyente en prueba.
- **Sobreajuste moderado:** ROC-AUC de 0.994 en entrenamiento frente a 0.887 en validación.
- **Desempeño desigual entre subgrupos:** el recall es muy bajo para los hombres (0.353) y para la 3.ª clase (0.495): el modelo detecta mal a los sobrevivientes de los grupos con menor tasa de supervivencia.
- **Datos faltantes:** cerca del 20 % de las edades se imputa, lo que puede sesgar las predicciones para pasajeros sin edad registrada.
- **Variables sensibles:** el modelo usa `Sex` y `Pclass` (y `TTitulo`, que codifica el sexo). Predice bien lo que ocurrió, pero eso no significa que esas variables sean apropiadas para tomar decisiones sobre personas hoy.
- **Asociación, no causalidad:** la importancia de una variable indica cuánto ayuda a predecir, no que cause la supervivencia.

## 7. Usos no recomendados

- Fijar primas, aprobar o negar seguros, ni ninguna decisión que afecte a personas reales.
- Priorizar a personas en evacuaciones o emergencias actuales.
- Cualquier uso fuera de un contexto académico o exploratorio.

## 8. Reproducibilidad

| Aspecto | Detalle |
|---|---|
| Python | 3.12.3 |
| scikit-learn | 1.9.1 |
| Versiones completas | Registradas en `titanic_model_v1.metadata.json` (`python_version`, `sklearn_version`, `pandas_version`) |
| Huella del modelo | Campo `model_sha256` de los metadatos |

Para reproducir el modelo desde un clon limpio:

```bash
make install
make test
make pipeline
```

> **Seguridad:** los archivos `.joblib` pueden ejecutar código al cargarse. Solo se debe cargar este modelo desde una fuente controlada y con una versión de scikit-learn compatible; `load_model` avisa si la versión cambió.
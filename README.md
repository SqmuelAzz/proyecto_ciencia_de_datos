# Análisis de supervivencia del Titanic: EDA y Machine Learning

**Autor:** Samuel Acevedo Mambuscay
**Programa:** Diplomado en Inteligencia Artificial y Ciencia de Datos
**Curso:** Herramientas y Lenguajes de IA y Ciencia de Datos

---

## 1. Descripción del proyecto

En este proyecto realizo un análisis exploratorio de datos (EDA) sobre el dataset del Titanic y, a partir de sus resultados, diseño, evalúo y guardo un modelo de machine learning que estima la probabilidad de supervivencia de un pasajero.

El caso se plantea desde la perspectiva de una empresa de seguros que estudia el naufragio del Titanic (1912) para identificar qué características de una persona se asocian con una mayor o menor probabilidad de sobrevivir, con el fin de alimentar modelos de riesgo y protocolos de evacuación.

**Preguntas que guían el trabajo:**

- **Exploratoria:** ¿qué factores tuvieron más influencia sobre la supervivencia de los pasajeros?
- **Predictiva:** dadas las características de un pasajero al embarcar, ¿con qué probabilidad habría sobrevivido y qué tan confiable es esa estimación?

El proyecto está construido para ser reproducible: todo el flujo, desde la descarga de los datos hasta el modelo guardado, se ejecuta con `make pipeline`, cada transformación tiene pruebas automáticas y no depende de rutas locales.

---

## 2. Datos

El dataset tiene 12 variables por pasajero. Las primeras 891 filas están etiquetadas (se conoce si el pasajero sobrevivió) y se usan para el análisis y el entrenamiento; el resto se usa para la prueba de humo del modelo final.

| Variable | Descripción | Tipo |
|---|---|---|
| `PassengerId` | Identificador del pasajero | entero |
| `Survived` | 0 = no sobrevivió, 1 = sobrevivió | binaria |
| `Pclass` | Clase del tiquete: 1 = alta, 2 = media, 3 = baja | ordinal |
| `Name` | Nombre completo, incluye el título (Mr, Mrs, Miss…) | texto |
| `Sex` | `male` / `female` | categórica |
| `Age` | Edad en años | decimal, con faltantes |
| `SibSp` | Número de hermanos o cónyuge a bordo | entero |
| `Parch` | Número de padres o hijos a bordo | entero |
| `Ticket` | Número de tiquete | texto |
| `Fare` | Tarifa pagada | decimal |
| `Cabin` | Cabina; la letra indica la cubierta | texto, con muchos faltantes |
| `Embarked` | Puerto de embarque: C = Cherbourg, Q = Queenstown, S = Southampton | categórica |

### Variables derivadas

| Variable | Construcción |
|---|---|
| `Person` | `child` si la edad es menor de 16 años; en otro caso, el sexo |
| `TCabin`, `HasCabin` | Letra de la cubierta (`X` si es desconocida) e indicador de cabina registrada |
| `Familia`, `Alone` | Número de familiares a bordo (`SibSp + Parch`) y si viajaba solo o acompañado |
| `TTitulo` | Título extraído del nombre, agrupado en Mr, Mrs, Miss y Otro |
| `AgeOriginal`, `AgeImputada` | Edad sin imputar e indicador de las filas imputadas |
| `NSex`, `NEmbarked`, `NCabin`, `NTitulo` | Codificación numérica de las variables categóricas |

---

## 3. Arquitectura del pipeline

Los datos se procesan en tres capas (arquitectura *medallion*) y terminan en un modelo entrenado:

```
Fuente (URL)
   │  00_ingestion.py
   ▼
BRONZE (CSV) ──01_data_engineering.py──▶ SILVER (Parquet) ──03_aggregation.py──▶ GOLD (Parquet)
                                               │
                                               │  05_ml_models.py
                                               ▼
                                   models/titanic_model_v1.joblib
                                   + titanic_model_v1.metadata.json
```

- **Bronze:** copia de los datos tal como llegan, con metadatos de ingesta (fuente, fecha y huella SHA-256) y un reporte de calidad.
- **Silver:** datos validados, tipados, limpios y con las variables derivadas. Es la entrada del análisis exploratorio y del modelo.
- **Gold:** tablas agregadas para el informe de hallazgos: tasas de supervivencia por grupo, prueba chi-cuadrado, interacción clase × sexo, análisis por puerto, tarifa y familia, y atípicos.
- **Modelo:** pipeline de scikit-learn entrenado con todos los datos etiquetados y guardado junto con sus metadatos (parámetros, variables, métricas de prueba, versiones de librerías y huella de los datos).

---

## 4. Estructura del repositorio

```
proyecto_ciencia_de_datos/
├── README.md
├── pyproject.toml              # Dependencias y configuración de herramientas
├── Makefile                    # Comandos del pipeline
├── .env.example                # Plantilla de configuración
├── docs/
│   ├── GUIA.md                 # Guía del taller
│   └── INSTALACION.md
├── flows/
│   ├── common/
│   │   ├── config.py           # Parámetros de configuración
│   │   ├── paths.py            # Rutas del proyecto
│   │   ├── logging.py          # Registro de eventos
│   │   └── hashing.py          # Huella SHA-256 de archivos
│   ├── job.py                  # Orquestador de etapas
│   ├── 00_ingestion.py         # Fuente → Bronze
│   ├── 01_data_engineering.py  # Bronze → Silver
│   ├── 03_aggregation.py       # Silver → Gold
│   └── 05_ml_models.py         # Silver → modelo entrenado y guardado
├── notebooks/
│   ├── 02_exploracion.ipynb    # Análisis exploratorio
│   └── 04_modelado.ipynb       # Formulación, comparación y evaluación de modelos
├── tests/                      # Pruebas automáticas por módulo
├── data/                       # bronze/, silver/, gold/ (generado al ejecutar)
├── models/                     # Modelo entrenado y metadatos (generado al ejecutar)
└── reports/
    ├── conclusiones_eda.md     # Hallazgos del análisis exploratorio
    ├── problema_ml.md          # Definición del problema, selección de modelos y limitaciones
    └── model_card.md           # Ficha del modelo final
```

---

## 5. Instalación

Requisitos: Python 3.11 o superior y Git. Desarrollado en WSL (Ubuntu) con VS Code.

```bash
git clone https://github.com/SqmuelAzz/proyecto_ciencia_de_datos.git
cd proyecto_ciencia_de_datos

python3 -m venv .venv
source .venv/bin/activate

make install          # equivale a: pip install -e ".[dev]"
cp .env.example .env
```

---

## 6. Ejecución

| Comando | Descripción |
|---|---|
| `make ingest` | Descarga los datos y genera la capa Bronze |
| `make silver` | Limpia y transforma los datos (capa Silver) |
| `make gold` | Genera las tablas agregadas (capa Gold) |
| `make train` | Evalúa el modelo final en el conjunto de prueba, lo entrena con todos los datos y lo guarda |
| `make pipeline` | Ejecuta todas las etapas en orden |
| `make test` | Ejecuta las pruebas automáticas |
| `make lint` | Revisa el estilo del código |

Ejecución completa desde cero:

```bash
make test
make pipeline
ls data/bronze data/silver data/gold models
```

Después se ejecutan los cuadernos con el kernel `.venv` (*Restart → Run All*):

1. `notebooks/02_exploracion.ipynb`: análisis exploratorio (Partes 1 a 9).
2. `notebooks/04_modelado.ipynb`: formulación del problema, modelado, evaluación y prueba de humo (Partes 10 a 17).

### Configuración

Los parámetros se definen en el archivo `.env`:

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `TITANIC_URL` | URL del curso | Fuente de los datos |
| `TITANIC_LOCAL_PATH` | vacío | Ruta a un CSV local, como alternativa a la URL |
| `TITANIC_N_LABELED` | `891` | Número de filas etiquetadas |
| `RANDOM_STATE` | `42` | Semilla para garantizar resultados reproducibles |
| `TEST_SIZE` | `0.2` | Proporción del conjunto de prueba |
| `CV_FOLDS` | `5` | Número de pliegues de la validación cruzada |

---

## 7. Metodología de modelado

1. **Formulación:** clasificación binaria supervisada con `Survived` como variable objetivo, usando solo información disponible al momento de embarcar. Métrica principal: **ROC-AUC**; métricas secundarias: precisión, recall, F1 y exactitud.
2. **Baselines:** un clasificador ingenuo (siempre "no sobrevive") y una regla de género ("sobrevive si es mujer"), que el modelo debe superar.
3. **Auditoría de fuga de datos:** se excluyen el objetivo y sus derivados, los identificadores y las columnas con codificación arbitraria; se usa la edad original en lugar de la imputada y se evalúan por separado las variables de cabina, por un posible artefacto del registro.
4. **Preprocesamiento en `Pipeline`:** imputación de la edad por sexo y clase aprendida solo con los datos de entrenamiento (`GroupAgeImputer`), mediana y estandarización para las numéricas, y *one-hot* para las categóricas.
5. **Comparación de modelos:** regresión logística, árbol de decisión, random forest, gradient boosting y k vecinos, frente al baseline, con validación cruzada estratificada de 5 pliegues.
6. **Evaluación en profundidad:** matriz de confusión, curvas ROC y precisión-recall, calibración, elección del umbral, errores por subgrupo y curva de aprendizaje.
7. **Ajuste de hiperparámetros:** `GridSearchCV` sobre los finalistas, con una comparación pareada para comprobar que la mejora supera la variabilidad entre pliegues.
8. **Interpretación:** coeficientes y *odds ratios*, importancia por permutación y ablación de grupos de variables.
9. **Evaluación final:** el conjunto de prueba se usa una sola vez, con un intervalo de confianza del 95 % por bootstrap.
10. **Modelo final:** entrenado con todos los datos etiquetados y guardado con sus metadatos; se verifica con una prueba de humo sobre los pasajeros sin etiqueta.

Las decisiones de cada etapa, los resultados y las limitaciones están documentados en `reports/problema_ml.md` y en la ficha del modelo, `reports/model_card.md`.

---

## 8. Avance

| Parte | Tema | Estado |
|---|---|---|
| 0 | Preparación del proyecto y capa Bronze | Completado |
| 1 | Primer contacto con los datos (Pasos 1–5) | Completado |
| 2 | Conociendo a los pasajeros (Pasos 6–10) | Completado |
| 3 | Ubicación, embarque y composición familiar (Pasos 11–15) | Completado |
| 4 | Análisis de supervivencia (Pasos 16–20) | Completado |
| 5 | Limpieza y preparación de datos (Pasos 21–26) | Completado |
| 6 | Correlaciones y estadísticas por grupos (Pasos 27–29) | Completado |
| 7 | Visualizaciones comparativas (Pasos 30–31) | Completado |
| 8 | Dataset final y verificación (Pasos 32–33) | Completado |
| 9 | Análisis crítico: interacciones, asociaciones y atípicos (Pasos 34–38) | Completado |
| 10 | Formulación del problema de ML (Pasos 39–42) | Completado |
| 11 | Preprocesamiento reproducible con `Pipeline` (Pasos 43–46) | Completado |
| 12 | Modelos candidatos y validación cruzada (Pasos 47–50) | Completado |
| 13 | Evaluación en profundidad (Pasos 51–55) | Completado |
| 14 | Ajuste de hiperparámetros (Pasos 56–58) | Completado |
| 15 | Interpretación y ablación de variables (Pasos 59–61) | Completado |
| 16 | Evaluación final sobre el conjunto de prueba (Pasos 62–63) | Completado |
| 17 | Entrenamiento final y guardado del modelo (Pasos 64–69) | Completado |

---

## 9. Decisiones de diseño

- **Separación por responsabilidades:** cada módulo cumple una sola función (ingesta, ingeniería de datos, agregación o modelado) y `job.py` solo coordina la ejecución.
- **Funciones puras:** las transformaciones reciben un `DataFrame` y devuelven uno nuevo sin modificar la entrada, lo que permite probarlas de forma aislada.
- **Pruebas automáticas:** cada función tiene pruebas que verifican su comportamiento, incluidos casos límite como pasajeros sin edad registrada o categorías no vistas en el entrenamiento.
- **Contratos de datos:** `validate_bronze` y `validate_silver` detienen el pipeline si los datos no cumplen la estructura esperada o tienen nulos en columnas clave.
- **Reproducibilidad:** semillas fijas, dependencias declaradas en `pyproject.toml` y rutas centralizadas en `paths.py`.
- **Trazabilidad:** metadatos de ingesta con huella SHA-256, registro de cada ejecución en `job.log` y metadatos del modelo con versiones, parámetros y métricas.
- **Prevención de fuga de datos:** Silver conserva la edad original y todo el preprocesamiento del modelo se ajusta solo con los datos de entrenamiento, dentro de un `Pipeline`.
- **Tratamiento explícito de faltantes:** `Embarked` se imputa con la moda, `Age` con el promedio por grupo (conservando la edad original) y `Cabin` se conserva como categoría "desconocida" (`X`).
- **Evaluación honesta:** todas las decisiones (modelo, hiperparámetros, umbral y variables) se toman con validación cruzada sobre el entrenamiento; el conjunto de prueba se usa una sola vez.

---

## 10. Tecnologías

Python · pandas · NumPy · Matplotlib · Seaborn · SciPy · scikit-learn · joblib · PyArrow · pytest · ruff · Jupyter · Git · WSL (Ubuntu) · VS Code
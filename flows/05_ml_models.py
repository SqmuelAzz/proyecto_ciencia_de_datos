"""Modelado: Silver -> modelo entrenado y guardado."""
from __future__ import annotations

import importlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from flows.common.config import settings
from flows.common.hashing import sha256_of
from flows.common.logging import get_logger
from flows.common.paths import MODELS, SILVER, SILVER_NAME

log = get_logger(__name__)

TARGET = "Survived"
NUM_FEATURES = ["Age", "SibSp", "Parch", "Fare", "Familia"]
CAT_FEATURES = ["Pclass", "Sex", "Embarked", "TTitulo", "Alone"]
CABIN_NUM = ["HasCabin"]            # features "sospechosas": se evalúan en el Paso 61
CABIN_CAT = ["TCabin"]
CANDIDATE_COLUMNS = NUM_FEATURES + CAT_FEATURES + CABIN_NUM + CABIN_CAT
PRIMARY_METRIC = "roc_auc"
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# --- Decisiones congeladas (resultado del cuaderno 04; justificadas en reports/problema_ml.md) ---
MODEL_NAME = "titanic_model_v1"
FINAL_ESTIMATOR = "hgb"                 # nombre en get_candidates()
FINAL_PARAMS = {                        # mejores hiperparámetros del Paso 57
    "model__l2_regularization": 1.0,
    "model__learning_rate": 0.1,
    "model__max_depth": 4,
    "model__max_iter": 200,
}
USE_CABIN_FEATURES = False              # decisión del Paso 61
THRESHOLD = 0.61                        # decisión del Paso 53

# ---------------------------------------------------------------------------
# Parte 10: datos y división
# ---------------------------------------------------------------------------

def load_dataset(path: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Lee Silver. Usa la edad ORIGINAL (con NaN): el modelo imputa por su cuenta."""
    df = pd.read_parquet(Path(path) if path else SILVER / SILVER_NAME)
    X = df[CANDIDATE_COLUMNS].copy()
    X["Age"] = df["AgeOriginal"]
    return X, df[TARGET].astype(int)


def split(X: pd.DataFrame, y: pd.Series, test_size: float | None = None,
          random_state: int | None = None):
    """Paso 41: devuelve X_train, X_test, y_train, y_test, estratificado por y."""
    test_size = settings.test_size if test_size is None else test_size
    random_state = settings.random_state if random_state is None else random_state
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ---------------------------------------------------------------------------
# Parte 11: preprocesamiento reproducible
# ---------------------------------------------------------------------------

class GroupAgeImputer(BaseEstimator, TransformerMixin):
    """Paso 44: rellena 'Age' con el promedio de su grupo, aprendido SOLO en fit()."""

    def __init__(self, group_cols=("Sex",), target="Age"):
        self.group_cols = group_cols
        self.target = target

    def fit(self, X, y=None):
        cols = list(self.group_cols)
        self.group_means_ = X.groupby(cols)[self.target].mean()
        self.global_mean_ = X[self.target].mean()
        return self

    def transform(self, X):
        X = X.copy()
        cols = list(self.group_cols)
        means = self.group_means_.rename("_fill").reset_index()
        fill = X[cols].merge(means, on=cols, how="left")["_fill"].to_numpy(dtype=float)
        fill = np.where(np.isnan(fill), self.global_mean_, fill)
        X[self.target] = X[self.target].fillna(pd.Series(fill, index=X.index))
        return X


def build_pipeline(model, num_features=None, cat_features=None) -> Pipeline:
    """Paso 45: GroupAgeImputer -> ColumnTransformer -> modelo, todo en un Pipeline."""
    num_features = NUM_FEATURES if num_features is None else num_features
    cat_features = CAT_FEATURES if cat_features is None else cat_features
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)),
    ])
    prep = ColumnTransformer(
        [("num", num, list(num_features)), ("cat", cat, list(cat_features))],
        sparse_threshold=0,                    # salida densa: todos los modelos la aceptan
    )
    return Pipeline([
        ("age_imputer", GroupAgeImputer(group_cols=("Sex", "Pclass"))),
        ("prep", prep),
        ("model", model),
    ])


# ---------------------------------------------------------------------------
# Parte 12: candidatos y validación cruzada
# ---------------------------------------------------------------------------

def get_candidates(random_state: int = 42) -> dict:
    """Paso 47: diccionario {nombre: estimador sin entrenar}."""
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=1000),
        "tree": DecisionTreeClassifier(max_depth=4, random_state=random_state),
        "forest": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(random_state=random_state),
        "knn": KNeighborsClassifier(),
    }


def cross_validate_candidates(X, y, num_features=None, cat_features=None,
                              random_state=None, cv_folds=None, return_folds=False):
    """Paso 48: media y desviación de cada métrica por modelo, ordenado por la métrica principal.

    Con return_folds=True devuelve también los puntajes de cada pliegue (formato largo),
    que el cuaderno usa para el boxplot del Paso 49.
    """
    random_state = settings.random_state if random_state is None else random_state
    cv_folds = settings.cv_folds if cv_folds is None else cv_folds
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    filas, pliegues = [], []
    for nombre, modelo in get_candidates(random_state).items():
        pipe = build_pipeline(modelo, num_features, cat_features)
        res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, return_train_score=True,
                             error_score="raise")
        fila = {"model": nombre}
        for m in SCORING:
            fila[f"{m}_mean"] = res[f"test_{m}"].mean()
            fila[f"{m}_std"] = res[f"test_{m}"].std()
        fila["train_roc_auc_mean"] = res["train_roc_auc"].mean()
        fila["gap_roc_auc"] = fila["train_roc_auc_mean"] - fila["roc_auc_mean"]
        filas.append(fila)
        pliegues += [{"model": nombre, "fold": i, PRIMARY_METRIC: s}
                     for i, s in enumerate(res[f"test_{PRIMARY_METRIC}"])]
        log.info("CV %-7s ROC-AUC = %.3f ± %.3f", nombre, fila["roc_auc_mean"],
                 fila["roc_auc_std"])
    tabla = (pd.DataFrame(filas).set_index("model")
             .sort_values(f"{PRIMARY_METRIC}_mean", ascending=False))
    return (tabla, pd.DataFrame(pliegues)) if return_folds else tabla


# ---------------------------------------------------------------------------
# Parte 14: hiperparámetros
# ---------------------------------------------------------------------------

def get_param_grids() -> dict[str, dict]:
    """Paso 56: rejillas por modelo, con el prefijo 'model__' que espera el Pipeline.

    En LogisticRegression no se incluye 'penalty': L2 ya es el valor por defecto y ese
    parámetro está obsoleto en scikit-learn >= 1.8.
    """
    return {
        "logreg": {"model__C": [0.01, 0.1, 1, 10]},
        "forest": {"model__n_estimators": [200, 400], "model__max_depth": [4, 6, 8, None],
                   "model__min_samples_leaf": [1, 3, 5]},
        "hgb": {"model__learning_rate": [0.03, 0.1], "model__max_depth": [2, 3, 4],
                "model__max_iter": [100, 200], "model__l2_regularization": [0.0, 1.0]},
    }


# ---------------------------------------------------------------------------
# Parte 16: evaluación final
# ---------------------------------------------------------------------------

def evaluate_on_test(pipeline, X_test, y_test, threshold: float = 0.5) -> dict:
    """Paso 62: exactitud, precisión, recall, F1, ROC-AUC y tamaño de la prueba (serializable)."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "threshold": float(threshold),
        "n_test": int(len(y_test)),
    }


def bootstrap_ci(y_true, proba, metric=roc_auc_score, n_boot: int = 1000,
                 random_state: int | None = None) -> tuple[float, float]:
    """Paso 62: intervalo de confianza al 95 % por bootstrap (percentiles 2.5 y 97.5)."""
    random_state = settings.random_state if random_state is None else random_state
    rng = np.random.default_rng(random_state)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    valores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:     # ROC-AUC necesita las dos clases
            continue
        valores.append(metric(y_true[idx], proba[idx]))
    low, high = np.percentile(valores, [2.5, 97.5])
    return float(low), float(high)


# ---------------------------------------------------------------------------
# Parte 17: modelo final, guardado y predicción
# ---------------------------------------------------------------------------

def final_features() -> tuple[list[str], list[str]]:
    """Listas de variables numéricas y categóricas según USE_CABIN_FEATURES."""
    num = NUM_FEATURES + (CABIN_NUM if USE_CABIN_FEATURES else [])
    cat = CAT_FEATURES + (CABIN_CAT if USE_CABIN_FEATURES else [])
    return num, cat


def train_final(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Paso 65: construye el pipeline con las decisiones congeladas y lo entrena con (X, y)."""
    num, cat = final_features()
    modelo = get_candidates(settings.random_state)[FINAL_ESTIMATOR]
    pipe = build_pipeline(modelo, num, cat).set_params(**FINAL_PARAMS)
    return pipe.fit(X, y)


def save_model(pipeline: Pipeline, metadata: dict, name: str = MODEL_NAME,
               models_dir: Path = MODELS) -> Path:
    """Paso 66: guarda el pipeline (.joblib) y un archivo de metadatos (.metadata.json)."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{name}.joblib"
    joblib.dump(pipeline, model_path)
    meta = {
        **metadata,
        "name": name,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "model_sha256": sha256_of(model_path),
    }
    (models_dir / f"{name}.metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    log.info("Modelo guardado en %s", model_path)
    return model_path


def load_model(path: Path) -> tuple[Pipeline, dict]:
    """Paso 66: carga el pipeline y sus metadatos; avisa si la versión de scikit-learn cambió."""
    path = Path(path)
    pipeline = joblib.load(path)
    meta = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if meta.get("sklearn_version") != sklearn.__version__:
        log.warning("Modelo entrenado con scikit-learn %s; versión actual %s",
                    meta.get("sklearn_version"), sklearn.__version__)
    return pipeline, meta


def predict(pipeline: Pipeline, X: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Paso 67: DataFrame con 'proba_survive' y 'pred' (0/1) para cada fila de X."""
    proba = pipeline.predict_proba(X)[:, 1]
    return pd.DataFrame({"proba_survive": proba, "pred": (proba >= threshold).astype(int)},
                        index=X.index)


def features_from_bronze(bronze: pd.DataFrame) -> pd.DataFrame:
    """Paso 69: aplica a pasajeros nuevos SOLO las transformaciones deterministas de Silver.

    No se imputa nada con estadísticas de los datos nuevos: el pipeline guardado imputa con
    lo que aprendió en el entrenamiento. 'Age' queda sin imputar, igual que en load_dataset().
    """
    de = importlib.import_module("flows.01_data_engineering")
    df = (bronze
          .pipe(de.add_person)
          .pipe(de.add_cabin_features)
          .pipe(de.add_family_features)
          .pipe(de.add_title)
          .pipe(de.consolidate_titles))
    return df[CANDIDATE_COLUMNS].copy()


def run(in_dir: Path | None = None, models_dir: Path = MODELS) -> Path:
    """Paso 68: Silver -> evaluación honesta -> modelo final entrenado y guardado."""
    silver_path = Path(in_dir) / SILVER_NAME if in_dir else SILVER / SILVER_NAME
    X, y = load_dataset(silver_path)
    X_tr, X_te, y_tr, y_te = split(X, y)

    # Evaluación: se entrena con el entrenamiento y se evalúa UNA vez en la prueba
    pipe_eval = train_final(X_tr, y_tr)
    metrics = evaluate_on_test(pipe_eval, X_te, y_te, THRESHOLD)
    low, high = bootstrap_ci(y_te, pipe_eval.predict_proba(X_te)[:, 1])
    metrics["roc_auc_ci95"] = [low, high]
    log.info("Prueba: ROC-AUC = %.3f (IC95 %.3f-%.3f), exactitud = %.3f, F1 = %.3f",
             metrics["roc_auc"], low, high, metrics["accuracy"], metrics["f1"])

    # Modelo definitivo: con TODOS los datos etiquetados
    pipeline = train_final(X, y)
    num, cat = final_features()
    metadata = {
        "estimator": FINAL_ESTIMATOR,
        "params": FINAL_PARAMS,
        "features": {"numeric": num, "categorical": cat},
        "target": TARGET,
        "threshold": THRESHOLD,
        "random_state": settings.random_state,
        "n_train": int(len(X)),
        "data_sha256": sha256_of(silver_path),
        "metrics": metrics,
    }
    return save_model(pipeline, metadata, models_dir=models_dir)
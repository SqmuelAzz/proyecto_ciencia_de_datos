"""Ingeniería de datos: Bronze -> Silver.

Cada transformación es una función pura: recibe un DataFrame y devuelve uno
nuevo. El orden importa y está fijado en build_silver().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flows.common.logging import get_logger
from flows.common.paths import (
    BRONZE, BRONZE_LABELED_NAME, SILVER, SILVER_FINAL_NAME, SILVER_NAME,
)

log = get_logger(__name__)

EXPECTED_COLUMNS = [
    "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
]
DECKS = list("ABCDEFGTX")            # orden FIJO: X = desconocida (reproducible)
TITLE_MAP = {                        # consolidación de títulos (Paso 26)
    "Master": "Mr", "the Countess": "Mrs", "Dona": "Mrs", "Mme": "Mrs",
    "Lady": "Miss", "Mlle": "Miss", "Ms": "Miss",
}
MAIN_TITLES = ["Mr", "Mrs", "Miss"]


# ---------- Implementadas (Parte 2) ----------

def validate_bronze(df: pd.DataFrame) -> pd.DataFrame:
    """Contrato de entrada: columnas esperadas y PassengerId sin duplicados."""
    faltantes = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en Bronze: {faltantes}")
    if df["PassengerId"].duplicated().any():
        raise ValueError("Hay PassengerId duplicados en Bronze")
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Survived, Pclass, SibSp, Parch -> int; Age y Fare -> float."""
    df = df.copy()
    for col in ["Survived", "Pclass", "SibSp", "Parch"]:
        df[col] = df[col].astype(int)
    for col in ["Age", "Fare"]:
        df[col] = df[col].astype(float)
    return df


def add_person(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Person': 'child' si Age < 16; si no, el valor de Sex.

    Regla para edad desconocida (NaN): se usa Sex. Esto subestima a los niños
    sin edad registrada; el sesgo se cuantifica en el Reto 8.1.
    """
    df = df.copy()
    df["Person"] = np.where(df["Age"] < 16, "child", df["Sex"])
    return df


# ---------- Pendientes (Partes 3 a 8): por ahora devuelven df sin cambios ----------

def add_cabin_features(df: pd.DataFrame) -> pd.DataFrame:    # Pasos 11 y 24
    """Agrega 'TCabin' (primera letra de Cabin, 'X' si es nula) y 'HasCabin' (0/1)."""
    df = df.copy()
    df["TCabin"] = df["Cabin"].str[0].fillna("X")
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    return df


def add_family_features(df: pd.DataFrame) -> pd.DataFrame:   # Paso 14
    """Agrega 'Familia' = SibSp + Parch y 'Alone' ('with Family' si Familia > 0, 'Alone' si no)."""
    df = df.copy()
    df["Familia"] = df["SibSp"] + df["Parch"]
    df["Alone"] = np.where(df["Familia"] > 0, "with Family", "Alone")
    return df


def add_survivor_label(df: pd.DataFrame) -> pd.DataFrame:    # Paso 16
    """Agrega 'Survivor' mapeando 0 -> 'No' y 1 -> 'Yes' (solo para etiquetar gráficos)."""
    df = df.copy()
    df["Survivor"] = df["Survived"].map({0: "No", 1: "Yes"})
    return df


def add_title(df: pd.DataFrame) -> pd.DataFrame:             # Paso 25
    return df


def consolidate_titles(df: pd.DataFrame) -> pd.DataFrame:    # Paso 26
    return df


def impute_embarked(df: pd.DataFrame) -> pd.DataFrame:       # Paso 21
    return df


def impute_age(df: pd.DataFrame) -> pd.DataFrame:            # Paso 22
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:   # Pasos 21, 23, 24, 26
    return df


def drop_original_columns(df: pd.DataFrame) -> pd.DataFrame:  # Paso 32
    return df


def validate_silver(df: pd.DataFrame) -> pd.DataFrame:       # Paso 33
    return df


# ---------- Orquestación ----------

def build_silver(bronze: pd.DataFrame) -> pd.DataFrame:
    """Orden deliberado: Person usa la edad ORIGINAL, así que va antes de imputar."""
    return (
        bronze
        .pipe(validate_bronze)
        .pipe(cast_types)
        .pipe(add_person)
        .pipe(add_cabin_features)
        .pipe(add_family_features)
        .pipe(add_survivor_label)
        .pipe(add_title)
        .pipe(consolidate_titles)
        .pipe(impute_embarked)
        .pipe(impute_age)
        .pipe(encode_categoricals)
    )


def run(in_dir: Path = BRONZE, out_dir: Path = SILVER) -> Path:
    bronze = pd.read_csv(Path(in_dir) / BRONZE_LABELED_NAME)
    silver = validate_silver(build_silver(bronze))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(out_dir / SILVER_NAME, index=False)
    drop_original_columns(silver).to_parquet(out_dir / SILVER_FINAL_NAME, index=False)
    log.info("Silver listo: %d filas, %d columnas", *silver.shape)
    return out_dir / SILVER_NAME
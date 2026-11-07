"""Agregaciones: Silver -> Gold. Cada función devuelve una tabla lista para un informe."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

from flows.common.logging import get_logger
from flows.common.paths import GOLD, SILVER, SILVER_NAME

log = get_logger(__name__)


def _clean(table: pd.DataFrame) -> pd.DataFrame:
    """Índice como columna normal y nombres de columna en texto (requisito de Parquet)."""
    table = table.reset_index().rename(columns=str)
    table.columns.name = None
    return table


# ---------------------------------------------------------------------------
# Parte 6
# ---------------------------------------------------------------------------

def group_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Paso 28: estadísticas descriptivas por grupos."""
    return {
        "age_by_survival": _clean(df.groupby("Survived")["Age"].describe()),
        "survival_by_sex": _clean(df.groupby("Sex")["Survived"].agg(["mean", "sum", "count"])),
        "survival_by_pclass": _clean(
            df.groupby("Pclass")["Survived"].agg(["mean", "sum", "count"])),
        "survival_by_alone": _clean(df.groupby("Alone")["Survived"].agg(["mean", "sum", "count"])),
    }


def chi2_sex_survival(df: pd.DataFrame) -> dict:
    """Paso 29: prueba de independencia chi-cuadrado entre Sex y Survived."""
    tabla = pd.crosstab(df["Sex"], df["Survived"])
    chi2, p, gl, _esperados = stats.chi2_contingency(tabla)
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(gl)}

# ---------------------------------------------------------------------------
# Parte 9
# ---------------------------------------------------------------------------

def survival_by_class_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 34: tasa de supervivencia, Pclass en filas y Sex en columnas."""
    return _clean(df.pivot_table(values="Survived", index="Pclass", columns="Sex",
                                 aggfunc="mean"))


def survival_by_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 35 (1): tasa y número de pasajeros por puerto."""
    return _clean(df.groupby("Embarked")["Survived"].agg(["mean", "count"]))


def survival_by_embarked_class(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 35 (2): puerto estratificado por clase (para detectar confusión)."""
    return _clean(df.pivot_table(values="Survived", index="Embarked", columns="Pclass",
                                 aggfunc="mean"))


def add_fare_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 36 (1): 'FareQ' = cuartil de la tarifa calculado DENTRO de cada clase."""
    df = df.copy()
    df["FareQ"] = df.groupby("Pclass")["Fare"].transform(
        lambda s: pd.qcut(s, 4, labels=False, duplicates="drop"))
    return df


def survival_by_fare_quartile_class(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 36 (2): tasa de supervivencia por cuartil de tarifa y clase."""
    return _clean(add_fare_quartile(df).pivot_table(
        values="Survived", index="FareQ", columns="Pclass", aggfunc="mean"))


def survival_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 37: tasa y número de pasajeros según el número de familiares a bordo."""
    return _clean(df.groupby("Familia")["Survived"].agg(["mean", "count"]))


def outlier_mask_iqr(s: pd.Series) -> pd.Series:
    """Paso 38 (1): True si el valor queda fuera de [Q1 - 1.5·IQR, Q3 + 1.5·IQR]."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 38 (2): atípicos de Fare, Age (original) y Familia, con la tasa de supervivencia
    de los atípicos frente al resto."""
    columnas = {
        "Fare": df["Fare"],
        "Age": df["AgeOriginal"] if "AgeOriginal" in df.columns else df["Age"],
        "Familia": df["Familia"],
    }
    filas = []
    for nombre, serie in columnas.items():
        mask = outlier_mask_iqr(serie.dropna()).reindex(df.index, fill_value=False)
        filas.append({
            "variable": nombre,
            "n_atipicos": int(mask.sum()),
            "tasa_superv_atipicos": float(df.loc[mask, "Survived"].mean()) if mask.any() else None,
            "tasa_superv_resto": float(df.loc[~mask, "Survived"].mean()),
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Orquestación (Parte 6: solo group_stats y chi-cuadrado; la Parte 9 añade el resto)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def run(in_dir: Path | None = None, out_dir: Path = GOLD) -> Path:
    df = pd.read_parquet(Path(in_dir or SILVER) / SILVER_NAME)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        **group_stats(df),
        "survival_by_class_sex": survival_by_class_sex(df),
        "survival_by_embarked": survival_by_embarked(df),
        "survival_by_embarked_class": survival_by_embarked_class(df),
        "survival_by_fare_quartile_class": survival_by_fare_quartile_class(df),
        "survival_by_family": survival_by_family(df),
        "outlier_summary": outlier_summary(df),
    }
    for name, table in tables.items():
        table.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "chi2_sex_survival.json").write_text(json.dumps(chi2_sex_survival(df), indent=2))
    log.info("Gold listo: %d tablas en %s", len(tables) + 1, out_dir)
    return out_dir
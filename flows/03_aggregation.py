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
# Orquestación (Parte 6: solo group_stats y chi-cuadrado; la Parte 9 añade el resto)
# ---------------------------------------------------------------------------

def run(in_dir: Path | None = None, out_dir: Path = GOLD) -> Path:
    df = pd.read_parquet(Path(in_dir or SILVER) / SILVER_NAME)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = group_stats(df)
    for name, table in tables.items():
        table.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "chi2_sex_survival.json").write_text(json.dumps(chi2_sex_survival(df), indent=2))
    log.info("Gold listo: %d tablas en %s", len(tables) + 1, out_dir)
    return out_dir
import importlib

import numpy as np
import pandas as pd
import pytest

ml = importlib.import_module("flows.05_ml_models")


@pytest.fixture
def toy():
    """Dataset sintético con las columnas candidatas, para probar sin depender de Silver."""
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame({
        "Age": np.where(rng.random(n) < 0.2, np.nan, rng.uniform(1, 70, n)),
        "SibSp": rng.integers(0, 4, n),
        "Parch": rng.integers(0, 3, n),
        "Fare": rng.uniform(5, 100, n),
        "Familia": rng.integers(0, 6, n),
        "Pclass": rng.choice([1, 2, 3], n),
        "Sex": rng.choice(["male", "female"], n),
        "Embarked": rng.choice(["S", "C", "Q"], n),
        "TTitulo": rng.choice(["Mr", "Mrs", "Miss", "Otro"], n),
        "Alone": rng.choice(["Alone", "with Family"], n),
        "HasCabin": rng.integers(0, 2, n),
        "TCabin": rng.choice(list("XXXXCB"), n),
    })
    y = pd.Series(((X["Sex"] == "female") & (rng.random(n) < 0.9)).astype(int))
    return X, y


# Parte 10 (Paso 41). Las pruebas de las Partes 11 en adelante se añaden después.
def test_split_is_stratified_and_reproducible(toy):
    X, y = toy
    a = ml.split(X, y, test_size=0.25, random_state=1)
    b = ml.split(X, y, test_size=0.25, random_state=1)
    pd.testing.assert_index_equal(a[0].index, b[0].index)
    assert abs(a[2].mean() - a[3].mean()) < 0.03
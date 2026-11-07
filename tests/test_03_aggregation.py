import importlib

import pandas as pd
import pytest

agg = importlib.import_module("flows.03_aggregation")


@pytest.fixture
def silver_like():
    return pd.DataFrame({
        "Survived": [1, 1, 0, 0, 1, 0, 0, 1],
        "Pclass": [1, 1, 1, 3, 3, 3, 3, 3],
        "Sex": ["female", "male", "male", "female", "female", "male", "male", "male"],
        "Embarked": ["S", "C", "S", "Q", "S", "S", "C", "Q"],
        "Fare": [80.0, 90.0, 70.0, 7.0, 8.0, 7.5, 9.0, 300.0],
        "Age": [30.0, 40.0, 50.0, 20.0, 25.0, 30.0, 35.0, 28.0],
        "Familia": [0, 1, 0, 2, 0, 0, 1, 0],
        "Alone": ["Alone", "with Family", "Alone", "with Family", "Alone", "Alone",
                  "with Family", "Alone"],
    })


def test_chi2_returns_valid_p_value(silver_like):
    res = agg.chi2_sex_survival(silver_like)
    assert set(res) == {"chi2", "p_value", "dof"}
    assert 0 <= res["p_value"] <= 1
import importlib

import numpy as np
import pandas as pd
import pytest

de = importlib.import_module("flows.01_data_engineering")


@pytest.fixture
def sample():
    return pd.DataFrame({
        "PassengerId": [1, 2, 3, 4, 5, 6],
        "Survived": [0, 1, 1, 0, 1, 0],
        "Pclass": [3, 1, 3, 1, 2, 3],
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
            "Heikkinen, Miss. Laina",
            "Palsson, Master. Gosta Leonard",
            "Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)",
            "Moran, Dr. James",
        ],
        "Sex": ["male", "female", "female", "male", "female", "male"],
        "Age": [22.0, 38.0, np.nan, 2.0, 33.0, np.nan],
        "SibSp": [1, 1, 0, 3, 0, 0],
        "Parch": [0, 0, 0, 1, 0, 0],
        "Ticket": list("ABCDEF"),
        "Fare": [7.25, 71.28, 7.92, 21.07, 86.5, 8.46],
        "Cabin": [None, "C85", None, None, "B77", None],
        "Embarked": ["S", "C", "S", None, "S", "Q"],
    })


def test_validate_bronze_rejects_missing_columns(sample):
    with pytest.raises(ValueError):
        de.validate_bronze(sample.drop(columns=["Fare"]))


def test_add_person_is_explicit_about_unknown_age(sample):
    out = de.add_person(sample)
    assert out["Person"].tolist() == ["male", "female", "female", "child", "female", "male"]

def test_family_features(sample):
    out = de.add_family_features(sample)
    assert out["Familia"].tolist() == [1, 1, 0, 4, 0, 0]
    assert out["Alone"].tolist() == ["with Family", "with Family", "Alone",
                                     "with Family", "Alone", "Alone"]


def test_cabin_features(sample):
    out = de.add_cabin_features(sample)
    assert out["TCabin"].tolist() == ["X", "C", "X", "X", "B", "X"]
    assert out["HasCabin"].tolist() == [0, 1, 0, 0, 1, 0]
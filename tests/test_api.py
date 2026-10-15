"""Tests for the FastAPI app in main.py."""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    # Using the context manager runs the lifespan, which loads the model.
    with TestClient(app) as c:
        yield c


def test_get_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {
        "message": "Welcome to the Census Income Prediction API!"
    }


def test_post_predict_below_50k(client):
    sample = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}


def test_post_predict_above_50k(client):
    sample = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 289436,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States",
    }
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}


def test_post_predict_missing_field(client):
    r = client.post("/predict", json={"age": 30})
    assert r.status_code == 422

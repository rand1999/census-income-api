"""Unit tests for the ML functions in starter/ml."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    inference,
    performance_on_slices,
    train_model,
)
from starter.train_model import CAT_FEATURES, LABEL, load_data


@pytest.fixture(scope="module")
def sample_data():
    return load_data().sample(n=1000, random_state=0)


@pytest.fixture(scope="module")
def processed(sample_data):
    X, y, encoder, lb = process_data(
        sample_data, categorical_features=CAT_FEATURES, label=LABEL,
        training=True,
    )
    return X, y, encoder, lb


def test_process_data_shapes(sample_data, processed):
    X, y, encoder, lb = processed
    assert X.shape[0] == len(sample_data) == y.shape[0]
    assert set(np.unique(y)) <= {0, 1}
    # Inference mode reuses the fitted encoder and gives the same width.
    X_inf, _, _, _ = process_data(
        sample_data.drop(columns=[LABEL]), categorical_features=CAT_FEATURES,
        training=False, encoder=encoder, lb=lb,
    )
    assert X_inf.shape == X.shape


def test_train_model_returns_fitted_rf(processed):
    X, y, _, _ = processed
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "classes_")


def test_inference_output(processed):
    X, y, _, _ = processed
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)) <= {0, 1}


def test_compute_model_metrics():
    y = np.array([1, 0, 1, 1, 0])
    preds = np.array([1, 0, 0, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(2 / 3)


def test_performance_on_slices(sample_data, processed):
    X, y, encoder, lb = processed
    model = train_model(X, y)
    result = performance_on_slices(
        model, sample_data, CAT_FEATURES, LABEL, encoder, lb,
        features=["sex"],
    )
    assert isinstance(result, pd.DataFrame)
    assert set(result["value"]) == set(sample_data["sex"].unique())
    assert result["n_samples"].sum() == len(sample_data)

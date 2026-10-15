import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from starter.ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and
    F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)


def save_artifact(obj, path):
    """Pickle `obj` to `path`, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_artifact(path):
    """Load a pickled object from `path`."""
    with open(path, "rb") as f:
        return pickle.load(f)


def performance_on_slices(
    model, data, categorical_features, label, encoder, lb, features=None
):
    """Compute precision, recall and F1 for every value of each
    categorical feature (data slice).

    Inputs
    ------
    model : trained model
    data : pd.DataFrame
        Raw (unprocessed) data including the label column.
    categorical_features : list[str]
        Categorical feature names used by `process_data`.
    label : str
        Name of the label column.
    encoder, lb : fitted OneHotEncoder and LabelBinarizer.
    features : list[str] | None
        Features to slice on. Defaults to all categorical features.

    Returns
    -------
    pd.DataFrame with columns
        feature, value, n_samples, precision, recall, fbeta
    """
    features = features or categorical_features
    rows = []
    for feature in features:
        for value in sorted(data[feature].unique()):
            slice_df = data[data[feature] == value]
            X_slice, y_slice, _, _ = process_data(
                slice_df,
                categorical_features=categorical_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb,
            )
            preds = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)
            rows.append(
                {
                    "feature": feature,
                    "value": value,
                    "n_samples": len(slice_df),
                    "precision": precision,
                    "recall": recall,
                    "fbeta": fbeta,
                }
            )
    return pd.DataFrame(rows)

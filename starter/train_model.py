# Script to train machine learning model.
# Run from the repository root with:  python -m starter.train_model

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    inference,
    performance_on_slices,
    save_artifact,
    train_model,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "census.csv"
MODEL_DIR = ROOT / "model"

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL = "salary"


def load_data(path=DATA_PATH):
    """Load census data and strip the stray whitespace in the raw CSV."""
    data = pd.read_csv(path, skipinitialspace=True)
    data.columns = data.columns.str.strip()
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].str.strip()
    return data


def main():
    data = load_data()

    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data[LABEL]
    )

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label=LABEL, training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(
        f"Test set -> precision: {precision:.4f}  "
        f"recall: {recall:.4f}  F1: {fbeta:.4f}"
    )

    save_artifact(model, MODEL_DIR / "model.pkl")
    save_artifact(encoder, MODEL_DIR / "encoder.pkl")
    save_artifact(lb, MODEL_DIR / "lb.pkl")

    slices = performance_on_slices(
        model, test, CAT_FEATURES, LABEL, encoder, lb
    )
    with open(ROOT / "slice_output.txt", "w") as f:
        f.write(
            f"Overall test set: precision={precision:.4f} "
            f"recall={recall:.4f} F1={fbeta:.4f}\n\n"
        )
        for _, row in slices.iterrows():
            f.write(
                f"{row['feature']} = {row['value']} "
                f"(n={row['n_samples']}): "
                f"precision={row['precision']:.4f} "
                f"recall={row['recall']:.4f} "
                f"F1={row['fbeta']:.4f}\n"
            )
    print("Saved model artifacts to", MODEL_DIR)
    print("Saved slice performance to", ROOT / "slice_output.txt")


if __name__ == "__main__":
    main()

"""FastAPI app serving the census income classifier."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from starter.ml.data import process_data
from starter.ml.model import inference, load_artifact
from starter.train_model import CAT_FEATURES

# Resolve paths relative to this file so it works locally and when hosted.
MODEL_DIR = Path(__file__).resolve().parent / "model"

artifacts: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifacts["model"] = load_artifact(MODEL_DIR / "model.pkl")
    artifacts["encoder"] = load_artifact(MODEL_DIR / "encoder.pkl")
    artifacts["lb"] = load_artifact(MODEL_DIR / "lb.pkl")
    yield
    artifacts.clear()


app = FastAPI(
    title="Census Income Prediction API",
    description="Predicts whether income exceeds $50K/yr from census data.",
    version="1.0.0",
    lifespan=lifespan,
)


class CensusData(BaseModel):
    """One person's census record. Hyphenated column names are handled
    with field aliases, so the request body uses the original CSV names."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
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
            ]
        },
    )

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


class Prediction(BaseModel):
    prediction: str


@app.get("/")
async def welcome() -> Dict[str, str]:
    return {"message": "Welcome to the Census Income Prediction API!"}


@app.post("/predict")
async def predict(data: CensusData) -> Prediction:
    df = pd.DataFrame([data.model_dump(by_alias=True)])
    X, _, _, _ = process_data(
        df,
        categorical_features=CAT_FEATURES,
        training=False,
        encoder=artifacts["encoder"],
        lb=artifacts["lb"],
    )
    pred = inference(artifacts["model"], X)
    label = artifacts["lb"].inverse_transform(pred)[0]
    return Prediction(prediction=str(label))

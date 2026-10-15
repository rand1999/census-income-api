"""POST one sample to the deployed API and print the result.

Usage:
    python live_post.py https://<your-app>.onrender.com
"""

import sys

import requests

DEFAULT_URL = "https://census-income-api.onrender.com"

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

if __name__ == "__main__":
    base_url = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL).rstrip("/")
    response = requests.post(f"{base_url}/predict", json=sample, timeout=60)
    print(f"Status code: {response.status_code}")
    print(f"Result: {response.json()}")

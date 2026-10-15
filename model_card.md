# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- **Type:** Random Forest classifier (scikit-learn 1.7.2), 100 trees, `max_depth=15`, `min_samples_split=4`, `random_state=42`.
- **Task:** Binary classification — predicts whether a person's annual income is `>50K` or `<=50K`.
- **Preprocessing:** Categorical features one-hot encoded (`OneHotEncoder(handle_unknown="ignore")`); the label is binarized with `LabelBinarizer`. Continuous features are passed through unscaled (tree models don't need scaling).
- **Artifacts:** `model/model.pkl`, `model/encoder.pkl`, `model/lb.pkl`.
- **Trained with:** `python -m starter.train_model`.

## Intended Use
An educational project demonstrating an ML pipeline deployed behind a FastAPI REST API with CI/CD. It is **not** intended for real decisions about individuals such as hiring, lending, insurance, or housing.

## Training Data
The UCI Census Income ("Adult") dataset, extracted from the 1994 U.S. Census database — 32,561 rows, 14 features plus the `salary` label (about 24% `>50K`). Whitespace in the raw CSV is stripped when loading. 80% of the data (stratified on the label, `random_state=42`) was used for training.

Categorical features: workclass, education, marital-status, occupation, relationship, race, sex, native-country.
Continuous features: age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week.

## Evaluation Data
The remaining 20% of the dataset (6,513 rows), held out with the same stratified split and processed with the encoder and label binarizer fitted on the training data.

## Metrics
Precision, recall and F1 on the positive class (`>50K`), measured on the held-out test set:

| Metric    | Value  |
|-----------|--------|
| Precision | 0.7950 |
| Recall    | 0.5861 |
| F1        | 0.6747 |

Performance on every value of every categorical feature is written to `slice_output.txt`. Examples:

| Slice                      | n     | Precision | Recall | F1     |
|----------------------------|-------|-----------|--------|--------|
| sex = Female               | 2158  | 0.8733    | 0.5347 | 0.6633 |
| sex = Male                 | 4355  | 0.7833    | 0.5956 | 0.6767 |
| race = White               | 5533  | 0.7996    | 0.5887 | 0.6781 |
| race = Black               | 662   | 0.8136    | 0.5161 | 0.6316 |
| race = Amer-Indian-Eskimo  | 73    | 0.7500    | 0.3333 | 0.4615 |

## Ethical Considerations
The data contains sensitive attributes (race, sex, native country, marital status) and reflects 1994 social and economic patterns, including historical bias. Recall differs noticeably across groups — e.g. lower for women and for smaller racial groups — so the model misses high earners in those groups more often. Several slices have very few samples, so their metrics are unreliable. Predictions should not be used to make or justify decisions about real people.

## Caveats and Recommendations
- The data is about 30 years old; incomes and the `$50K` threshold no longer mean the same thing, so the model will not transfer to today's population.
- The classes are imbalanced and recall (0.59) is noticeably lower than precision; consider class weighting, threshold tuning, or gradient-boosted models to improve it.
- Hyperparameters were not tuned with cross-validation; k-fold CV and a fairness audit (e.g. with Aequitas) are recommended before any wider use.
- Unseen category values at inference time are ignored by the encoder rather than rejected.

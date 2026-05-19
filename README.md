# Titanic Survival Prediction — ML Workshop

A beginner-friendly machine learning workshop built around the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic).

## The competition

The Titanic dataset is the classic entry point for learning supervised machine learning. The task is simple: given passenger information (age, sex, ticket class, cabin, family size, etc.), predict whether a passenger survived the disaster.

The training set contains 891 passengers with known outcomes. The test set contains 418 passengers — your model's predictions on these are what you submit to Kaggle for scoring.

## What this workshop covers

Working through the notebook, you will:

- **Explore** the dataset — spot missing values, understand distributions, and find which features correlate with survival
- **Impute** missing data using domain knowledge rather than a naive global fill (median age per class, port from fare proximity, deck from cabin letter)
- **Engineer features** that compress signal the model can't easily find on its own — passenger titles, family size, fare per person, deck
- **Build a sklearn Pipeline** — understand why wiring transformers into a Pipeline is essential to prevent data leakage during cross-validation
- **Evaluate** the model with 5-fold cross-validation before touching the test set
- **Submit** — generate `submission.csv` in Kaggle's required format

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/data-exploration.ipynb
```

Run the cells top to bottom. Each section has a markdown explanation before the code.

## Data

Download `train.csv` and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/titanic/data) and place them in the `data/` folder.

## Stack

Python · pandas · scikit-learn · seaborn · matplotlib

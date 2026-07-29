# PUBG Machine Learning App

This Streamlit app trains and evaluates machine learning models using PUBG match data from `PUBG.csv`.

## Features

- Train a Linear Regression model to predict the `Win` probability.
- Train classification models to predict a binary win result using:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
- Display performance metrics for each model.

## Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Run the app

From the project folder, run:

```bash
streamlit run app.py
```

## Files

- `app.py` - Streamlit application
- `PUBG.csv` - dataset used for training and evaluation
- `requirements.txt` - Python package dependencies
- `README.md` - project overview and usage instructions

## Notes

- The app encodes text columns using `LabelEncoder`.
- It drops rows with missing values before training.
- For classification, it converts `Win` into a binary result (`Result`).

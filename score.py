"""
score.py

This script loads a trained model from a pickle file, preprocesses the data, and classifies based on probabilities and a threshold that should be tweaked as needed.
The 'preprocessing' function adds a 'missing' column indicating the proportion of missing values across selected features.

Features used for calculating missing data:
- Age
- Shape
- Margin
- Density

Usage:
python score.py <path_to_pickle_model> <path_to_dataset_csv>
"""

import pickle
import pandas as pd
import numpy as np


def preprocessing(dataset):
    dataset = dataset.apply(lambda x: pd.to_numeric(x,errors='coerce'))
    dataset["Margin"] = dataset["Margin"].fillna("Missing").apply(lambda x: np.where((x == 2.0) | (x == 3.0),"2_3",x)).str.replace(".0","").astype("category")
    dataset["Density"] = dataset["Density"].fillna("Missing").apply(lambda x: np.where((x == 1.0) | (x == 4.0),"1_4",x)).str.replace(".0","").astype("category")
    dataset["Density"].value_counts(normalize=True)

    missing_data_rows = dataset.isna().sum(axis=1) / dataset.drop("Severity", axis=1).shape[1]

    dataset_processed = dataset.copy().assign(missing=missing_data_rows.astype("category"))

    return dataset_processed

def load_data(dataset_path):
    return pd.read_csv(dataset_path)

def main(model_path, dataset_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    dataset = load_data(dataset_path)

    features = ["Age", "Shape", "Margin", "Density"]

    dataset_processed = preprocessing(dataset[features])

    probabilities = model.predict_proba(dataset_processed)[:, 1]

    predictions = (probabilities > 0.2).astype(int)

    print("Example predictions:")
    print(predictions)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python score.py <path_to_pickle_model> <path_to_dataset_csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

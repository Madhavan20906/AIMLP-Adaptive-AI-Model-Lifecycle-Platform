"""
sample_data/make_samples.py
Creates a few varied test datasets so Mode 1 can be verified across
different problem types, not just the fraud dataset used for Mode 2.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification, make_regression

OUT = Path(__file__).parent


def make_churn_sample(n=3000, seed=1):
    """Classification with mixed categorical + numerical features, moderate imbalance."""
    X, y = make_classification(
        n_samples=n, n_features=10, n_informative=6, weights=[0.75, 0.25],
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"num_feat_{i}" for i in range(10)])
    rng = np.random.RandomState(seed)
    df["contract_type"] = rng.choice(["monthly", "annual", "two_year"], size=n)
    df["region"] = rng.choice(["north", "south", "east", "west"], size=n)
    df["monthly_charges"] = np.round(rng.uniform(20, 150, size=n), 2)
    # inject some missing values and duplicates, since real CSVs have them
    mask = rng.rand(n) < 0.03
    df.loc[mask, "monthly_charges"] = np.nan
    df["churn"] = y
    df = pd.concat([df, df.sample(20, random_state=seed)], ignore_index=True)  # duplicates
    return df


def make_house_price_sample(n=2000, seed=2):
    """Regression, purely numeric."""
    X, y = make_regression(n_samples=n, n_features=8, noise=15, random_state=seed)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])
    df["price"] = y
    return df


if __name__ == "__main__":
    make_churn_sample().to_csv(OUT / "churn_sample.csv", index=False)
    make_house_price_sample().to_csv(OUT / "house_price_sample.csv", index=False)
    print("Wrote churn_sample.csv (classification) and house_price_sample.csv (regression)")

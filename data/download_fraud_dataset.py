"""
data/download_fraud_dataset.py
Fetches the real ULB Credit Card Fraud dataset used to test Mode 2 in this
project (same dataset as the standalone drift_pipeline project). Not
required for Mode 1, which works on any CSV you provide.
"""
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
OUT = Path(__file__).parent / "creditcard.csv"

if __name__ == "__main__":
    if OUT.exists():
        print(f"Already present at {OUT}")
    else:
        print(f"Downloading (~100MB) to {OUT} ...")
        urllib.request.urlretrieve(URL, OUT)
        print("Done.")

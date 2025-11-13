# aml_pipeline.py
# Run as: python notebooks/aml_pipeline.py
from src.data import generate_synthetic_transactions
from src.features import featurize
from src.models import train
import pandas as pd
import os

def main():
    df = generate_synthetic_transactions(n=5000)
    df = featurize(df)
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet("data/processed/transactions_featurized.parquet", index=False)
    clf, iso, scaler = train(df)
    print("Pipeline finished.")

if __name__ == '__main__':
    main()
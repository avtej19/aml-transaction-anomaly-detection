# AML Transaction Anomaly Detection

**Goal:** End-to-end pipeline that detects suspicious transactions using unsupervised and supervised methods. Demonstrates data pipeline, feature engineering, model training, evaluation, explainability, and a simple demo API.

## Contents
- `src/` — modular Python scripts (data loading, features, modeling)
- `notebooks/aml_pipeline.py` — runnable script to generate synthetic data and train models
- `data/sample/transactions_sample.csv` — small synthetic sample for quick run
- `models/` — saved model artifacts after training
- `deployment/app.py` — simple Flask demo API
- `requirements.txt` — Python dependencies
- `model_card.md` — short description of model use, limitations, and ethics

## Quickstart (local)
1. Create a virtual environment and install deps:
   ```
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. Prepare data & train models:
   ```
   python src/data.py --generate 2000
   python src/models.py
   ```
3. Run demo API:
   ```
   python deployment/app.py
   ```
4. Open `reports/figures` for evaluation plots.

## Notes for reviewers
- Focus on Precision@k and operational metrics.
- See `model_card.md` for deployment considerations and limitations.
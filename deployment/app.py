from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
clf = None
scaler = None
iso = None

def load_models():
    global clf, scaler, iso
    clf = joblib.load("models/rf_fraud.pkl")
    scaler = joblib.load("models/scaler.pkl")
    iso = joblib.load("models/iso.pkl")

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.json.get("data")
    df = pd.DataFrame(data)
    # basic featurize (must match src/features)
    df['log_amount'] = np.log1p(df['amount'])
    df['night_txn'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
    df['high_risk_flag'] = ((df['country_risk']==2) & (df['amount']>2000)).astype(int)
    df['recent_account'] = (df['account_age_days'] < 30).astype(int)
    features = ["log_amount","hour","country_risk","txn_type","account_age_days","num_txn_24h","night_txn","high_risk_flag","recent_account"]
    X = df[features].fillna(0)
    Xs = scaler.transform(X)
    scores = clf.predict_proba(X)[:,1]
    result = []
    for i, s in enumerate(scores):
        result.append({"index": i, "score": float(s), "is_flagged": bool(s>0.5)})
    return jsonify(result)

if __name__ == '__main__':
    load_models()
    app.run(host="0.0.0.0", port=5000)
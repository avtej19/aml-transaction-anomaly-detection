# Model Card — AML Transaction Anomaly Detection

**Model type:** Unsupervised IsolationForest + Supervised RandomForest re-ranker

**Use case:** Flagging potentially suspicious transactions for human review in AML workflows.

**Metrics reported:** Precision@1%, ROC AUC, PR AUC, Recall at low FPR.

**Limitations & Risks:**
- Synthetic/data-shift: model was trained on synthetic data — real-world performance may differ.
- False positives cause operational cost; thresholds must be tuned for downstream capacity.
- Privacy: do not include PII in public datasets.

**Mitigations:**
- Human-in-loop review for top-k alerts.
- Regular retraining and drift detection (PSI/population stability index).
# Experiments Log

## Experiment 1 — DNN Baseline

Model:
Dense(128 → 64 → 32 → 16 → 1)

Result:
ROC-AUC: 0.6848

Observation:
Performance plateau suggests dataset complexity limitation.

---

## Experiment 2 — XGBoost

Parameters:
- trees: 500
- depth: 6
- early stopping: 20

Result:
ROC-AUC: 0.6817

Observation:
Reached convergence after 36 iterations.

---

## Experiment 3 — Ensemble

Weighted average:
DNN 55% + XGBoost 45%

Result:
ROC-AUC: 0.6882
F1 improved.
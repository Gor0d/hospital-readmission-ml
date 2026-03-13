# Model Card — Hospital Readmission Model

## Model Overview

Binary classification model predicting risk of hospital readmission within 30 days.

Architecture:
- DNN (Keras)
- XGBoost
- Weighted ensemble

## Intended Use

Assist healthcare teams in identifying high-risk patients before discharge.

Possible interventions:
- follow-up call
- medication review
- early outpatient appointment

## Training Data

Dataset:
- Synthetic dataset (10k patients)
- Optional: UCI Diabetes Hospital Dataset (~101k patients)

## Metrics

ROC-AUC: 0.69  
F1 Score: 0.76  
Recall: 0.91 (threshold 0.37)

## Limitations

- Dataset limited to diabetes patients
- Synthetic data may not capture real hospital complexity
- Not validated in clinical environment

## Ethical Considerations

Predictions should assist clinical decision-making, not replace it.
# 📝 Automated Essay Scoring 2.0

**Learning Agency Lab – Automated Essay Scoring**

📌 **Kaggle Competition:**
[https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/)

---

## 📊 Dataset Overview

* **Total Samples:** 17,307
* **Training Samples:** 13,845
* **Test Samples:** 3,462

Each sample consists of a student-written essay and a corresponding human-assigned score.
The objective is to automatically predict essay scores with high agreement to human raters.

**Evaluation Metric:** **Quadratic Weighted Kappa (QWK)**

---

## 🧪 Baseline: Classical NLP Methods

Classical machine learning models were trained using traditional NLP pipelines (TF-IDF features + regression models) to establish strong baselines.

### Model Performance (QWK)

| Model                       | Training QWK | Testing QWK |
| --------------------------- | ------------ | ----------- |
| Linear Regression           | 0.9750       | 0.4041      |
| **Ridge Regression**        | 0.8193       | **0.7044**  |
| Lasso                       | 0.3230       | 0.3230      |
| ElasticNet                  | 0.3230       | 0.3871      |
| SVR                         | 0.9569       | 0.6994      |
| Gradient Boosting Regressor | 0.6997       | 0.6426      |

### 🔍 Observations

* Linear Regression shows severe overfitting
* Ridge Regression generalizes best among linear models
* SVR performs competitively but is computationally expensive
* Tree-based models perform reasonably but lag behind regularized linear methods

---

## 🤖 Experiment 1: Fine-Tuning BERT (Baseline Transformer)

To capture semantic and contextual information beyond bag-of-words representations, a transformer-based approach was adopted.

### Training Configuration

* **Model:** Pretrained BERT
* **Epochs:** 3
* **Loss Function:** Mean Squared Error (MSE)
* **Evaluation Metric:** Quadratic Weighted Kappa (QWK)

### Evaluation Results

```json
{
  "epoch": 3.0,
  "eval_loss": 0.3556,
  "eval_qwk": 0.7874,
  "eval_runtime": 54.74,
  "eval_samples_per_second": 63.24,
  "eval_steps_per_second": 15.82
}
```

### Key Takeaway

Fine-tuning BERT significantly improves performance over classical NLP baselines, confirming that **contextual representations are critical for automated essay scoring**.

---

## 🧠 Experiment 2: From Architecture Choice → Representation Study

### Original Implicit Assumptions

The initial BERT-based approach embedded several assumptions:

* The `[CLS]` token sufficiently represents a long essay
* Mean Squared Error (MSE) is a suitable proxy for QWK
* Fixed-length padding does not affect learning
* Random train/validation splits are adequate

These assumptions were not independently testable in the original setup.

---

### Research Motivation

Treating **BERT + CLS pooling** as a fixed design choice assumes a single vector can summarize long-form, discourse-heavy text. From a research perspective, this hides strong assumptions about how meaning is aggregated across tokens.

---
### Result
Final QWK: 0.7863
```json
{'eval_loss': 0.3803684115409851, 
'eval_qwk': 0.7622537580649085, 
'eval_runtime': 44.6211, 
'eval_samples_per_second': 77.587, 
'eval_steps_per_second': 9.704, 
'epoch': 5.0}
```
---

## 🚀 Next Steps

* Pooling ablation studies
* Length-aware and discourse-aware representations
* Ordinal or QWK-aligned loss functions
* DeBERTa-based models and multi-task learning

# 📝 Automate-Essay-Scoring

**Learning Agency Lab – Automated Essay Scoring 2.0**

📌 Kaggle Competition:  
https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/

---

## 📊 Dataset Overview

- **Total Samples:** 17,307  
- **Training Samples:** 13,845  
- **Test Samples:** 3,462  

Each sample consists of a student-written essay and a corresponding human-assigned score.  
The goal is to automatically predict essay scores with high agreement to human raters.

The evaluation metric used throughout this project is **Quadratic Weighted Kappa (QWK)**.

---

## 🧪 Baseline: Classical NLP Methods

Classical machine learning models were trained using traditional NLP pipelines (e.g., TF-IDF features) to establish strong baselines.

### Model Performance (QWK)

| Model                        | Training QWK | Testing QWK |
|-----------------------------|--------------|-------------|
| Linear Regressor            | 0.9750       | 0.4041     |
| **Ridge Regression**        | 0.8193       | **0.7044** |
| Lasso                       | 0.3230       | 0.3230     |
| ElasticNet                  | 0.3230       | 0.3871     |
| SVR                         | 0.9569       | 0.6994     |
| Gradient Boosting Regressor | 0.6997       | 0.6426     |

### 🔍 Observations

- Linear Regression shows severe overfitting.
- Ridge Regression provides the best generalization among linear models.
- SVR performs competitively but is computationally expensive.
- Tree-based models perform reasonably but lag behind regularized linear approaches.

---

## 🤖 Fine-Tuning BERT

To capture semantic and contextual information beyond bag-of-words representations, a transformer-based approach was used.

### Training Configuration

- **Model:** BERT (pretrained)
- **Epochs:** 3
- **Loss Function:** Mean Squared Error (MSE)
- **Evaluation Metric:** Quadratic Weighted Kappa (QWK)

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

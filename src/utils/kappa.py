import numpy as np
from sklearn.metrics import cohen_kappa_score

def quadratic_weighted_kappa(y_true, y_pred):
    y_pred = np.rint(y_pred).astype(int)
    y_pred = np.clip(y_pred, y_true.min(), y_true.max())
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")
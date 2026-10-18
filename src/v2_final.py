# v2_final.py - final-model helpers used by NB3 (cells 6-7) and NB4.
#  - soft_voting(): the roadmap Soft-Voting ensemble (XGB + RF + CatBoost, equal weights)
#  - CVAP: Cross Venn-Abers predictors (Vovk et al. 2015), written out explicitly because the
#    'venn-abers' package does not install on Python 3.13. Per fold an inductive Venn-Abers
#    predictor gives (p0, p1); folds are merged with the GEOMETRIC-mean formula of Vovk 2015:
#    p = GM(p1) / (GM(1 - p0) + GM(p1)); uncertainty = mean interval width (p1 - p0).
#  - calibration_slope_intercept(): logistic recalibration of predictions (Cox 1958).
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from v2_models import build

SOFT_VOTE = ["XGB", "RF", "CatBoost"]


def soft_voting(params, y_train, seed=42):
    return VotingClassifier([(n, build(n, params[n], y_train=y_train, seed=seed))
                             for n in SOFT_VOTE], voting="soft", n_jobs=1)


def _ivap(cal_s, cal_y, test_s):
    p0, p1 = np.empty(len(test_s)), np.empty(len(test_s))
    for i, s in enumerate(test_s):
        for lab, out in ((0, p0), (1, p1)):
            ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            ir.fit(np.r_[cal_s, s], np.r_[cal_y, lab])
            out[i] = ir.predict([s])[0]
    return p0, p1


class CVAP:
    def __init__(self, base, n_splits=5, seed=42):
        self.base, self.n_splits, self.seed = base, n_splits, seed

    def fit(self, X, y):
        X = X.reset_index(drop=True); y = np.asarray(y)
        self.parts_ = []
        for tr, ca in StratifiedKFold(self.n_splits, shuffle=True,
                                      random_state=self.seed).split(X, y):
            m = clone(self.base).fit(X.iloc[tr], y[tr])
            self.parts_.append((m, m.predict_proba(X.iloc[ca])[:, 1], y[ca]))
        return self

    def predict(self, X):
        P0, P1 = [], []
        for m, cal_s, cal_y in self.parts_:
            p0, p1 = _ivap(cal_s, cal_y, m.predict_proba(X)[:, 1])
            P0.append(p0); P1.append(p1)
        P0, P1 = np.clip(np.array(P0), 1e-6, 1 - 1e-6), np.clip(np.array(P1), 1e-6, 1 - 1e-6)
        gm = lambda a: np.exp(np.log(a).mean(axis=0))
        p = gm(P1) / (gm(1 - P0) + gm(P1))
        return {"p": p, "p0": P0.mean(axis=0), "p1": P1.mean(axis=0),
                "width": (P1 - P0).mean(axis=0)}


def calibration_slope_intercept(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y).coef_[0][0]
    # calibration-in-the-large: intercept with the linear predictor as an offset
    from scipy.optimize import minimize_scalar
    nll = lambda a: -np.mean(y * (a + lp[:, 0]) - np.log1p(np.exp(a + lp[:, 0])))
    intercept = minimize_scalar(nll, bounds=(-5, 5), method="bounded").x
    return float(slope), float(intercept)


def youden_threshold(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    j = np.argmax(tpr - fpr)
    return float(thr[j]), float(tpr[j]), float(1 - fpr[j])

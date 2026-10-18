# v2_eval.py - test-set evaluation helpers (NB3 cell 7 and NB4).
#  - delong_test(): paired DeLong test for two correlated AUCs (DeLong 1988; fast algorithm of
#    Sun & Xu 2014)
#  - boot_ci(): stratified bootstrap percentile 95% CI for any metric
#  - threshold_metrics(), net_benefit()
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def _delong_components(y, scores):
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(np.r_[pos[r], neg[r]]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(cov)


def delong_test(y, p1, p2):
    """Two-sided p-value for AUC(p1) == AUC(p2) on the same patients."""
    y = np.asarray(y)
    aucs, cov = _delong_components(y, np.vstack([p1, p2]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(2 * stats.norm.sf(abs(z)))


def delong_ci(y, p):
    aucs, cov = _delong_components(np.asarray(y), np.atleast_2d(p))
    se = np.sqrt(cov[0, 0])
    return float(aucs[0]), float(aucs[0] - 1.96 * se), float(aucs[0] + 1.96 * se)


def boot_ci(y, p, fn, n_boot=2000, seed=42):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]
        vals.append(fn(y[idx], p[idx]))
    return float(fn(y, p)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def threshold_metrics(y, p, thr):
    y, pred = np.asarray(y), (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sd = lambda a, b: a / b if b else np.nan
    return {"sensitivity": sd(tp, tp + fn), "specificity": sd(tn, tn + fp),
            "ppv": sd(tp, tp + fp), "npv": sd(tn, tn + fn), "tp": tp, "fp": fp, "tn": tn,
            "fn": fn}


def net_benefit(y, p, thresholds):
    y, p = np.asarray(y), np.asarray(p)
    n, prev = len(y), y.mean()
    rows = []
    for t in thresholds:
        pred = p >= t
        tp = ((pred) & (y == 1)).sum(); fp = ((pred) & (y == 0)).sum()
        rows.append({"threshold": t, "model": tp / n - fp / n * t / (1 - t),
                     "treat_all": prev - (1 - prev) * t / (1 - t), "treat_none": 0.0})
    return rows


AUC = roc_auc_score
AP = average_precision_score
BRIER = brier_score_loss

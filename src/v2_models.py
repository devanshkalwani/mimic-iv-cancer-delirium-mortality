# v2_models.py - model definitions + Optuna search spaces (one copy used by NB3 and NB4).
# Class imbalance: class weights in every model (no resampling) unless cell 4 shows
# SMOTEENN is better by > 0.005 CV AUC (rule fixed in advance).
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

MODELS = ["LR", "DT", "RF", "XGB", "CatBoost", "LightGBM"]


# Search ranges. Widened after the first run, where XGB/LightGBM depth, XGB min_child_weight
# and RF depth hit the edge of the range (an optimum at the edge may lie outside it).
# kind: "int" | "float" | "logfloat" | "cat"
SPACES = {
    "LR": {"C": ("logfloat", 1e-4, 100)},
    "DT": {"max_depth": ("int", 2, 12), "min_samples_leaf": ("int", 5, 150)},
    "RF": {"n_estimators": ("int", 300, 1500, 100), "max_depth": ("int", 3, 30),
           "min_samples_leaf": ("int", 1, 30),
           "max_features": ("cat", ["sqrt", "log2", 0.3, 0.5])},
    "XGB": {"n_estimators": ("int", 200, 2000, 100), "learning_rate": ("logfloat", 0.003, 0.1),
            "max_depth": ("int", 1, 6), "subsample": ("float", 0.4, 1.0),
            "colsample_bytree": ("float", 0.3, 1.0), "min_child_weight": ("int", 1, 60),
            "reg_lambda": ("logfloat", 0.01, 30)},
    "CatBoost": {"iterations": ("int", 300, 2000, 100),
                 "learning_rate": ("logfloat", 0.003, 0.1),
                 "depth": ("int", 2, 8), "l2_leaf_reg": ("logfloat", 0.5, 30),
                 "bagging_temperature": ("float", 0, 1)},
    "LightGBM": {"n_estimators": ("int", 200, 2000, 100),
                 "learning_rate": ("logfloat", 0.003, 0.1),
                 "num_leaves": ("int", 3, 63), "max_depth": ("int", 1, 8),
                 "subsample": ("float", 0.4, 1.0), "colsample_bytree": ("float", 0.3, 1.0),
                 "min_child_samples": ("int", 5, 200)},
}


def space(name, t):
    p = {}
    for k, spec in SPACES[name].items():
        kind = spec[0]
        if kind == "int":
            step = spec[3] if len(spec) > 3 else 1
            p[k] = t.suggest_int(k, spec[1], spec[2], step=step)
        elif kind == "float":
            p[k] = t.suggest_float(k, spec[1], spec[2])
        elif kind == "logfloat":
            p[k] = t.suggest_float(k, spec[1], spec[2], log=True)
        else:
            p[k] = t.suggest_categorical(k, spec[1])
    return p


def at_boundary(name, best):
    """Numeric parameters whose best value sits on the edge of the search range."""
    out = []
    for k, spec in SPACES[name].items():
        if spec[0] == "cat":
            continue
        lo, hi = spec[1], spec[2]
        v = best[k]
        if spec[0] == "logfloat":
            near = v <= lo * 1.05 or v >= hi / 1.05
        else:
            step = spec[3] if len(spec) > 3 else (1 if spec[0] == "int" else (hi - lo) * 0.02)
            near = v <= lo + step * 0.5 or v >= hi - step * 0.5
        if near:
            out.append(f"{k}={v}")
    return out


def build(name, p, y_train=None, seed=42, weighted=True):
    """y_train is only used to count classes for XGBoost's weight (no outcome leakage:
    it is the training fold's own labels)."""
    if name == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=p["C"], class_weight="balanced" if weighted else None, max_iter=5000))
    if name == "DT":
        return DecisionTreeClassifier(class_weight="balanced" if weighted else None,
                                      random_state=seed, **p)
    if name == "RF":
        return RandomForestClassifier(class_weight="balanced_subsample" if weighted else None,
                                      n_jobs=-1, random_state=seed, **p)
    if name == "XGB":
        spw = float((y_train == 0).sum() / (y_train == 1).sum()) if weighted else 1.0
        return XGBClassifier(scale_pos_weight=spw, eval_metric="logloss", n_jobs=-1,
                             random_state=seed, tree_method="hist", **p)
    if name == "CatBoost":
        return CatBoostClassifier(auto_class_weights="Balanced" if weighted else None,
                                  random_seed=seed, verbose=0, thread_count=-1,
                                  allow_writing_files=False, **p)
    if name == "LightGBM":
        return LGBMClassifier(class_weight="balanced" if weighted else None, subsample_freq=1,
                              random_state=seed, n_jobs=-1, verbose=-1, **p)
    raise ValueError(name)

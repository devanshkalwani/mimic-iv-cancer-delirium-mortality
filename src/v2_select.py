# v2_select.py - SHAP-RFE feature selection (CatBoost + TreeSHAP), used on a TRAINING set
# or inside a training fold only. Rules fixed in advance:
#  - each round: 5-fold inner CV, CatBoost (balanced class weights), SHAP importance
#    measured on the inner validation parts, weakest 20% of features removed
#  - stop at 10 features
#  - chosen set = the SMALLEST set whose inner-CV AUC is within 0.005 of the best round
#  - STABILITY SELECTION: the whole SHAP-RFE is repeated with 5 different fold seeds;
#    final set = features chosen in >= 3 of the 5 runs (majority)
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

STEP, MIN_FEATURES, INNER_FOLDS, TOL = 0.20, 10, 5, 0.005
SEEDS, MIN_VOTES = [42, 43, 44, 45, 46], 3


def _catboost(seed):
    return CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
                              auto_class_weights="Balanced", random_seed=seed,
                              verbose=0, thread_count=-1, allow_writing_files=False)


def shap_rfe(X, y, seed=42):
    X = X.reset_index(drop=True); y = pd.Series(y).reset_index(drop=True)
    feats, history = list(X.columns), []
    skf = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
    while True:
        aucs, imp = [], np.zeros(len(feats))
        for tr, va in skf.split(X, y):
            m = _catboost(seed).fit(X.loc[tr, feats], y[tr])
            aucs.append(roc_auc_score(y[va], m.predict_proba(X.loc[va, feats])[:, 1]))
            sv = m.get_feature_importance(Pool(X.loc[va, feats], y[va]), type="ShapValues")
            imp += np.abs(sv[:, :-1]).mean(axis=0)
        history.append({"n_features": len(feats), "inner_auc": float(np.mean(aucs)),
                        "features": list(feats),
                        "importance": dict(zip(feats, imp / INNER_FOLDS))})
        if len(feats) <= MIN_FEATURES:
            break
        keep = max(MIN_FEATURES, int(len(feats) * (1 - STEP)))
        feats = [feats[i] for i in np.argsort(-imp)[:keep]]
    best = max(h["inner_auc"] for h in history)
    chosen = min([h for h in history if h["inner_auc"] >= best - TOL],
                 key=lambda h: h["n_features"])
    return chosen["features"], chosen["importance"], history


def stability_select(X, y):
    """SHAP-RFE repeated over SEEDS; keep features chosen in >= MIN_VOTES runs."""
    votes, shap_sum, runs = {}, {}, []
    for s in SEEDS:
        feats, imp, hist = shap_rfe(X, y, seed=s)
        runs.append({"seed": s, "n_chosen": len(feats),
                     "best_inner_auc": max(h["inner_auc"] for h in hist)})
        for f in feats:
            votes[f] = votes.get(f, 0) + 1
            shap_sum[f] = shap_sum.get(f, 0) + imp[f]
    tab = pd.DataFrame({"feature": list(votes), "votes": list(votes.values()),
                        "mean_abs_shap": [shap_sum[f] / votes[f] for f in votes]})
    tab = tab.sort_values(["votes", "mean_abs_shap"], ascending=False).reset_index(drop=True)
    final = tab.feature[tab.votes >= MIN_VOTES].tolist()
    return final, tab, pd.DataFrame(runs)


def evaluate_set(X, y, feats, seed=2024):
    """Common yardstick for comparing feature pools: default CatBoost, 5x3 repeated CV,
    fold seed different from the selection seeds. TRAINING data only."""
    from sklearn.model_selection import RepeatedStratifiedKFold
    X = X.reset_index(drop=True); y = pd.Series(y).reset_index(drop=True)
    aucs = []
    for tr, va in RepeatedStratifiedKFold(n_splits=5, n_repeats=3,
                                          random_state=seed).split(X, y):
        m = _catboost(seed).fit(X.loc[tr, feats], y[tr])
        aucs.append(roc_auc_score(y[va], m.predict_proba(X.loc[va, feats])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


# ---------------- measurement-process features (how often something was charted) ----------------
def is_measurement(f):
    return f.endswith(("_n", "_obs_hours")) or f in ("Lactate_measured",
                                                     "tpife_n_vitals_observed")


# ---------------- feature family (one definition used in NB3 and NB4) ----------------
TPIFE_SUFFIX = ("_frac_abn", "_frac_severe", "_auc_dev", "_transitions_per24h",
                "_episodes_per24h", "_longest_episode_h", "_first_episode_h", "_unresolved",
                "_deterioration", "_max_dev", "_ever_abn", "_ever_severe", "_last_state",
                "_late_minus_early_mean")
ORGAN_SUPPORT = {"vaso_any", "dobutamine_any", "n_vasoactive_agents", "nee_max",
                 "nee_mean_24h", "vaso_hours", "fio2_max", "pf_min", "sf_min",
                 "resp_support_level", "uo_ml_kg_h"}


def feature_family(f):
    if f.startswith("csi_"): return "Composite Stability Index"
    if f.startswith("osbi_"): return "organ burden (OSBI)"
    if f.startswith("stab_"): return "stability component"
    if f.startswith("tpife_"): return "TPIFE stage 4 (aggregation)"
    if f.startswith("elix_"): return "comorbidity (Elixhauser)"
    if f.startswith("cancer_") or f in ("age_at_admission", "sex_male"):
        return "demographics/cancer"
    if f.startswith(("sofa_", "ap_", "apache_")): return "severity score"
    if f in ("weight_kg", "gcs_verbal_ETT_24h"): return "context"
    if f in ORGAN_SUPPORT: return "organ support"
    if f.startswith(("sirs_", "hypotension_", "hypoxaemia_", "creat_rise", "shock_index")):
        return "interaction / SIRS"
    if f.endswith(TPIFE_SUFFIX) or any(s in f for s in ("_early_", "_mid_", "_late_")):
        return "TPIFE stages 1-3"
    return "basic temporal"


TPIFE_FAMILIES = {"TPIFE stages 1-3", "TPIFE stage 4 (aggregation)",
                  "Composite Stability Index", "stability component", "organ burden (OSBI)"}

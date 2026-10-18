# v2_prep.py - training-only preprocessing used by every Notebook 3/4 v2 cell.
# Everything that is LEARNED from data (20% rule, constant/duplicate columns, clipping
# limits, MICE, Stability-Index scaling, standard scaling) is fitted on the rows passed
# to Prep.fit() - always a TRAINING set or a training fold - and only APPLIED elsewhere.
# The outcome is never passed in.
import numpy as np, pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

P2 = "../data/processed_v2/"
ID_COLS = ["stay_id", "subject_id", "hadm_id"]
LABEL = "mortality_28d"
SPLIT_TAGS = ["est_calendar_year", "temporal_set"]
CSI_VITALS = ["HR", "MAP", "RR", "SpO2"]
# Lactate is a Sepsis-3 criterion and a core ICU mortality marker. Its missingness (~20%)
# sits exactly on the 20% cut-off, so the rule would keep or drop it by chance of the split.
# Pre-specified clinical exception: kept if <= 30% missing in the training rows.
PROTECTED = ["Lactate_mean", "Lactate_min", "Lactate_max", "Lactate_first", "Lactate_last",
             "Lactate_ever_abn", "Lactate_ever_severe", "Lactate_last_state",
             "hypotension_x_lactate"]
PROTECTED_MAX_MISSING = 0.30
CANCER_LEVELS = ["Gastrointestinal", "Genitourinary", "Hematologic",
                 "Other/Unspecified", "Respiratory/Thoracic"]


def load_table():
    T = pd.read_csv(P2 + "model_table_v2.csv")
    S = pd.read_csv(P2 + "splits_v2.csv")
    assert (T.stay_id.values == S.stay_id.values).all(), "split file out of order"
    feats = [c for c in T.columns if c not in ID_COLS + [LABEL] + SPLIT_TAGS]
    return T, S, feats


def add_fixed_features(X):
    """Fixed-rule features (no fitting): maximum deviation beyond the normal band."""
    X = X.copy()
    # a count of measurements / observed hours that is missing means "none": 0, not unknown
    for c in [c for c in X.columns if c.endswith(("_n", "_obs_hours"))]:
        X[c] = X[c].fillna(0)
    X["Lactate_measured"] = (X["Lactate_n"] > 0).astype(float)
    th = pd.read_csv(P2 + "tpife_thresholds.csv").set_index("variable")
    for v in CSI_VITALS:
        ml, mh = th.loc[v, "mild_low"], th.loc[v, "mild_high"]
        lo, hi = X[f"{v}_min"], X[f"{v}_max"]
        d = pd.Series(0.0, index=X.index)
        if pd.notna(ml):
            d = np.maximum(d, ml - lo)
        if pd.notna(mh):
            d = np.maximum(d, hi - mh)
        d[lo.isna() | hi.isna()] = np.nan
        X[f"{v}_max_dev"] = d
    return X


def one_hot(X):
    X = X.copy()
    for lv in CANCER_LEVELS:
        X["cancer_" + lv.split("/")[0].lower()] = (X.cancer_type == lv).astype(float)
    return X.drop(columns="cancer_type")


class Prep:
    def __init__(self, seed=42, max_missing=0.20):
        self.seed, self.max_missing = seed, max_missing

    def fit(self, X):
        X = one_hot(X)
        miss = X.isna().mean()
        self.train_missing = miss
        limit = pd.Series(self.max_missing, index=miss.index)
        limit[[c for c in PROTECTED if c in limit.index]] = PROTECTED_MAX_MISSING
        self.dropped_missing = sorted(miss[miss > limit].index)
        X = X.drop(columns=self.dropped_missing)
        nun = X.nunique(dropna=True)
        self.dropped_constant = sorted(nun[nun <= 1].index)
        X = X.drop(columns=self.dropped_constant)
        dup = X.T.duplicated()
        self.dropped_duplicate = sorted(X.columns[dup.values])
        X = X.drop(columns=self.dropped_duplicate)
        self.cols = list(X.columns)
        self.binary = [c for c in self.cols if set(X[c].dropna().unique()) <= {0, 1}]
        self.cont = [c for c in self.cols if c not in self.binary]
        self.lo = X[self.cont].quantile(0.005)
        self.hi = X[self.cont].quantile(0.995)
        X[self.cont] = X[self.cont].clip(self.lo, self.hi, axis=1)
        self.vmin, self.vmax = X.min(), X.max()
        self.imp = IterativeImputer(estimator=BayesianRidge(), max_iter=10,
                                    n_nearest_features=50, random_state=self.seed,
                                    skip_complete=True)
        self.imp.fit(X)
        Xi = self._finish_impute(X)
        self.csi_parts = {v: [c for c in [f"{v}_transitions_per24h", f"{v}_episodes_per24h",
                                          f"{v}_frac_abn", f"{v}_max_dev",
                                          f"{v}_longest_episode_h", f"stab_{v}_arv",
                                          f"stab_{v}_cv"] if c in Xi.columns]
                          for v in CSI_VITALS}
        allp = [c for p in self.csi_parts.values() for c in p]
        self.csi_mu, self.csi_sd = Xi[allp].mean(), Xi[allp].std().replace(0, 1)
        Xi = self._add_csi(Xi)
        self.out_cols = list(Xi.columns)
        self.scaler = StandardScaler().fit(Xi)
        return self

    def _finish_impute(self, X):
        Xi = pd.DataFrame(self.imp.transform(X[self.cols]), columns=self.cols, index=X.index)
        Xi = Xi.clip(self.vmin, self.vmax, axis=1)           # stay inside training range
        Xi[self.binary] = Xi[self.binary].round()
        return Xi

    def _add_csi(self, Xi):
        Xi = Xi.copy()
        for v, parts in self.csi_parts.items():
            if parts:
                z = (Xi[parts] - self.csi_mu[parts]) / self.csi_sd[parts]
                Xi[f"csi_{v}"] = z.mean(axis=1)               # higher = LESS stable
        cs = [f"csi_{v}" for v in CSI_VITALS if f"csi_{v}" in Xi]
        Xi["csi_total"] = Xi[cs].mean(axis=1)
        return Xi

    def transform(self, X, scaled=False):
        X = one_hot(X)
        X = X[self.cols].copy()
        X[self.cont] = X[self.cont].clip(self.lo, self.hi, axis=1)
        Xi = self._add_csi(self._finish_impute(X))[self.out_cols]
        if scaled:
            Xi = pd.DataFrame(self.scaler.transform(Xi), columns=self.out_cols, index=Xi.index)
        return Xi

    def complete_case_mask(self, X):
        """Patients with every kept (<=20% missing) feature actually measured."""
        X = one_hot(X)
        return X[self.cols].notna().all(axis=1)

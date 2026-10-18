# v2_nb4.py - shared helpers for Notebook 4 (results, figures, tables). READ-ONLY on NB3.
#  - freeze_check(): every NB4 cell first confirms that no frozen NB1-NB3 output has changed
#    since NB4 started (md5 manifest written once in NB4 cell 1)
#  - journal figure style (Okabe-Ito colour-blind-safe palette, 300 dpi PNG + vector PDF)
import os, glob, json, hashlib
import matplotlib
import matplotlib.pyplot as plt

P2, M2, FIG = "../data/processed_v2/", "../models_v2/", "../figures_v2/"
OUT4 = P2 + "nb4/"
ARMS = ["random_split", "temporal_split"]
ARM_LABEL = {"random_split": "Random-split test (n=556)",
             "temporal_split": "Temporal (2018-2022) test (n=1,404)"}
FROZEN = ["model_table_v2.csv", "splits_v2.csv", "cohort_final.csv",
          "cohort_delirium_timing.csv", "prep_*_t*.csv", "selected_features_*.csv",
          "train_oof_final_*.csv", "test_*.csv", "cv/*.csv", "pilot_labs/*.csv"]
FROZEN_M = ["final_*.joblib", "prep_*.joblib", "tuning_*.json", "locked_rules_*.json",
            "feature_pool_decision.json", "test_opened_log.json"]
MANIFEST = OUT4 + "frozen_manifest.json"

# Okabe-Ito palette (colour-blind safe)
C = {"model": "#0072B2", "stack": "#56B4E9", "lr": "#009E73", "sofa": "#E69F00",
     "apache": "#D55E00", "grey": "#999999", "black": "#000000", "pink": "#CC79A7",
     "yellow": "#F0E442"}


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _files():
    fs = [f for p in FROZEN for f in glob.glob(P2 + p)]
    fs += [f for p in FROZEN_M for f in glob.glob(M2 + p)]
    return sorted(set(fs))


def make_manifest():
    man = {os.path.relpath(f, ".."): _md5(f) for f in _files()}
    json.dump(man, open(MANIFEST, "w"), indent=1)
    return man


def freeze_check():
    """Stop if any frozen NB1-NB3 output changed after NB4 started."""
    man = json.load(open(MANIFEST))
    bad = [k for k, v in man.items()
           if not os.path.exists("../" + k) or _md5("../" + k) != v]
    assert not bad, f"FROZEN FILES CHANGED - stop and report: {bad[:5]}"
    return len(man)


def style():
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, name):
    """Save a figure as 300-dpi PNG and editable vector PDF in figures_v2/."""
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(FIG + name + ".png")
    fig.savefig(FIG + name + ".pdf")
    print(f"   saved figures_v2/{name}.png + .pdf")

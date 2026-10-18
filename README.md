# 28-day mortality prediction in ICU patients with cancer and delirium (MIMIC-IV v3.1)

Code and aggregate results for the article *(title and DOI added on publication)*.
The analysis follows TRIPOD+AI (Collins et al., BMJ 2024;385:e078378).

## What is in this repository
| Folder | Content |
|---|---|
| `notebooks/` | The four analysis notebooks, in run order (outputs removed) |
| `src/` | Helper modules imported by the notebooks (`v2_*.py`) |
| `results/` | Aggregate result tables: paper tables (`tables/`), NB3 test/CV summaries (`nb3/`), verification reports (`verification/`) |
| `figures/` | All paper figures (300-dpi PNG + vector PDF) |

Pipeline: **NB1** cohort (cancer + delirium, first ICU stay, alive and in ICU at 24 h; 2,777 patients) ->
**NB2** first-24-h features incl. the TPIFE temporal framework (391 candidates) ->
**NB3** training-only preparation, stability SHAP-RFE feature selection, Optuna tuning, nested 5x5
cross-validation, pre-specified model choice (Soft-Voting XGBoost + random forest + CatBoost, isotonic
calibration, Cross Venn-Abers uncertainty), locked thresholds/risk tiers, one-time test evaluation on a random
80/20 split and a temporal split (<=2017 / >2017) ->
**NB4** figures, tables, SHAP explanations, risk tiers, subgroup/fairness analysis, sample size (Riley),
three independent verification passes.

NB1 and NB2 are the complete working record and also contain earlier, superseded steps; the final pipeline is
the one that writes to the `*_v2` folders (`data/processed_v2/`, `models_v2/`, `figures_v2/`). Individual record
identifiers typed into superseded spot-check cells have been replaced by `REDACTED_ID_<k>` (those cells are not
part of the final pipeline).

## Data access
No MIMIC data are included. MIMIC-IV v3.1 is available to credentialed users from PhysioNet
(https://physionet.org/content/mimiciv/) after completing the required training and signing the data use agreement.
Place the raw `.csv.gz` files in `data/raw/` and run the notebooks in order from the `notebooks/` folder
(copy `src/*.py` next to the notebooks). Outputs are written to `data/processed_v2/`, `models_v2/`, `figures_v2/`.

## Trained models
The trained model files are not posted publicly because they were fitted on credentialed patient data.
Everything needed to rebuild them exactly is here: the code, the selected features
(`results/nb3/selected_features_*.csv`), the tuned hyperparameters (`results/nb3/tuning_*.json`,
`results/tables/S6_hyperparameters.csv`), fixed random seeds, and the locked decision threshold and risk-tier
cut-offs (`results/nb3/locked_rules_*.json`). Credentialed MIMIC-IV users may request the fitted model files
from the authors.

## Environment
Python 3.13.9; exact package versions in `requirements.txt`.

## Licence
Code: MIT (see `LICENSE`). Use of MIMIC-IV data is governed by the PhysioNet credentialed data use agreement.

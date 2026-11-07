# ClimbMLCode

Code and results for **"Evaluating Class-Imbalanced Learning Methods Across Base Classifiers: A Focused Extension of the CLIMB Benchmark"** by Rabia Habib (Department of Computer Science, Kinnaird College for Women University, Lahore, Pakistan).

## What this project does

[CLIMB](https://arxiv.org/abs/2505.17451) (Liu et al., NeurIPS 2025) is a benchmark of 29 class-imbalanced learning (CIL) methods on 73 tabular datasets. All of its experiments use a **Decision Tree** as the base classifier, and it lists non-tree base learners as future work.

This repository extends a slice of that benchmark. It reruns CIL methods with a second base classifier, **Logistic Regression**, and asks whether method performance and ranking stay the same when the base classifier changes.

- **Datasets:** 20 CLIMB datasets, fetched from OpenML and converted to binary problems.
- **Methods:** 11 configurations (table below), including two controls: a class-weighted classifier (`cost`) and plain bagging with no balancing (`bag`).
- **Base classifiers:** Decision Tree and Logistic Regression.
- **Metrics:** AUPRC, Macro F1 and balanced accuracy are recorded for every run.
- **Analysis (on AUPRC):** Kendall's tau-b between the Decision Tree and Logistic Regression method rankings, Friedman test, pairwise Wilcoxon signed-rank tests with Holm correction, Critical Difference diagrams, and ensemble gain over the no-balancing baseline.

## Methods

| Code | Method | Family | Implementation |
|---|---|---|---|
| `base` | No balancing | baseline | the base classifier alone |
| `rus` | Random undersampling | undersampling | `imblearn` `RandomUnderSampler` |
| `enn` | Edited nearest neighbours | cleaning | `imblearn` `EditedNearestNeighbours` |
| `tomek` | Tomek links | cleaning | `imblearn` `TomekLinks` |
| `smote` | SMOTE | oversampling | `imblearn` `SMOTE` |
| `cost` | Class-weighted classifier | cost sensitive | `class_weight='balanced'` on the base classifier |
| `spe` | Self-paced ensemble | undersample ensemble | `imbalanced_ensemble` `SelfPacedEnsembleClassifier` |
| `ubag` | UnderBagging | undersample ensemble | `imbalanced_ensemble` `UnderBaggingClassifier` |
| `obag` | OverBagging | oversample ensemble | `imbalanced_ensemble` `OverBaggingClassifier` |
| `sbag` | SMOTEBagging | oversample ensemble | `imbalanced_ensemble` `SMOTEBaggingClassifier` |
| `bag` | Plain bagging, no balancing | standard ensemble | `sklearn` `BaggingClassifier` |

`bag` is a control. It shows how much of an ensemble's gain comes from averaging many models rather than from balancing.

## Datasets

mc2, vertebra-column, haberman, vehicle, wholesale-customers, ilpd-numeric, planning-relax, blood-transfusion-service-center, SPECTF, JapaneseVowels, ibm-employee-attrition, kc1, thoracic-surgery, pc4, adult, bank-marketing, vowel, law-school-admission-bianry, Credit_Approval_Classification, phoneme.

The `load_dataset` function in the notebook:

- fetches each dataset from OpenML (`vertebra-column` version 2, `vehicle` version 1, latest version for the rest), retrying with exponential backoff on network errors;
- caches each dataset as a pickle in `results/dataset_cache/`, so later runs make no network calls;
- drops rows with a missing target and sets the **minority (least frequent) class as the positive class (label 1)**. For `vehicle` the positive class is the most frequent of its classes, following the CLIMB setup;
- converts features to numeric, one-hot encodes categorical features with `pd.get_dummies(drop_first=True)`, and fills missing values with 0. This is done once, before cross-validation, and uses no label information.

Imbalance ratio (IR) is the majority count divided by the minority count.

## Experimental setup

- **Cross-validation:** `RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)`, which gives 25 folds per dataset. `results_full_sweep.csv` holds all 25. `final_submission_results.csv` keeps folds 0 to 4 (the first repeat), and the statistical analysis uses that file.
- **Ensemble size:** 10 estimators for `spe`, `ubag`, `obag`, `sbag` and `bag`.
- **Hyperparameters:** library defaults, no tuning. Logistic Regression uses `max_iter=1000`. `random_state=42` is set on the classifiers, ensembles, `RandomUnderSampler` and `SMOTE`.
- **Pipeline:** `imblearn.pipeline.Pipeline` with median imputation, then `StandardScaler`, then the resampler (or `passthrough`), then the classifier. All steps are fitted on the training fold only, and the held-out fold is never resampled.
- **Metrics per run:** AUPRC (`average_precision_score` on predicted probabilities), Macro F1 and balanced accuracy.
- **Decision Tree scores:** for the no-balancing Decision Tree, the number of distinct predicted probabilities per fold is logged in the `notes` column as `Unique_Probs:<n>`.

### Differences from the original CLIMB setup

| | CLIMB | This work |
|---|---|---|
| Base classifier | Decision Tree | Decision Tree and Logistic Regression |
| Ensemble size | 100 | 10 |
| Hyperparameters | Tuned | Library defaults |
| Datasets | 73 | 20 |
| Methods | 29 | 11 |

## Reproduction check

The first cells print the imbalance ratio of four datasets (mc2, vertebra-column, haberman, vehicle) to confirm the binarisation matches CLIMB, and a later cell tabulates mean Decision Tree AUPRC per dataset and method for comparison with CLIMB's reported values.

## Repository contents

| File | Purpose |
|---|---|
| `mlscratchcode.ipynb` | The full pipeline: loader, dry run, full sweep, reformatting, statistics and figures |
| `results_full_sweep.csv` | Raw results, all 25 folds |
| `final_submission_results.csv` | Reformatted results, folds 0 to 4, used for the analysis |
| `results/dataset_cache/` | Cached OpenML datasets (created on first run) |
| `figure1_kendall_tau_distribution.png` | Distribution of Kendall's tau across datasets |
| `figure2_average_ranks_comparison.png` | Average method ranks, Decision Tree and Logistic Regression side by side |
| `figure2_critical_difference_diagrams.png` | Critical Difference diagrams, bars drawn from Wilcoxon-Holm p-values (alpha = 0.05) |
| `figure3_ensemble_gain_comparison.png` | Ensemble gain, Decision Tree against Logistic Regression, one point per dataset |

## Setup

The notebook was written for Kaggle Notebooks (CPU only, no GPU needed). The first cell installs:

```bash
pip install imbalanced-learn imbalanced-ensemble scikit-learn pandas numpy scipy
```

The analysis cells also use `matplotlib`, `seaborn` and `statsmodels`, which Kaggle provides by default.

**Compatibility patches.** The installed `imbalanced-ensemble` imports two helpers that newer scikit-learn versions removed (`_print_elapsed_time` and `if_delegate_has_method`). The first cell edits the installed package's `pipeline.py` to replace them, and later cells add small stand-ins for `_print_elapsed_time` and `Pipeline._check_fit_params`. Run these cells before any import from `imbalanced_ensemble`.

## Running the experiments

Run the notebook cells in order:

1. Install packages and patch `imbalanced-ensemble`.
2. Define `load_dataset` and check the imbalance ratios of the four reproduction datasets.
3. Optional dry run on `mc2` with 2-fold cross-validation.
4. Full sweep. Settings (`RUN_ID`, `OUTPUT_CSV`, `CV_STRATEGY`, the dataset, method and classifier lists) are defined at the top of the cell.
5. Reformat the results into `final_submission_results.csv`.
6. Kendall's tau, Friedman test, Wilcoxon-Holm post-hoc tests, Critical Difference diagrams and ensemble gain.

Every metric value is appended to the CSV as soon as it is computed, and combinations already in the file are skipped, so an interrupted sweep can be restarted. A fit that raises an error is written with status `error`, empty metric values and the error message in `notes`.

## Output format

`results_full_sweep.csv` has one row per dataset, method, classifier, fold and metric:

| Column | Meaning |
|---|---|
| `run_id`, `timestamp` | Which sweep produced the row, and when |
| `dataset`, `n_samples`, `n_features`, `imbalance_ratio` | Dataset and its properties (features counted after encoding) |
| `method`, `base_classifier` | Method code and `decision_tree` or `logistic_regression` |
| `fold` | Fold index, 0 to 24 |
| `metric`, `value` | `auprc`, `macro_f1` or `balanced_accuracy`, and its value |
| `fit_seconds`, `status`, `notes` | Fit time, `ok` or `error`, and any message |

`final_submission_results.csv` keeps folds 0 to 4 and adds:

| Column | Meaning |
|---|---|
| `ir_group` | `low` (IR below 9), `medium` (9 to below 20), `high` (20 to below 50), `extreme` (50 and above) |
| `method_family` | Family from the methods table |
| `ensemble_size` | 10 for `spe`, `ubag`, `obag`, `sbag` and `bag`; 1 for all other methods |
| `tuning_mode` | Always `default` |


## Contact

Rabia Habib, Department of Computer Science, Kinnaird College for Women University, Lahore, Pakistan.

Email: rhkhan0411@gmail.com

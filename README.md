# CLIMB ML Code Scratch

Research code and notebook artifacts for the CLIMB work on class-imbalanced learning for tabular data.

## Paper

**CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data**

The repository's source project includes a paper PDF whose filename identifies this title. Please confirm the final published title before citing this repository.

## Contents

- `mlscratchcode.ipynb` — the notebook to be uploaded, with its saved cell outputs embedded in the notebook.
- `results/` — only add result files that are needed to reproduce or understand the notebook; avoid uploading caches, temporary files, or restricted datasets.

Notebook outputs (tables, printed experiment summaries, and figures) are saved inside the `.ipynb` file, so a separate output folder is not required for those saved results.

## Tools

The notebook uses Python and scientific Python / machine-learning packages, including NumPy, pandas, scikit-learn, imbalanced-learn, imbalanced-ensemble, SciPy, Matplotlib, and Seaborn. It fetches benchmark datasets through OpenML and uses local result files for some analysis steps.

## Running the notebook

1. Install Python and Jupyter.
2. Install the packages imported by the notebook (for example: `numpy`, `pandas`, `scikit-learn`, `imbalanced-learn`, `imbalanced-ensemble`, `scipy`, `matplotlib`, and `seaborn`).
3. Open `mlscratchcode.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab and run the cells in order.
4. Some later analysis cells expect a generated or supplied `results_full_sweep.csv`. The notebook also writes `final_submission_results.csv` and a Kendall-tau figure when those cells are run. These files are not included unless separately added to the repository.

## Data and outputs

The notebook's saved outputs are included inline. Dataset caches and raw or restricted datasets should not be committed; use the notebook's dataset-fetching code where possible. Review all outputs and files for sensitive information before making this repository public.

## Availability

This repository is intended to be made publicly accessible for research-code availability. Verify its visibility in GitHub repository **Settings → General → Danger Zone → Change visibility** before sharing the link.

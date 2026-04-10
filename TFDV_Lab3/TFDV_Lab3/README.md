# Adult Income Data Validation with TFDV

## Overview
This project replicates the TensorFlow Data Validation (TFDV) workflow from the original Coursera MLOps lab (`C2W1_Assignment.ipynb`) using a different dataset: **UCI Adult Income**.

The notebook demonstrates data validation across training, evaluation, and serving datasets, including schema inference, anomaly checks, schema environments, drift/skew checks, sliced statistics, and schema freezing.

### What I changed from the original lab
1. Replaced original dataset with **UCI Adult Income**.
2. Added Adult-specific preprocessing:
   - explicit header mapping
   - missing value handling for `?` / ` ?`
   - row cleaning with null removal
3. Created 70/15/15 `train/eval/serving` split and removed label (`income`) from serving.
4. Removed irrelevant feature (`fnlwgt`) from validation flow.
5. Added schema environment handling for serving where `income` is intentionally absent.
6. Added controlled synthetic-shift experiments to demonstrate drift/skew detection behavior.
7. Updated slice API usage for TFDV 1.15.1 compatibility.

## Dataset
- Source: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
- Features:
  `age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country, income`
- Label: `income`

## Environment
- Runtime: Google Colab Runtime version 2025.07
- Python 3.11
- GPU: T4

### Main package versions
- `tensorflow==2.15.0`
- `tensorflow-data-validation==1.15.1`
- `tensorflow-metadata==1.15.0`
- `tfx-bsl==1.15.1`
- `apache-beam==2.53.0`
- `numpy==1.24.3`
- `pandas==1.5.3`
- `pyarrow==10.0.1`

## Project Structure
- `Adult_Income_TFDV.ipynb` - main notebook
- `data/adult_income.csv`
- `“train.csv, eval.csv, serving.csv, and synthetic demo files are generated during notebook          execution.”`
- `output/schema.pbtxt` - frozen validated schema

## Notebook Workflow
The notebook follows the same section headings as the original lab:
1. Setup and imports
2. Load dataset and split
3. Generate/visualize training statistics
4. Infer schema
5. Compare eval stats and detect/fix anomalies
6. Schema environments for serving data
7. Drift/skew checks
8. Sliced statistics (by `occupation`)
9. Freeze schema

## Key Results
1. **Training and evaluation validation** completed successfully (eval may show no anomalies on clean random split).
2. **Serving validation before environments** flags missing `income` as expected.
3. **Serving validation after environments** resolves expected label-missing anomaly.
4. **Drift/skew checks** run with configured thresholds; synthetic shift experiments demonstrate detectable drift.
5. Final schema exported to `output/schema.pbtxt`.

## Notes
- If `tensorflow` import fails in Colab due to `jax`/`numpy` conflict, uninstall `jax` and `jaxlib`, then restart runtime.
- For TFDV 1.15.1 slice functions, use `tfdv.experimental_get_feature_value_slicer(...)`.

## How to Run
1. Open `Adult_Income_TFDV.ipynb` in Colab.
2. Run dependency installation cell.
3. Restart runtime/session when prompted.
4. Run all cells top-to-bottom.
5. Verify schema is saved at `output/schema.pbtxt`.

## Why HTML Export Is Included

GitHub may not render interactive TFDV visualizations from `.ipynb` outputs consistently.
To make results easier to review, I included an HTML export of the notebook.

### How to View Rendered Output
1. Download `Adult_Income_TFDV.html` from this repo.
2. Open it locally in a browser (Chrome/Safari/Firefox).
3. This shows the notebook content and captured outputs in a static, review-friendly format.

> Note: Some TFDV plots are interactive widgets and may still be best viewed by opening the notebook in Colab.
# GitHub Actions and GCP Connections: Beginner Lab

## Overview
This lab demonstrates how to automate a machine learning workflow using GitHub Actions and Google Cloud Platform (GCP). The workflow automatically trains a Logistic Regression model on the Wine dataset and uploads the trained model to Google Cloud Storage (GCS) on every push to the main branch.

## Changes from Original Template
- **Dataset:** Wine dataset (instead of Iris)
- **Model:** Logistic Regression (instead of RandomForestClassifier)
- **Trigger:** Workflow runs on push to main (instead of scheduled cron)
- **Model versioning:** Saved as `wine_lr_<timestamp>.joblib` in GCS

## Learning Objectives
1. Set up a GCP project and configure a service account for automation.
2. Grant GitHub Actions access to your GCP project.
3. Automate the process of training and saving a model using GitHub Actions.
4. Use Google Cloud Storage (GCS) to store your trained machine learning model.

## Project Structure
```
MLOps/
├── .github/
│   └── workflows/
│       └── run.yaml              ← GitHub Actions workflow (executed by GitHub)
├── GITHUB_ACTIONS/
│   ├── workflows/
│   │   └── run.yaml              ← reference copy for submission
│   ├── requirements.txt
│   ├── train_and_save_model.py
│   └── README.md
```

## Setup

### Step 1: Create a New GitHub Repository
Use an existing repo or create a new one. This lab lives inside the `GITHUB_ACTIONS/` folder of the MLOps repo.

### Step 2: Set Up Google Cloud Platform (GCP)
1. Create a new GCP project in Google Cloud Console
2. Go to **IAM & Admin → Service Accounts** and create a new service account
3. Assign the **Storage Admin** role to the service account
4. Generate a JSON key and download it securely

### Step 3: Create a Google Cloud Storage (GCS) Bucket
1. Go to **Cloud Storage → Buckets** in GCP Console
2. Create a new bucket (e.g. `mlops-github-actions-bucket`)
3. Note the bucket name — it's used in `train_and_save_model.py`

### Step 4: Add GCP Secret to GitHub
1. Go to your GitHub repo → **Settings → Secrets and variables → Actions**
2. Click **New repository secret**
3. Name it `GCP_SA_KEY` and paste the entire contents of your JSON key file
4. Click **Add secret**

### Step 5: Workflow Setup
The workflow file lives at `.github/workflows/run.yaml`. It triggers automatically on every push to main and does the following:
- Checks out the code
- Sets up Python 3.10
- Caches pip dependencies
- Installs requirements
- Authenticates with GCP using the service account key
- Runs `train_and_save_model.py` to train and upload the model

### Step 6: Verify
After pushing, go to the **Actions** tab in GitHub to watch the run. Once complete, check your GCS bucket — you should see a file like `trained_models/wine_lr_<timestamp>.joblib`.

## Results
- Workflow ran successfully in ~35 seconds
- Model uploaded to `gs://mlops-github-actions-bucket/trained_models/wine_lr_20260410-163205.joblib`
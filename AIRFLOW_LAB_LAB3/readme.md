# Airflow Lab 3: ML Pipeline for Ad Click Prediction

**Author:** Shloka Trivedi  
**Course:** IE7374 - Machine Learning Operations  
**Date:** February 2026

---

## Overview

This lab demonstrates how to deploy and orchestrate a Machine Learning pipeline using Apache Airflow on a Google Cloud Platform (GCP) Virtual Machine. The pipeline predicts whether a user will click on an advertisement based on their browsing behavior and demographics.

### Key Features:
- End-to-end ML pipeline orchestrated by Airflow
- Random Forest Classifier for ad click prediction
- Automated data preprocessing and feature scaling
- Model evaluation with comprehensive metrics
- Deployed on GCP VM

---

## Architecture

```
load_data_task --> data_preprocessing_task --> build_model_task --> evaluate_model_task --> send_email_task
```

### Pipeline Tasks:
1. **load_data_task** - Loads advertising.csv dataset
2. **data_preprocessing_task** - Cleans, scales, and splits data
3. **build_model_task** - Trains Random Forest model and saves it
4. **evaluate_model_task** - Evaluates model and saves metrics
5. **send_email_task** - Sends success email notification via Gmail SMTP

---

## Project Structure

```
airflow/
├── dags/
│   ├── my_dag.py                    # DAG definition
│   ├── data/
│   │   └── advertising.csv          # Dataset
│   ├── src/
│   │   ├── __init__.py
│   │   ├── model_development.py     # ML functions
│   │   └── success_email.py         # Email notification
│   └── model/
│       ├── random_forest_model.sav  # Trained model
│       └── model_metrics.txt        # Evaluation results
└── requirements.txt
```

---

## Setup Instructions

### Prerequisites
- Google Cloud Platform account
- Basic knowledge of Linux commands

### Step 1: Create GCP Virtual Machine

1. Go to GCP Console, then Compute Engine, then VM Instances
2. Click Create Instance
3. Configure:
   - Name: `airflow-lab3`
   - Machine type: `e2-medium` (2 vCPU, 4GB RAM)
   - Boot disk: Ubuntu 22.04 LTS, 20GB
   - Firewall: Allow HTTP and HTTPS traffic
4. Click Create

### Step 2: Configure Firewall for Port 8080

1. Go to VPC Network, then Firewall
2. Click Create Firewall Rule
3. Configure:
   - Name: `allow-airflow-8080`
   - Targets: All instances in the network
   - Source IP ranges: `0.0.0.0/0`
   - Protocols/ports: TCP: `8080`
4. Click Create

### Step 3: SSH into VM and Install Dependencies

```bash
# Update system
sudo apt update

# Install Python packages
sudo apt install python3-pip python3-venv python3-full -y

# Create virtual environment
python3 -m venv airflow_venv
source airflow_venv/bin/activate

# Install Apache Airflow
pip install apache-airflow

# Install additional dependencies
pip install pandas scikit-learn pyarrow
```

### Step 4: Initialize Airflow

```bash
# Reset/Initialize database
airflow db reset -y

# Start Airflow (all-in-one)
airflow standalone
```

### Step 5: Configure Email (SMTP)

1. Create a Gmail App Password at https://myaccount.google.com/apppasswords
2. Add the SMTP connection in Airflow:

```bash
airflow connections add 'smtp_default' --conn-type 'smtp' --conn-host 'smtp.gmail.com' --conn-login 'your_email@gmail.com' --conn-password 'your_app_password' --conn-port 587
```

### Step 6: Upload DAG Files

Upload the following files to `~/airflow/dags/`:
- `my_dag.py`
- `src/model_development.py`
- `data/advertising.csv`

### Step 7: Access Airflow UI

1. Open browser: `http://<VM_EXTERNAL_IP>:8080`
2. Login credentials are shown in the terminal output
3. Find and enable the DAG: `ML_Pipeline`
4. Click the play button to trigger the DAG
5. Check your email for success notification

---

## Dataset

Advertising Dataset - Contains user behavior data for predicting ad clicks.

| Feature | Description |
|---------|-------------|
| Daily Time Spent on Site | Time spent on website (minutes) |
| Age | User's age |
| Area Income | Average income of user's area |
| Daily Internet Usage | Daily internet usage (minutes) |
| Male | Gender (1=Male, 0=Female) |
| Clicked on Ad | Target variable (1=Clicked, 0=Not Clicked) |

---

## Model Details

**Algorithm:** Random Forest Classifier

**Hyperparameters:**
- n_estimators: 100
- max_depth: 10
- min_samples_split: 5
- min_samples_leaf: 2
- random_state: 42

**Data Split:** 70% Training, 30% Testing

---

## Results

After running the pipeline, the model achieves:

| Metric | Score |
|--------|-------|
| Accuracy | 95.67% |
| Precision | 96.69% |
| Recall | 94.81% |
| F1-Score | 95.74% |

Detailed metrics are saved in `model/model_metrics.txt`

---

## Changes from Original Lab

1. Changed DAG name to `ML_Pipeline`
2. Changed owner to `Shloka Trivedi`
3. Replaced Logistic Regression with Random Forest Classifier
4. Added model evaluation task with comprehensive metrics (accuracy, precision, recall, F1-score)
5. Added logging throughout the pipeline
6. Metrics saved to file for documentation
7. Added daily schedule (`@daily`)
8. Configured Gmail SMTP for email notifications

---

## Cleanup

To avoid charges, stop or delete the VM when done:

```bash
# From GCP Console
# VM Instances -> Select VM -> Stop (or Delete)
```

---
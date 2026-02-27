# DATA_DVC_LAB1 — Data Versioning with DVC & Google Cloud Storage

This lab demonstrates data versioning using [DVC](https://dvc.org/) with Google Cloud Storage (GCS) as the remote backend.

---

## Prerequisites
- Python 3.x with `pip`
- A Google Cloud account with billing enabled
- A [Kaggle](https://www.kaggle.com) account to download the dataset

---

## 1. Setup & Installation

Clone the repository and navigate to the lab folder:

```bash
git clone https://github.com/Trivedi-s/MLops.git
cd MLops/DATA_DVC_LAB1
```

Install DVC with Google Cloud support:

```bash
pip install 'dvc[gs]'
```

Initialize DVC:

```bash
dvc init -f
```

---

## 2. GCP Setup

### 2.1 Create a GCP Project
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project (e.g. `dvc-lab1`)

### 2.2 Create a GCS Bucket
1. Navigate to **Cloud Storage → Buckets → Create**
2. Set a unique bucket name (e.g. `your-name-dvc-lab1`)
3. Set region to `us-east1`
4. Leave all other settings as default and click **Create**

### 2.3 Create a Service Account & Download Key
1. Go to **IAM & Admin → Service Accounts → Create Service Account**
2. Name it `lab2`, set role to **Owner**, click **Done**
3. Click on the service account → **Keys** tab → **Add Key → Create new key → JSON**
4. Save the downloaded JSON file in the `DATA_DVC_LAB1/` folder as `dvc-lab1-key.json`

> ⚠️ The `dvc-lab1-key.json` file is listed in `.gitignore` and will **not** be pushed to GitHub.

---

## 3. DVC Configuration

Add your GCS bucket as the DVC remote:

```bash
dvc remote add -d myremote gs://<your-bucket-name>
```

Set the credentials path:

```bash
dvc remote modify myremote credentialpath lab2-key.json
```

Verify the config:

```bash
cat .dvc/config
```

Expected output:
```
[core]
    remote = myremote
['remote "myremote"']
    url = gs://<your-bucket-name>
    credentialpath = lab2-key.json
```

---

## 4. Data Versioning & Reverting

### 4.1 Download the Dataset
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata), place it in the `data/` folder and rename it to `CC_GENERAL.csv`.

### 4.2 Track & Push V1
```bash
dvc add data/CC_GENERAL.csv
git add data/CC_GENERAL.csv.dvc data/.gitignore
git commit -m "Track CC_GENERAL.csv with DVC - V1"
git push origin main
dvc push
```

### 4.3 Simulate a Data Change & Push V2
```bash
echo "test,test,test" >> data/CC_GENERAL.csv
dvc add data/CC_GENERAL.csv
git add data/CC_GENERAL.csv.dvc
git commit -m "Updated dataset v2"
git push origin main
dvc push
```

### 4.4 Revert to V1
Get the commit hash for V1:
```bash
git log --oneline
```

Checkout the V1 `.dvc` file and restore the data:
```bash
git checkout <v1-commit-hash> -- data/CC_GENERAL.csv.dvc
dvc checkout
```

Verify the revert (the "test,test,test" line should be gone):
```bash
tail -5 data/CC_GENERAL.csv
```

Commit the reverted state:
```bash
git add data/CC_GENERAL.csv.dvc
git commit -m "Reverted to V1 dataset"
git push origin main
```

---

## Project Structure
```
DATA_DVC_LAB1/
├── .dvc/               # DVC metadata and config
├── data/
│   ├── .gitignore      # Ensures raw data is not pushed to Git
│   └── CC_GENERAL.csv.dvc  # DVC tracking file for the dataset
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Notes
- The actual dataset (`CC_GENERAL.csv`) is **not** stored in Git — it is managed by DVC and stored in GCS.
- Each version of the dataset is stored as a separate object in GCS under `files/md5/`.
- To pull the latest data from GCS on a fresh clone: `dvc pull`
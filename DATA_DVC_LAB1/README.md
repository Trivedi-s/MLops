# DATA_DVC_LAB1 — Data Versioning with DVC & Google Cloud Storage

This lab demonstrates data versioning using [DVC](https://dvc.org/) with Google Cloud Storage (GCS) as the remote backend. Two datasets are tracked: a credit card dataset and the Titanic passenger dataset.

---

## Prerequisites
- Python 3.x with `pip`
- A Google Cloud account with billing enabled
- A [Kaggle](https://www.kaggle.com) account to download the datasets

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
4. Save the downloaded JSON file in the `DATA_DVC_LAB1/` folder as `lab2-key.json`

> ⚠️ The `lab2-key.json` file is listed in `.gitignore` and will **not** be pushed to GitHub.

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

## 4. Dataset 1 — Credit Card Data (CC_GENERAL.csv)

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata), place it in the `data/` folder and rename it to `CC_GENERAL.csv`.

### Track & Push V1
```bash
dvc add data/CC_GENERAL.csv
git add data/CC_GENERAL.csv.dvc data/.gitignore
git commit -m "Track CC_GENERAL.csv with DVC - V1"
git push origin main
dvc push
```

### Simulate a Data Change & Push V2
```bash
echo "test,test,test" >> data/CC_GENERAL.csv
dvc add data/CC_GENERAL.csv
git add data/CC_GENERAL.csv.dvc
git commit -m "Updated CC_GENERAL dataset v2"
git push origin main
dvc push
```

### Revert to V1
```bash
git log --oneline  # find the V1 commit hash
git checkout a6cfdab -- data/CC_GENERAL.csv.dvc
dvc checkout
git add data/CC_GENERAL.csv.dvc
git commit -m "Reverted CC_GENERAL to V1"
git push origin main
```

---

## 5. Dataset 2 — Titanic Passenger Data (TITANIC.csv)

Download the Titanic dataset from [Kaggle](https://www.kaggle.com/datasets/brendan45774/test-file), place it in the `data/` folder and rename it to `TITANIC.csv`.

### Track & Push V1
```bash
dvc add data/TITANIC.csv
git add data/.gitignore data/TITANIC.csv.dvc
git commit -m "Add Titanic dataset tracked with DVC - V1"
git push origin main
dvc push
```

### Add FamilySize Feature & Push V2
A new feature `FamilySize` is added as `SibSp + Parch + 1`:

```bash
/opt/anaconda3/bin/python3 -c "
import pandas as pd
df = pd.read_csv('data/TITANIC.csv')
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df.to_csv('data/TITANIC.csv', index=False)
print(df.head())
"
```

```bash
dvc add data/TITANIC.csv
git add data/TITANIC.csv.dvc
git commit -m "Added FamilySize feature to Titanic dataset v2"
git push origin main
dvc push
```

### Revert to V1
```bash
git log --oneline  # find the V1 Titanic commit hash
git checkout cd53095 -- data/TITANIC.csv.dvc
dvc checkout
git add data/TITANIC.csv.dvc
git commit -m "Reverted Titanic dataset to V1"
git push origin main
```

Verify revert (FamilySize column should be gone):
```bash
/opt/anaconda3/bin/python3 -c "
import pandas as pd
df = pd.read_csv('data/TITANIC.csv')
print(df.columns.tolist())
"
```

---

## Project Structure
```
DATA_DVC_LAB1/
├── .dvc/                       # DVC metadata and config
├── data/
│   ├── .gitignore              # Ensures raw data is not pushed to Git
│   ├── CC_GENERAL.csv.dvc      # DVC tracking file for credit card dataset
│   └── TITANIC.csv.dvc         # DVC tracking file for Titanic dataset
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Notes
- Raw datasets (`CC_GENERAL.csv`, `TITANIC.csv`) are **not** stored in Git — managed by DVC and stored in GCS.
- Each version of each dataset is stored as a separate object in GCS under `files/md5/`.
- To pull the latest data from GCS on a fresh clone: `dvc pull`
# Cloud Run Intermediate Lab

This version includes custom modifications: a different BigQuery dataset state, a new `/status` health check endpoint, a new `/count` endpoint, and custom GCS upload content.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Modifications Made](#modifications-made)
- [Step 1: Set Up Google Cloud Project and Resources](#step-1-set-up-google-cloud-project-and-resources)
- [Step 2: Create and Containerize the Application](#step-2-create-and-containerize-the-application)
- [Step 3: Push the Docker Image to Container Registry](#step-3-push-the-docker-image-to-container-registry)
- [Step 4: Deploy to Google Cloud Run](#step-4-deploy-to-google-cloud-run)
- [Step 5: Access and Test the Application](#step-5-access-and-test-the-application)
- [Step 6: Monitor and Log the Service](#step-6-monitor-and-log-the-service)
- [Step 7: Clean Up Resources](#step-7-clean-up-resources)

---

## Prerequisites

- **Google Cloud Account** with billing enabled
- **Google Cloud SDK** installed ([install guide](https://cloud.google.com/sdk/docs/install))
- **Docker** installed and running ([Docker Desktop](https://www.docker.com/products/docker-desktop))
- **Basic Knowledge** of Python, Flask, and Docker

---

## Modifications Made

The following changes were made from the original lab:

| # | Modification | Detail |
|---|---|---|
| 1 | BigQuery state | Changed from Texas (`TX`) to California (`CA`) |
| 2 | GCS upload | Changed filename to `my_custom_file.txt` with custom content |
| 3 | New `/status` endpoint | Returns a JSON health check response |
| 4 | New `/count` endpoint | Returns total row count for California in the BigQuery dataset |

---

## Step 1: Set Up Google Cloud Project and Resources

### 1. Create a Google Cloud Project

- Go to the [Google Cloud Console](https://console.cloud.google.com/)
- Create a new project named `cloud-run-intermediate-lab`
- Note the **Project ID**: `cloud-run-intermediate-lab`

### 2. Enable Necessary APIs

Navigate to **APIs & Services > Library** and enable:
- Cloud Run Admin API
- Cloud Storage API
- BigQuery API
- Container Registry API

Or via CLI:
```bash
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### 3. Create a Cloud Storage Bucket

- Go to **Cloud Storage > Buckets > Create**
- Name: `intermediate_cloud_run_bucket`
- Location type: **Region**, `us-central1`
- Storage class: **Standard**
- Click **Create**

### 4. Set Up a Service Account

- Go to **IAM & Admin > Service Accounts > Create Service Account**
- Name: `cloud-run-sa`
- Assign the following roles:
  - **Storage Admin**
  - **BigQuery User**
- Click **Done**

Grant yourself permission to use the service account:
```bash
gcloud iam service-accounts add-iam-policy-binding \
  cloud-run-sa@cloud-run-intermediate-lab.iam.gserviceaccount.com \
  --member=user:YOUR_EMAIL \
  --role=roles/iam.serviceAccountUser
```

---

## Step 2: Create and Containerize the Application

### app.py

```python
from flask import Flask
from google.cloud import storage, bigquery
import os

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Hello from MY intermediate lab!"

# MODIFICATION 1: Changed upload filename and content
@app.route('/upload')
def upload_file():
    storage_client = storage.Client()
    bucket_name = os.environ.get('BUCKET_NAME')
    if not bucket_name:
        return 'BUCKET_NAME environment variable is not set.', 500
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob('my_custom_file.txt')
    blob.upload_from_string('Hello from my custom Cloud Run lab submission!')
    return f'Custom file uploaded to {bucket_name}.'

# MODIFICATION 2: Changed BigQuery state from TX to CA
@app.route('/query')
def query_bigquery():
    client = bigquery.Client()
    query = """
        SELECT name, SUM(number) as total
        FROM `bigquery-public-data.usa_names.usa_1910_current`
        WHERE state = 'CA'
        GROUP BY name
        ORDER BY total DESC
        LIMIT 10
    """
    query_job = client.query(query)
    results = query_job.result()
    names = [row.name for row in results]
    return f'Top names in California: {", ".join(names)}'

# MODIFICATION 3: New /status health check endpoint
@app.route('/status')
def status():
    return {"status": "healthy", "service": "cloud-run-intermediate-app"}, 200

# MODIFICATION 4: New /count endpoint
@app.route('/count')
def count_rows():
    client = bigquery.Client()
    query = """
        SELECT COUNT(*) as total_rows
        FROM `bigquery-public-data.usa_names.usa_1910_current`
        WHERE state = 'CA'
    """
    query_job = client.query(query)
    results = query_job.result()
    for row in results:
        return f'Total rows for California: {row.total_rows}'

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
```

### requirements.txt

```
flask
gunicorn
google-cloud-storage
google-cloud-bigquery
```

### Dockerfile

```dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
```

---

## Step 3: Push the Docker Image to Container Registry

### 1. Authenticate Docker with Google Cloud

```bash
gcloud auth configure-docker
```

### 2. Build the Image (with platform flag for Apple Silicon Macs)

```bash
docker build --platform linux/amd64 -t gcr.io/cloud-run-intermediate-lab/cloud-run-intermediate-app .
```

### 3. Push the Image

```bash
docker push gcr.io/cloud-run-intermediate-lab/cloud-run-intermediate-app
```

---

## Step 4: Deploy to Google Cloud Run

```bash
gcloud run deploy cloud-run-intermediate-service \
  --image gcr.io/cloud-run-intermediate-lab/cloud-run-intermediate-app \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --update-env-vars BUCKET_NAME=intermediate_cloud_run_bucket \
  --service-account cloud-run-sa@cloud-run-intermediate-lab.iam.gserviceaccount.com
```

---

## Step 5: Access and Test the Application

### Get the Service URL

```bash
gcloud run services describe cloud-run-intermediate-service \
  --region us-central1 \
  --format "value(status.url)"
```

### Test All Endpoints

```bash
# Root
curl https://YOUR_SERVICE_URL/

# Upload to GCS
curl https://YOUR_SERVICE_URL/upload

# Query BigQuery (Top names in California)
curl https://YOUR_SERVICE_URL/query

# Health check
curl https://YOUR_SERVICE_URL/status

# Row count for California
curl https://YOUR_SERVICE_URL/count
```

## Step 6: Monitor and Log the Service

### View Logs in Cloud Console

- Navigate to **Cloud Run** in the GCP Console
- Click on **cloud-run-intermediate-service**
- Go to the **Logs** tab

### View Logs via CLI

```bash
gcloud logging read "resource.type=cloud_run_revision" --limit 20
```

### Monitor Metrics

- In the Cloud Run service details, go to the **Metrics** tab
- Observe CPU usage, memory usage, and request latency

---

## Step 7: Clean Up Resources

```bash
# Delete Cloud Run service
gcloud run services delete cloud-run-intermediate-service --region us-central1

# Delete container image
gcloud container images delete gcr.io/cloud-run-intermediate-lab/cloud-run-intermediate-app --force-delete-tags

# Delete GCS bucket
gsutil rm -r gs://intermediate_cloud_run_bucket
```

---

## Conclusion

In this lab:

- Set up a GCP project with Cloud Run, Cloud Storage, and BigQuery
- Built a Flask app with 5 endpoints including custom additions
- Containerized and deployed the app to Cloud Run with a service account
- Tested all endpoints and verified GCS uploads and BigQuery queries
- Monitored logs and metrics via the Cloud Console
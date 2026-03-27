from flask import Flask
from google.cloud import storage, bigquery
import os

app = Flask(__name__)

# --- Root endpoint ---
@app.route('/')
def hello_world():
    return "Hello from MY intermediate lab!"

# --- MODIFICATION 1: Changed upload file name and content ---
@app.route('/upload')
def upload_file():
    storage_client = storage.Client()
    bucket_name = os.environ.get('BUCKET_NAME')
    if not bucket_name:
        return 'BUCKET_NAME environment variable is not set.', 500
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob('my_custom_file.txt')  # changed filename
    blob.upload_from_string('Hello from my custom Cloud Run lab submission!')  # changed content
    return f'Custom file uploaded to {bucket_name}.'

# --- MODIFICATION 2: Changed BigQuery state from TX to CA ---
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

# --- MODIFICATION 3: New /status health check endpoint ---
@app.route('/status')
def status():
    return {"status": "healthy", "service": "cloud-run-intermediate-app"}, 200

# --- MODIFICATION 4: New /count endpoint ---
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

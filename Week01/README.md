# MLOps Week 1 — ETL to AWS S3

This project extracts a sample taxi dataset, cleans it, and uploads a Parquet version to AWS S3.

## Steps
1. Download CSV
2. Convert timestamps, drop missing data
3. Save as Parquet
4. Upload to S3 via boto3

## How to Run
```bash
pip install -r requirements.txt
python etl.py
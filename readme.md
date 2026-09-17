# Insurance Fraud Detection - ETL Pipeline

This repository contains the data ingestion and transformation (ETL) pipeline for the Insurance Claims Fraud Detection project. The pipeline extracts raw claims data from an Azure Data Lake Storage (ADLS) Gen2 Bronze container, imputes missing values, synthesizes unstructured adjuster notes for NLP processing, and loads the optimized Parquet file into the Silver container.

## Prerequisites

1. **Azure Storage Account**: An ADLS Gen2 storage account with `bronze` and `silver` containers created.
2. **Raw Data**: The Kaggle Car Insurance Fraud dataset (`car_insurance_claim.csv`) must be uploaded to the root of your `bronze` container.
3. **Environment Variables**: Create a `.env` file in the root directory of this project and add your Azure connection string:
   ```env
   AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=YOUR_ACCOUNT_NAME;AccountKey=YOUR_ACCOUNT_KEY;EndpointSuffix=core.windows.net"
   ```

---

## Execution Method 1: Docker (Recommended)

Running via Docker ensures a clean, isolated environment using the CPU-optimized version of PyTorch to keep the image lightweight.

**1. Build the Docker Image**
```bash
docker build -t fraud-pipeline .
```

**2. Execute the ETL Script**
This command mounts your `.env` variables, runs the Python script, and automatically removes the container (`--rm`) once the process finishes.
```bash
docker run --rm --env-file .env fraud-pipeline python main.py
```

---

## Execution Method 2: Local Python Environment

If you prefer to run the script directly on your host machine without Docker, use a virtual environment.

**1. Create and Activate a Virtual Environment**
* **Windows (PowerShell):**
  ```powershell
  py -3.11 -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **Linux/macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
# Overwrite standard PyTorch with the smaller CPU-only wheel
pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
```

**3. Execute the Script**
```bash
python main.py
```

---

## Expected Output

Upon successful execution, the terminal will log the processing steps. You can verify the success by navigating to your Azure Portal:
1. Open your ADLS Gen2 Storage Account.
2. Navigate to **Containers** > **silver**.
3. Verify that `silver_claims_fused.parquet` has been successfully generated and uploaded. This file is now ready to be imported into Google Colab for DistilBERT and XGBoost model training.
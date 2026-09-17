import os
import io
import pandas as pd
import numpy as np
import random
from dotenv import load_dotenv
from azure.storage.filedatalake import DataLakeServiceClient

def main():
    # 1. Setup and Authentication
    load_dotenv()
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING is missing in .env")
        
    service_client = DataLakeServiceClient.from_connection_string(connection_string)
    print("Authenticating with Azure Data Lake Gen2...")

    # 2. Extract: Read Raw CSV from Bronze Layer
    print("Downloading car_insurance_claim.csv from /bronze...")
    bronze_client = service_client.get_file_system_client(file_system="bronze")
    raw_file_client = bronze_client.get_file_client("car_insurance_claim.csv")
    
    downloaded_bytes = raw_file_client.download_file().readall()
    df = pd.read_csv(io.BytesIO(downloaded_bytes))

    # 3. Transform: Data Cleaning & Imputation
    print("Executing data transformations and imputations...")
    
    # The Kaggle automobile insurance dataset uses '?' instead of standard nulls
    df.replace('?', np.nan, inplace=True)

    # Drop purely empty or irrelevant structural columns (Kaggle artifacts)
    if '_c39' in df.columns:
        df.drop(columns=['_c39'], inplace=True)

    # Impute missing values for categorical columns using the mode (most frequent)
    categorical_cols = ['collision_type', 'property_damage', 'police_report_available']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Ensure datetime columns are formatted correctly for Parquet compatibility
    date_cols = ['incident_date', 'policy_bind_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Synthesize Unstructured Text for FinBERT
    def generate_notes(fraud_status):
        if fraud_status == 'Y':
            return random.choice([
                "Claimant's description is inconsistent with the physical vehicle damage.",
                "Hesitant to provide an official police report. Details are vague.",
                "Damage appears to be pre-existing wear, not from a recent collision.",
                "Submitted documents appear altered; repair shop cannot be verified.",
                "Witness statements conflict with the insured's timeline of events."
            ])
        else:
            return random.choice([
                "Standard claim. Damage is consistent with a low-speed rear collision.",
                "All documentation provided promptly. Police report matches the narrative.",
                "Insured was cooperative. Proceeding with standard estimate payout.",
                "Minor fender bender. No red flags identified during initial triage."
            ])

    # Apply the synthetic text generator
    if 'fraud_reported' in df.columns:
        df['adjuster_notes'] = df['fraud_reported'].apply(generate_notes)

    # 4. Load: Convert to Parquet and Push to Silver Layer
    print("Converting to Parquet and streaming to /silver...")
    parquet_buffer = io.BytesIO()
    df.to_parquet(parquet_buffer, index=False, engine='pyarrow')
    parquet_buffer.seek(0)

    silver_client = service_client.get_file_system_client(file_system="silver")
    silver_file_client = silver_client.get_file_client("silver_claims_fused.parquet")
    
    silver_file_client.upload_data(parquet_buffer.getvalue(), overwrite=True)
    print("Successfully uploaded silver_claims_fused.parquet to Azure /silver container.")

if __name__ == "__main__":
    main()
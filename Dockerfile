FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# CPU-only PyTorch
# Required by FinBERT / Transformers
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Copy application
COPY app.py .
COPY inference.py .

# Copy datasets used by the dashboard
COPY dataset3_gold_with_notes_wild.csv .
COPY dataset1_gold_with_notes_wild.csv .

# Copy trained model artifacts
COPY model_outputs ./model_outputs
COPY triumvirate_outputs ./triumvirate_outputs

# Streamlit port
EXPOSE 8501

# Start Streamlit
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
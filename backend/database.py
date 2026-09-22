import os
import duckdb

ARTIFACTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
CSV_PATH = os.path.join(ARTIFACTS_DIR, "dataset1_gold_with_notes_wild.csv")

conn = duckdb.connect(database=":memory:", read_only=False)

def init_db():
    """Initializes tables and dynamically ingests the gold CSV into DuckDB."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS triage_logs (
            claim_id VARCHAR,
            claim_amount DOUBLE,
            supervised_risk VARCHAR,
            risk_score DOUBLE,
            anomaly_score DOUBLE,
            status VARCHAR,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    if os.path.exists(CSV_PATH):
        try:
            temp_df = conn.execute(f"SELECT * FROM read_csv_auto('{CSV_PATH}') LIMIT 1").fetchdf()
            cols = [c.lower() for c in temp_df.columns]

            # Resolve claim id column
            id_col = next((c for c in temp_df.columns if c.lower() in ['claim_id', 'policy_number', 'id']), None)
            id_expr = f"CAST({id_col} AS VARCHAR)" if id_col else "'CLM-' || CAST(row_number() OVER () AS VARCHAR)"

            # Resolve amount column
            amt_col = next((c for c in temp_df.columns if 'amount' in c.lower() or 'claim' in c.lower() and c.lower() != id_col), None)
            amt_expr = f"CAST(COALESCE(try_cast({amt_col} AS DOUBLE), 500.0) AS DOUBLE)" if amt_col else "500.0"

            # Resolve risk/fraud column
            fraud_col = next((c for c in temp_df.columns if 'fraud' in c.lower() or 'risk' in c.lower() or 'label' in c.lower()), None)
            if fraud_col:
                risk_expr = f"""
                    CASE 
                        WHEN try_cast({fraud_col} AS VARCHAR) IN ('1', 'Y', 'True', 'true', 'High') THEN 'High'
                        ELSE 'Low'
                    END
                """
                score_expr = f"""
                    CASE 
                        WHEN try_cast({fraud_col} AS VARCHAR) IN ('1', 'Y', 'True', 'true', 'High') THEN 0.88
                        ELSE 0.14
                    END
                """
                status_expr = f"""
                    CASE 
                        WHEN try_cast({fraud_col} AS VARCHAR) IN ('1', 'Y', 'True', 'true', 'High') THEN 'Flagged'
                        ELSE 'Approved'
                    END
                """
            else:
                risk_expr = "'Low'"
                score_expr = "0.15"
                status_expr = "'Approved'"

            ingest_query = f"""
                INSERT INTO triage_logs (claim_id, claim_amount, supervised_risk, risk_score, anomaly_score, status, timestamp)
                SELECT 
                    {id_expr} AS claim_id,
                    {amt_expr} AS claim_amount,
                    {risk_expr} AS supervised_risk,
                    {score_expr} AS risk_score,
                    0.65 AS anomaly_score,
                    {status_expr} AS status,
                    CURRENT_TIMESTAMP AS timestamp
                FROM read_csv_auto('{CSV_PATH}')
                LIMIT 50;
            """
            conn.execute(ingest_query)
        except Exception as e:
            print(f"[Warning] Failed to ingest gold CSV: {e}. Populating fallback seed data.")
            _insert_fallback_data()
    else:
        _insert_fallback_data()

def _insert_fallback_data():
    conn.execute("""
        INSERT INTO triage_logs VALUES
        ('CLM-1100152', 500.00, 'High', 0.88, 3.30, 'Flagged', CURRENT_TIMESTAMP),
        ('CLM-1180158', 550.00, 'High', 0.82, 2.80, 'Flagged', CURRENT_TIMESTAMP),
        ('CLM-1020297', 300.00, 'Medium', 0.54, 1.50, 'Review', CURRENT_TIMESTAMP),
        ('CLM-1100991', 500.00, 'Low', 0.12, 0.50, 'Approved', CURRENT_TIMESTAMP);
    """)

def get_recent_triage_queue(limit: int = 10):
    """Retrieves top triage items ordered by risk score descending."""
    query = f"""
        SELECT claim_id, claim_amount, supervised_risk, risk_score, anomaly_score, status
        FROM triage_logs
        ORDER BY risk_score DESC
        LIMIT {limit};
    """
    rows = conn.execute(query).fetchall()
    return [
        {
            "claim_id": r[0],
            "claim_amount": r[1],
            "supervised_risk": r[2],
            "risk_score": r[3],
            "anomaly_score": r[4],
            "status": r[5]
        }
        for r in rows
    ]

def log_prediction(claim_id: str, claim_amount: float, risk_tier: str, risk_score: float, anomaly_score: float):
    """Inserts a newly evaluated claim into the live DuckDB queue."""
    status = "Flagged" if risk_score >= 0.70 else ("Review" if risk_score >= 0.40 else "Approved")
    conn.execute("""
        INSERT INTO triage_logs (claim_id, claim_amount, supervised_risk, risk_score, anomaly_score, status, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (claim_id, claim_amount, risk_tier, risk_score, anomaly_score, status))
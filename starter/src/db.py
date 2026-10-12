"""Shared TimescaleDB helpers: engine + hypertable schema.

Used by seed_db.py (bulk replay) and ingest_mqtt.py (live MQTT ingest).
"""
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS machine_telemetry (
        timestamp TIMESTAMPTZ NOT NULL,
        product_id TEXT,
        machine_type TEXT,
        air_temp_k DOUBLE PRECISION,
        process_temp_k DOUBLE PRECISION,
        rpm DOUBLE PRECISION,
        torque_nm DOUBLE PRECISION,
        tool_wear_min DOUBLE PRECISION,
        machine_failure INTEGER
    )
"""

# Column order shared by every writer of machine_telemetry.
COLUMNS = ["timestamp", "product_id", "machine_type", "air_temp_k",
           "process_temp_k", "rpm", "torque_nm", "tool_wear_min",
           "machine_failure"]


def get_engine():
    return create_engine(
        f"postgresql+psycopg2://{os.getenv('POSTGRES_USER', 'twin')}"
        f":{os.getenv('POSTGRES_PASSWORD', 'change_me')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}"
        f":{os.getenv('POSTGRES_PORT', '5432')}"
        f"/{os.getenv('POSTGRES_DB', 'twin')}"
    )


def ensure_schema(engine):
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
        conn.execute(text(SCHEMA_SQL))
        conn.execute(text(
            "SELECT create_hypertable('machine_telemetry', 'timestamp', "
            "if_not_exists => TRUE)"))

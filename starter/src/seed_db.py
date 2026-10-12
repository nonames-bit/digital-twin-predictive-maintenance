"""Replay AI4I 2020 rows into TimescaleDB as a synthetic minute-sampled feed.

Creates the hypertable and inserts the 10,000 rows in timestamp order.
This stands in for the live MQTT feed until that's built (see simulator.py).

Usage: docker compose up -d db   then   python src/seed_db.py
"""
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import COLUMNS, ensure_schema, get_engine

DATA = Path(__file__).resolve().parent.parent.parent / "ai4i2020.csv"


def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df["timestamp"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(df.index, unit="min")
    feed = df[["timestamp", "Product ID", "Type", "Air temperature",
               "Process temperature", "Rotational speed", "Torque",
               "Tool wear", "Machine failure"]].rename(columns={
        "Product ID": "product_id", "Type": "machine_type",
        "Air temperature": "air_temp_k", "Process temperature": "process_temp_k",
        "Rotational speed": "rpm", "Torque": "torque_nm",
        "Tool wear": "tool_wear_min", "Machine failure": "machine_failure"})
    feed = feed[COLUMNS]

    engine = get_engine()
    ensure_schema(engine)
    with engine.begin() as conn:
        feed.to_sql("machine_telemetry", conn, if_exists="append",
                    index=False, method="multi", chunksize=1000)
        n = conn.execute(text("SELECT count(*) FROM machine_telemetry")).scalar()
    print(f"machine_telemetry rows: {n}")


if __name__ == "__main__":
    main()

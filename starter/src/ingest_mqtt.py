"""MQTT → TimescaleDB ingestor: subscribes to the live feed, writes the hypertable.

Batch-inserts messages from ``twin/telemetry`` into ``machine_telemetry``.
This is the local OT path; the cloud path is aws_bridge.py.

Usage:
    docker compose up -d mosquitto db
    python src/ingest_mqtt.py

Env: MQTT_BROKER_HOST (default localhost), MQTT_BROKER_PORT (default 1883),
     INGEST_BATCH (default 50), INGEST_FLUSH_SEC (default 5).
"""
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from paho.mqtt import client as mqtt

from db import COLUMNS, ensure_schema, get_engine

load_dotenv()

TOPIC = "twin/telemetry"
BATCH = int(os.getenv("INGEST_BATCH", "50"))
FLUSH_SEC = float(os.getenv("INGEST_FLUSH_SEC", "5"))

engine = get_engine()
ensure_schema(engine)
buffer: list[dict] = []
last_flush = time.time()


def flush():
    global last_flush
    if not buffer:
        return
    df = pd.DataFrame(buffer, columns=COLUMNS)
    with engine.begin() as conn:
        df.to_sql("machine_telemetry", conn, if_exists="append",
                  index=False, method="multi", chunksize=1000)
    print(f"ingested {len(buffer)} rows")
    buffer.clear()
    last_flush = time.time()


def on_message(_cli, _ud, msg):
    try:
        row = json.loads(msg.payload.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    buffer.append({c: row.get(c) for c in COLUMNS})
    if len(buffer) >= BATCH or time.time() - last_flush >= FLUSH_SEC:
        flush()


def main():
    host = os.getenv("MQTT_BROKER_HOST", "localhost")
    port = int(os.getenv("MQTT_BROKER_PORT", "1883"))
    cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="twin-ingestor")
    cli.on_message = on_message
    cli.connect(host, port, keepalive=60)
    cli.subscribe(TOPIC, qos=1)
    print(f"subscribed to {TOPIC} @ {host}:{port} → TimescaleDB")
    try:
        cli.loop_forever()
    except KeyboardInterrupt:
        flush()
        print("stopped")


if __name__ == "__main__":
    main()

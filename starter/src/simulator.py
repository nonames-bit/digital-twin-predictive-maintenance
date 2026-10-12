"""Synthetic OT feed simulator: replays AI4I 2020 rows as live MQTT messages.

Publishes one JSON telemetry message per interval to ``twin/telemetry`` on the
local Mosquitto broker. Stands in for the real PLC / Ignition feed.

Usage:
    docker compose up -d mosquitto
    python src/simulator.py [--once] [--rate 0.5]

Env: MQTT_BROKER_HOST (default localhost), MQTT_BROKER_PORT (default 1883).
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from paho.mqtt import client as mqtt

load_dotenv()

DATA = Path(__file__).resolve().parent.parent.parent / "ai4i2020.csv"
TOPIC = "twin/telemetry"

RENAME = {"Product ID": "product_id", "Type": "machine_type",
          "Air temperature": "air_temp_k", "Process temperature": "process_temp_k",
          "Rotational speed": "rpm", "Torque": "torque_nm",
          "Tool wear": "tool_wear_min", "Machine failure": "machine_failure"}


def load_rows():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df = df.rename(columns=RENAME)
    return df[list(RENAME.values())].to_dict("records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="publish one pass, then exit")
    ap.add_argument("--rate", type=float, default=1.0, help="seconds between messages")
    args = ap.parse_args()

    host = os.getenv("MQTT_BROKER_HOST", "localhost")
    port = int(os.getenv("MQTT_BROKER_PORT", "1883"))
    cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="twin-simulator")
    cli.connect(host, port, keepalive=60)
    cli.loop_start()

    rows = load_rows()
    print(f"publishing {len(rows)} rows to {TOPIC} @ {host}:{port} "
          f"({'once' if args.once else 'looping'})")
    try:
        while True:
            for row in rows:
                msg = {"timestamp": datetime.now(timezone.utc).isoformat(), **row}
                cli.publish(TOPIC, json.dumps(msg), qos=1)
                time.sleep(args.rate)
            if args.once:
                return
    except KeyboardInterrupt:
        print("stopped")
    finally:
        cli.loop_stop()
        cli.disconnect()


if __name__ == "__main__":
    main()

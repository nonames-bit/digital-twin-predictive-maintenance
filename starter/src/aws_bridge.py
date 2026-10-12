"""Cloud bridge: forwards the local MQTT feed to AWS IoT Core.

Subscribes to ``twin/telemetry`` on the local Mosquitto broker and republishes
every message to the same topic on AWS IoT Core over mutual-TLS MQTT. This is
the OT→cloud seam: the plant-floor path (ingest_mqtt.py → TimescaleDB) keeps
working with zero cloud dependency; this bridge is purely additive.

Prerequisites (see ../../terraform/README.md):
    terraform apply   →  IoT thing, policy, certificate + S3 bucket

Then place the certs and set in .env:
    AWS_IOT_ENDPOINT / AWS_IOT_THING_NAME
    AWS_IOT_CERT_PATH / AWS_IOT_KEY_PATH / AWS_IOT_CA_PATH

Usage:  python src/aws_bridge.py
"""
import json
import os
import sys
from pathlib import Path

from awsiot import mqtt_connection_builder
from dotenv import load_dotenv
from paho.mqtt import client as mqtt

load_dotenv()

LOCAL_TOPIC = "twin/telemetry"
CLOUD_TOPIC = "twin/telemetry"  # refined per-asset topics in production


def die(msg):
    print(f"aws_bridge: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    endpoint = os.getenv("AWS_IOT_ENDPOINT")
    thing = os.getenv("AWS_IOT_THING_NAME", "twin-gateway-01")
    cert = os.getenv("AWS_IOT_CERT_PATH", "./certs/device.pem.crt")
    key = os.getenv("AWS_IOT_KEY_PATH", "./certs/private.pem.key")
    ca = os.getenv("AWS_IOT_CA_PATH", "./certs/AmazonRootCA1.pem")
    if not endpoint:
        die("AWS_IOT_ENDPOINT not set — run terraform apply first (see terraform/README.md).")
    for p, name in [(cert, "cert"), (key, "key"), (ca, "CA")]:
        if not Path(p).exists():
            die(f"{name} file not found at {p} — download it from the AWS IoT console.")

    cloud = mqtt_connection_builder.mtls_from_path(
        endpoint=endpoint,
        cert_filepath=cert,
        pri_key_filepath=key,
        ca_filepath=ca,
        client_id=thing,
        clean_session=False,
        keep_alive_secs=30,
    )
    print(f"connecting to AWS IoT Core @ {endpoint} as {thing} …")
    cloud.connect().result()
    print("connected — bridging twin/telemetry → AWS IoT Core")

    def on_local_message(_cli, _ud, msg):
        cloud.publish(topic=CLOUD_TOPIC, payload=msg.payload, qos=1)

    local = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="twin-aws-bridge")
    local.on_message = on_local_message
    local.connect(os.getenv("MQTT_BROKER_HOST", "localhost"),
                  int(os.getenv("MQTT_BROKER_PORT", "1883")), keepalive=60)
    local.subscribe(LOCAL_TOPIC, qos=1)
    try:
        local.loop_forever()
    except KeyboardInterrupt:
        print("stopped")
    finally:
        cloud.disconnect()


if __name__ == "__main__":
    main()

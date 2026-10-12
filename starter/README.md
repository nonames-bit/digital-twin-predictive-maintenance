# Digital Twin — starter scaffold

Day-1 runnable skeleton for the predictive-maintenance Digital Twin
(AI4I 2020 dataset, milling machine).

## What's here

| File | What it does |
|---|---|
| `docker-compose.yml` | TimescaleDB (+pgvector), Mosquitto MQTT broker, MLflow tracking server |
| `.env.example` | Credentials — copy to `.env` and fill in |
| `requirements.txt` | Python deps |
| `src/db.py` | Shared TimescaleDB engine + hypertable schema |
| `src/train.py` | XGBoost baseline: stratified CV, SMOTE **inside** folds only, PR-AUC + recall, MLflow logging |
| `src/seed_db.py` | One-shot replay: creates the hypertable, inserts the 10,000 AI4I rows |
| `src/simulator.py` | Synthetic OT feed: replays AI4I rows as live MQTT messages on `twin/telemetry` |
| `src/ingest_mqtt.py` | Local OT path: subscribes to `twin/telemetry`, batch-writes the hypertable |
| `src/aws_bridge.py` | Cloud path: forwards the local feed to AWS IoT Core (mutual TLS), additive — the local path never depends on it |
| `mosquitto/mosquitto.conf` | Dev broker config (anonymous local access; TLS + auth in production) |
| `README.md` | Quickstart + roadmap to the full plan |

## Quickstart

```bash
cp .env.example .env
docker compose up -d            # Postgres:5432, Mosquitto:1883, MLflow:5000, Ignition:8088
pip install -r requirements.txt
python src/seed_db.py           # one-time: hypertable + minute-sampled replay
python src/train.py             # baseline run, logged to MLflow
```

Ignition Gateway: http://localhost:8088 (admin / `GATEWAY_ADMIN_PASSWORD`
from `.env`). See [../docs/ignition-integration.md](../docs/ignition-integration.md)
for the DB connection + Perspective dashboard setup.

## Live local feed (no cloud needed)

```bash
docker compose up -d mosquitto db
python src/ingest_mqtt.py &     # MQTT → TimescaleDB
python src/simulator.py         # synthetic sensor feed → twin/telemetry
```

## Cloud (optional — the local path works without any of this)

1. `cd ../terraform && terraform init && terraform apply` — creates the IoT
   thing + certificate and the S3 MLflow-artifact bucket.
2. Save the cert outputs to `starter/certs/` and fill `AWS_IOT_ENDPOINT` etc.
   in `.env` (see `terraform/README.md`).
3. `python src/aws_bridge.py` — every local `twin/telemetry` message is
   republished to AWS IoT Core.
4. For MLflow artifact backup to S3: set
   `MLFLOW_ARTIFACT_ROOT=s3://<bucket>/mlflow-artifacts` in `.env` and restart
   the mlflow service.

## Leakage guard (on purpose)

`TWF/HDF/PWF/OSF/RNF` are the components of the target (`Machine failure` is
their logical OR) — they are **dropped as features** in `train.py`. Using them
would be target leakage.

## Roadmap to the full plan (not built yet — your call on order)

1. **Live OT feed**: replace `seed_db.py` replay with MQTT Sparkplug B from the
   Ignition simulator → TimescaleDB.
2. **Inference API**: FastAPI + Docker, serves the MLflow-registered model.
3. **Alerts**: Telegram on high failure probability (keep it simple first).
4. **RAG assistant**: machine manuals in pgvector (same Postgres) + LangChain;
   predict failure *mode* (TWF/HDF/PWF/OSF/RNF as multi-label) → retrieve the
   matching manual section → alert the technician.
5. **Ignition Perspective**: dashboard reading the hypertable.

SMOTE stays inside training folds only; judge the model on **recall / PR-AUC**,
never accuracy (3.4% positives make accuracy meaningless).

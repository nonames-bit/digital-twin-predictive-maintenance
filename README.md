# DigitalTwin — Predictive Maintenance Digital Twin

Digital twin of a milling machine for predictive maintenance: a live MQTT
telemetry feed lands in TimescaleDB, an XGBoost model scores failure risk
(tracked in MLflow), and an optional AWS IoT Core bridge carries the same feed
to the cloud. Built as a portfolio project for applied AI in manufacturing.

> **Honest note:** the dataset is synthetic (AI4I 2020, UCI — 10,000 rows,
> 339 failures). The value here is the engineering: the OT→cloud data path,
> leakage-safe ML methodology, and MLOps plumbing — not the data itself.

## Architecture

```
simulator.py ──MQTT (twin/telemetry)──▶┬─▶ ingest_mqtt.py ─▶ TimescaleDB hypertable
   (synthetic                          │         │
    OT feed)                           │         ▼
                                       │    train.py (XGBoost, SMOTE inside folds)
                                       │         │  MLflow tracking + model registry
                                       │         ▼
                                       ├─▶ aws_bridge.py ─▶ AWS IoT Core (mutual TLS)
                                       │                     S3 ← MLflow artifacts (optional)
                                       ▼
                                  Ignition Gateway (Docker, :8088)
                                  Perspective dashboards ← named queries
```

## Quickstart (local, no cloud needed)

```bash
cd starter
cp .env.example .env
docker compose up -d        # TimescaleDB :5432, Mosquitto :1883, MLflow :5000
pip install -r requirements.txt
python src/seed_db.py       # one-shot: hypertable + 10k-row replay
python src/train.py         # baseline — logged to MLflow, model registered
```

Live feed:

```bash
docker compose up -d mosquitto db
python src/ingest_mqtt.py &  # MQTT → TimescaleDB
python src/simulator.py      # synthetic sensors → twin/telemetry
```

## Cloud (optional)

```bash
cd terraform && terraform init && terraform apply
```

Provisions the AWS IoT thing + certificate and the S3 MLflow-artifact bucket.
See [terraform/README.md](terraform/README.md), then run
`python starter/src/aws_bridge.py`.

## ML methodology (the part that matters)

- **No target leakage:** `TWF/HDF/PWF/OSF/RNF` are components of the target —
  dropped as features.
- **SMOTE inside CV folds only** (imbalanced-learn pipeline) — never before
  the split.
- **Judged on recall / PR-AUC**, never accuracy (3.4% positives).

## Layout

| Path | Contents |
|---|---|
| `starter/` | Runnable scaffold: compose, simulator, ingestor, trainer, AWS bridge |
| `terraform/` | AWS IoT Core thing + S3 bucket |
| `docs/` | Dataset reference + [Ignition integration guide](docs/ignition-integration.md) |
| `scada/named_queries/` | SQL for the Ignition Perspective dashboard |
| `ai4i2020.csv` | AI4I 2020 dataset (UCI, synthetic) |

## Roadmap

1. FastAPI inference API (Docker) serving the MLflow-registered model
2. Telegram alerts on high failure probability
3. RAG over machine manuals (pgvector, same Postgres) → failure-mode guidance
4. ~~Ignition Perspective dashboard~~ → done, see [docs/ignition-integration.md](docs/ignition-integration.md) (trial Gateway + named queries over the hypertable)

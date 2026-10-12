# Ignition Integration

Ignition reads the `machine_telemetry` hypertable directly over JDBC — no MQTT
modules required for the dashboard path. The trial is fully functional for
2 hours at a time and can be re-armed indefinitely; your data stays safe in
the Docker TimescaleDB container either way.

## 1. Run the Gateway (Docker — recommended)

The `ignition` service is already in `starter/docker-compose.yml`
(official `inductiveautomation/ignition:8.3` image, trial mode):

```bash
cd starter
cp .env.example .env   # set GATEWAY_ADMIN_PASSWORD
docker compose up -d ignition
```

Open **http://localhost:8088** → log in as `admin` with your password.
First boot commissions itself from the environment (no setup wizard clicks).

Notes:
- Trial is fully functional for 2 hours, re-armable; gateway data persists in
  the `ignition-data` volume either way.
- The same compose file runs on a cloud VPS unchanged — that's the
  "Ignition on a server" path when you want a public Perspective URL.
- Manual install (without Docker) also works: download the trial from
  inductiveautomation.com. In that case use `localhost` below instead of the
  compose service names.

## 2. Database connection to TimescaleDB

Gateway web UI → **Config → Databases → Connections → Create**:

| Field | Value |
|---|---|
| Name | `twin` |
| JDBC Driver | PostgreSQL |
| Connect URL | `jdbc:postgresql://db:5432/twin` |
| Username / Password | from `starter/.env` (`POSTGRES_USER` / `POSTGRES_PASSWORD`) |

`db` is the compose service name — the Gateway resolves it over the Docker
network. (Manual install: use `localhost`.) Save, then **Validate**.

## 3. Named queries

In the Designer, under your project → **Named Queries → New Folder `twin`**,
create these (SQL files also live in `scada/named_queries/` for copy-paste):

**`twin/latest_telemetry`** — last N rows for tables/KPIs
```sql
SELECT timestamp, product_id, machine_type, air_temp_k, process_temp_k,
       rpm, torque_nm, tool_wear_min, machine_failure
FROM machine_telemetry
ORDER BY timestamp DESC
LIMIT :limit
```
Parameters: `limit` (Integer, default 100).

**`twin/telemetry_range`** — time window for the trend chart
```sql
SELECT timestamp, rpm, torque_nm, air_temp_k, process_temp_k, tool_wear_min
FROM machine_telemetry
WHERE timestamp BETWEEN :startTs AND :endTs
ORDER BY timestamp
```
Parameters: `startTs`, `endTs` (DateTime).

**`twin/kpis`** — single latest row for the KPI cards
```sql
SELECT * FROM machine_telemetry ORDER BY timestamp DESC LIMIT 1
```

**`twin/failures_24h`** — alarm banner counter
```sql
SELECT COUNT(*) AS failures
FROM machine_telemetry
WHERE machine_failure = 1
  AND timestamp > NOW() - INTERVAL '24 hours'
```

## 4. Perspective view — `Twin/Overview`

Suggested layout (all bindings → Named Query):

- **KPI row** (Flex, 5 cards): Air temp, Process temp, RPM, Torque, Tool wear —
  each bound to `twin/kpis`, value property of the matching column.
- **Trend chart** (TimeSeries): bound to `twin/telemetry_range` with
  `startTs = now(-1h)`, `endTs = now()`. Pens: rpm, torque_nm.
- **Recent readings** (Table): bound to `twin/latest_telemetry`.
- **Alarm banner** (Label, red): visible binding
  `twin/failures_24h.failures > 0`, text `"⚠ FAILURES IN LAST 24H"`.

## 5. Optional: live tags over MQTT

If you want Ignition tags instead of (or next to) DB polling:

1. Install the **Cirrus Link MQTT Engine** module (trial available) on the Gateway.
2. MQTT Engine → Servers → new server: `tcp://mosquitto:1883`, no auth (dev).
   (`mosquitto` is the compose service name; manual install: `localhost`.)
3. Subscribe to `twin/telemetry`; use a JSON payload parser to split fields
   into tags under `[MQTT Engine]twin/telemetry/...`.
4. Bind Perspective components to the tags for sub-second updates; keep the
   named queries for history/trends.

## 6. What runs where (recap)

| Piece | Runs on |
|---|---|
| TimescaleDB, Mosquitto, MLflow | Docker (`starter/docker-compose.yml`) |
| simulator / ingest_mqtt / train | Python on your PC |
| Perspective dashboard, named queries | Ignition trial Gateway |
| AWS IoT bridge, S3, Terraform | Optional cloud path |

Production notes: give Mosquitto TLS + credentials, move the Gateway behind
HTTPS, and use a licensed Ignition (the trial is for dev/portfolio only).

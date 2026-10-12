# Rama `referencia` — andamio del Digital Twin

Esta rama es el **andamio de referencia** del proyecto: una implementación
funcional de la arquitectura (MQTT -> TimescaleDB -> XGBoost/MLflow ->
Ignition en Docker -> puente AWS opcional).

- `main` es donde Jaime construye el proyecto él mismo (modo mentor).
- Esta rama es solo para **leer y consultar**: ver cómo se conecta cada
  pieza, copiar patrones y comparar con tu propio código.

Estructura:
- `starter/` — docker-compose (TimescaleDB, Mosquitto, MLflow, Ignition),
  simulador e ingesta MQTT, entrenamiento XGBoost, puente AWS IoT Core.
- `terraform/` — IoT thing, políticas, certificados y bucket S3.
- `docs/` — guía de integración con Ignition.
- `scada/named_queries/` — consultas SQL para los dashboards de Perspective.
- `ai4i2020.csv` — dataset de ejemplo (sintético, 10k filas).

Validado: sintaxis Python y YAML del compose. `docker compose up` y
`terraform plan` están sin ejecutar: eso te toca a ti. :-)

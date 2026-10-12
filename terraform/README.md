# Terraform — AWS side of the Digital Twin

Provisions the cloud half of the project:

| Resource | Used by |
|---|---|
| S3 bucket (versioned, private) | MLflow artifact backup — set `MLFLOW_ARTIFACT_ROOT=s3://<bucket>/mlflow-artifacts` in `starter/.env` |
| IoT thing + policy + certificate | `starter/src/aws_bridge.py` — mutual-TLS MQTT bridge local → AWS IoT Core |

## Usage

```bash
cd terraform
terraform init
terraform plan
terraform apply
```

After apply:

1. Save the device certificate and private key from the outputs:
   ```bash
   mkdir -p ../starter/certs
   terraform output -raw certificate_pem > ../starter/certs/device.pem.crt
   terraform output -raw private_key   > ../starter/certs/private.pem.key
   # Amazon Root CA: curl -o ../starter/certs/AmazonRootCA1.pem \
   #   https://www.amazontrust.com/repository/AmazonRootCA1.pem
   ```
2. Copy the outputs into `starter/.env`:
   - `AWS_IOT_ENDPOINT` ← `iot_data_endpoint`
   - `AWS_IOT_THING_NAME` ← `iot_thing_name`
   - `MLFLOW_ARTIFACT_ROOT=s3://<mlflow_artifact_bucket>/mlflow-artifacts` (optional)
3. Run the bridge: `python starter/src/aws_bridge.py`

## Notes

- The IoT policy is least-privilege: connect + publish/subscribe only under
  `twin/*`. Tighten per-asset topics in production.
- `certs/` is gitignored — never commit private keys.
- `terraform destroy` removes everything; the S3 bucket must be empty first.

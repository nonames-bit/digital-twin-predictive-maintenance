output "mlflow_artifact_bucket" {
  description = "S3 bucket for MLflow artifact backup"
  value       = aws_s3_bucket.mlflow_artifacts.id
}

output "iot_thing_name" {
  description = "IoT thing name for the bridge"
  value       = aws_iot_thing.gateway.name
}

output "iot_data_endpoint" {
  description = "ATS data endpoint → AWS_IOT_ENDPOINT in .env"
  value       = data.aws_iot_endpoint.data_ats.endpoint_address
}

output "certificate_pem" {
  description = "Device certificate — save to certs/device.pem.crt (shown once)"
  value       = aws_iot_certificate.gateway.certificate_pem
  sensitive   = true
}

output "private_key" {
  description = "Device private key — save to certs/private.pem.key (shown once)"
  value       = aws_iot_certificate.gateway.private_key
  sensitive   = true
}

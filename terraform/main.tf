provider "aws" {
  region = var.aws_region
}

data "aws_partition" "current" {}
data "aws_caller_identity" "current" {}

# --- S3 bucket: MLflow artifact backup ---
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "${var.project}-mlflow-artifacts-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "mlflow_artifacts" {
  bucket                  = aws_s3_bucket.mlflow_artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- AWS IoT Core: thing + least-privilege policy + certificate ---
resource "aws_iot_thing" "gateway" {
  name = var.thing_name
}

resource "aws_iot_policy" "gateway" {
  name = "${var.project}-gateway-policy"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["iot:Connect"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["iot:Publish"]
        Resource = "arn:${data.aws_partition.current.partition}:iot:${var.aws_region}:${data.aws_caller_identity.current.account_id}:topic/twin/*"
      },
      {
        Effect   = "Allow"
        Action   = ["iot:Subscribe", "iot:Receive"]
        Resource = "arn:${data.aws_partition.current.partition}:iot:${var.aws_region}:${data.aws_caller_identity.current.account_id}:topicfilter/twin/*"
      },
    ]
  })
}

resource "aws_iot_certificate" "gateway" {
  active = true
}

resource "aws_iot_policy_attachment" "gateway" {
  policy = aws_iot_policy.gateway.name
  target = aws_iot_certificate.gateway.arn
}

resource "aws_iot_thing_principal_attachment" "gateway" {
  thing     = aws_iot_thing.gateway.name
  principal = aws_iot_certificate.gateway.arn
}

# IoT data (ATS) endpoint for the bridge's .env
data "aws_iot_endpoint" "data_ats" {
  endpoint_type = "iot:Data-ATS"
}

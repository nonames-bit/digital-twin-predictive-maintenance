variable "aws_region" {
  description = "AWS region for the IoT thing and artifact bucket"
  type        = string
  default     = "ca-central-1"
}

variable "project" {
  description = "Name prefix for all resources"
  type        = string
  default     = "twin"
}

variable "thing_name" {
  description = "AWS IoT thing name for the gateway bridge"
  type        = string
  default     = "twin-gateway-01"
}

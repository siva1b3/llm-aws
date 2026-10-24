terraform {
  # Declare the providers this configuration requires.
  required_providers {
    aws = {
      # Use the official AWS provider published by HashiCorp.
      source = "hashicorp/aws"
    }
  }
}

provider "aws" {
  # Use the US East (N. Virginia) region by default.
  region              = "us-east-1"
  # Reject credentials for any AWS account outside this list.
  allowed_account_ids = ["900065085340"]
}

# Read the account ID, ARN, and user ID of the active AWS identity.
# This data source retrieves information; it does not create AWS resources.
data "aws_caller_identity" "current" {}

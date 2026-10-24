terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

provider "aws" {
  region              = "us-east-1"
  allowed_account_ids = ["900065085340"]
}

data "aws_caller_identity" "current" {}

output "aws_account_id" {
  value = data.aws_caller_identity.current.account_id
}

output "aws_identity" {
  value = data.aws_caller_identity.current.arn
}

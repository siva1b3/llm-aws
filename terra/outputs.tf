output "aws_account_id" {
  value = data.aws_caller_identity.current.account_id
}

output "aws_identity" {
  value = data.aws_caller_identity.current.arn
}

output "aws_user_id" {
  description = "Unique ID of the IAM user or assumed-role session"
  value       = data.aws_caller_identity.current.user_id
}

output "aws_summary" {
  description = "AWS identity details in one place"
  value = {
    account_id = data.aws_caller_identity.current.account_id
    arn        = data.aws_caller_identity.current.arn
    user_id    = data.aws_caller_identity.current.user_id
  }
}

output "hello_aws" {
  value = "Hello from AWS account ${data.aws_caller_identity.current.account_id}!"
}

output "account_console_url" {
  value = "https://${data.aws_caller_identity.current.account_id}.signin.aws.amazon.com/console"
}
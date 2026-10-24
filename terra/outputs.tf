output "aws_account_id" {
  value = data.aws_caller_identity.current.account_id
}

output "aws_identity" {
  value = data.aws_caller_identity.current.arn
}

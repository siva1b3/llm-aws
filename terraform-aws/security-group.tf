data "aws_security_group" "main" {
  id = "sg-0849de47513ee506e"
}

output "selected_security_group" {
  value = data.aws_security_group.main.id
}
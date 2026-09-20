# This learning example stores the private key in Terraform state.
resource "tls_private_key" "ec2" {
  count     = var.create_key_pair ? 1 : 0
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "generated" {
  count      = var.create_key_pair ? 1 : 0
  key_name   = var.new_key_pair_name
  public_key = tls_private_key.ec2[0].public_key_openssh
}

# The Compose bind mount persists this file on your Windows machine.
resource "local_sensitive_file" "private_key" {
  count                = var.create_key_pair ? 1 : 0
  content              = tls_private_key.ec2[0].private_key_pem
  filename             = "${path.module}/keys/${var.new_key_pair_name}.pem"
  file_permission      = "0600"
  directory_permission = "0700"
}

output "generated_key_pair_name" {
  value = one(aws_key_pair.generated[*].key_name)
}

output "generated_pem_path" {
  value = one(local_sensitive_file.private_key[*].filename)
}

# Reuse your existing key pair and security group; Terraform does not own them.
# With no subnet_id, EC2 selects a default subnet in the default VPC.
resource "aws_instance" "three" {
  count                       = var.enable_three_instances ? 3 : 0
  ami                         = "ami-0f8a61b66d1accaee"
  instance_type               = "t3a.large"
  key_name                    = "terraform-ec2-key"
  vpc_security_group_ids      = ["sg-0849de47513ee506e"]
  associate_public_ip_address = true
  ebs_optimized               = true
  hibernation                 = false
  monitoring                  = false

  # The root device and its base snapshot come from the selected AMI.
  root_block_device {
    encrypted             = false
    delete_on_termination = true
    volume_size           = 60
    volume_type           = "gp3"
    iops                  = 3000
    throughput            = 125
  }

  credit_specification {
    cpu_credits = "standard"
  }

  instance_market_options {
    market_type = "spot"

    spot_options {
      spot_instance_type             = "one-time"
      instance_interruption_behavior = "terminate"
    }
  }

  private_dns_name_options {
    hostname_type                        = "ip-name"
    enable_resource_name_dns_a_record    = true
    enable_resource_name_dns_aaaa_record = false
  }

  maintenance_options {
    auto_recovery = "disabled"
  }

  tags = {
    Name = "test-three-${count.index + 1}"
  }
}


output "ec2_three_instance_ids" {
  value = aws_instance.three[*].id
}

output "ec2_three_public_ips" {
  value = aws_instance.three[*].public_ip
}

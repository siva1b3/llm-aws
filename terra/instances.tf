locals {
  enabled_machines = { for name, machine in var.machines : name => machine if machine.is_true }
}

data "aws_subnet" "selected" {
  count = length(local.enabled_machines) > 0 ? 1 : 0
  id    = "subnet-c63cabe7" # us-east-1a, vpc-72db7f0f
}

resource "aws_instance" "machine" {
  for_each = local.enabled_machines

  ami                         = "ami-02a9cdd4e98bc3c78"
  instance_type               = "t3a.large"
  key_name                    = "main"
  subnet_id                   = data.aws_subnet.selected[0].id
  vpc_security_group_ids      = ["sg-0849de47513ee506e"]
  associate_public_ip_address = true
  ebs_optimized               = true
  monitoring                  = false
  disable_api_termination     = false
  disable_api_stop            = false

  # The root device and its source snapshot are inherited from the AMI.
  root_block_device {
    encrypted             = false
    delete_on_termination = true
    volume_size           = 24
    volume_type           = "gp3"
    iops                  = 3000
    throughput            = 125
  }

  credit_specification {
    cpu_credits = "standard"
  }

  dynamic "instance_market_options" {
    for_each = each.value.is_spot ? [true] : []
    content {
      market_type = "spot"
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
    Name = each.key
  }

  lifecycle {
    precondition {
      condition     = data.aws_subnet.selected[0].vpc_id == "vpc-72db7f0f"
      error_message = "The subnet must belong to vpc-72db7f0f, the VPC of security group sg-0849de47513ee506e."
    }
  }
}

output "machines" {
  description = "Created instances and their connection addresses."
  value = {
    for name, instance in aws_instance.machine : name => {
      id         = instance.id
      public_ip  = instance.public_ip
      private_ip = instance.private_ip
      is_spot    = var.machines[name].is_spot
    }
  }
}

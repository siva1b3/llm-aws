locals {
  # Resolve omitted disk sizes before creating enabled machines.
  enabled_machines = {
    for name, machine in var.machines : name => merge(machine, {
      root_volume_size = coalesce(machine.root_volume_size, local.ec2_defaults.root_volume.volume_size)
    }) if machine.is_true
  }
}

data "aws_subnet" "selected" {
  count = length(local.enabled_machines) > 0 ? 1 : 0
  id    = local.ec2_defaults.subnet_id
}

resource "aws_instance" "machine" {
  for_each = local.enabled_machines

  ami                         = local.ec2_defaults.ami
  instance_type               = each.value.instance_type
  key_name                    = local.ec2_defaults.key_name
  subnet_id                   = data.aws_subnet.selected[0].id
  vpc_security_group_ids      = local.ec2_defaults.security_group_ids
  associate_public_ip_address = local.ec2_defaults.associate_public_ip_address
  ebs_optimized               = local.ec2_defaults.ebs_optimized
  monitoring                  = local.ec2_defaults.monitoring
  disable_api_termination     = local.ec2_defaults.disable_api_termination
  disable_api_stop            = local.ec2_defaults.disable_api_stop

  # The root device and source snapshot are inherited from the AMI.
  root_block_device {
    encrypted             = local.ec2_defaults.root_volume.encrypted
    delete_on_termination = local.ec2_defaults.root_volume.delete_on_termination
    volume_size           = each.value.root_volume_size
    volume_type           = local.ec2_defaults.root_volume.volume_type
    iops                  = local.ec2_defaults.root_volume.iops
    throughput            = local.ec2_defaults.root_volume.throughput
  }

  # CPU credits apply only to burstable T-family instances, not GPU instances.
  dynamic "credit_specification" {
    for_each = contains(["t2", "t3", "t3a", "t4g"], split(".", each.value.instance_type)[0]) ? [true] : []
    content {
      cpu_credits = local.ec2_defaults.cpu_credits
    }
  }

  # Omit this block for On-Demand machines.
  dynamic "instance_market_options" {
    for_each = each.value.is_spot ? [true] : []
    content {
      market_type = "spot"
    }
  }

  private_dns_name_options {
    hostname_type                        = local.ec2_defaults.hostname_type
    enable_resource_name_dns_a_record    = local.ec2_defaults.enable_resource_name_dns_a_record
    enable_resource_name_dns_aaaa_record = local.ec2_defaults.enable_resource_name_dns_aaaa_record
  }

  maintenance_options {
    auto_recovery = local.ec2_defaults.auto_recovery
  }

  tags = {
    Name = each.key
  }

  lifecycle {
    precondition {
      condition     = data.aws_subnet.selected[0].vpc_id == local.ec2_defaults.vpc_id
      error_message = "The subnet must belong to the configured VPC ${local.ec2_defaults.vpc_id}."
    }
  }
}

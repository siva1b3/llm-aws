data "aws_subnet" "selected" {
  count = length(local.enabled_machines) > 0 ? 1 : 0
  id    = local.shared_settings.subnet_id
}

resource "aws_instance" "machine" {
  for_each = local.enabled_machines

  ami                                  = local.shared_settings.ami
  instance_type                        = local.machine_specs[each.key].instance_type
  key_name                             = local.shared_settings.key_name
  subnet_id                            = data.aws_subnet.selected[0].id
  vpc_security_group_ids               = local.shared_settings.security_group_ids
  associate_public_ip_address          = true
  ebs_optimized                        = true
  monitoring                           = false
  disable_api_termination              = false
  disable_api_stop                     = false
  instance_initiated_shutdown_behavior = "stop"

  # The root device and its source snapshot are inherited from the AMI.
  root_block_device {
    encrypted             = false
    delete_on_termination = true
    volume_size           = local.machine_specs[each.key].volume_size
    volume_type           = "gp3"
    iops                  = 3000
    throughput            = 125
  }

  dynamic "credit_specification" {
    for_each = local.machine_specs[each.key].burstable ? [true] : []
    content {
      cpu_credits = "standard"
    }
  }

  dynamic "instance_market_options" {
    for_each = each.value.is_spot ? [true] : []
    content {
      market_type = "spot"
      spot_options {
        spot_instance_type             = "persistent"
        instance_interruption_behavior = "stop"
      }
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
      condition     = data.aws_subnet.selected[0].vpc_id == local.shared_settings.vpc_id
      error_message = "The subnet must belong to vpc-72db7f0f, the VPC of security group sg-0849de47513ee506e."
    }
  }
}

resource "aws_ec2_instance_state" "machine" {
  for_each    = local.enabled_machines
  instance_id = aws_instance.machine[each.key].id
  state       = local.desired_states[each.key]
}

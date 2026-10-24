mock_provider "aws" {
  mock_data "aws_subnet" {
    defaults = {
      vpc_id = "vpc-72db7f0f"
    }
  }
}

run "basic_only_destroys_gpu" {
  command = plan
  variables {
    system_state = "all_basic_on"
  }
  assert {
    condition = (
      length(aws_instance.machine) == 3 &&
      alltrue([for name in ["test_002", "test_003", "test_004"] : local.desired_states[name] == "running"]) &&
      alltrue([for name in ["gpu_test_001", "gpu_test_002", "gpu_test_003"] : local.desired_states[name] == "destroyed"])
    )
    error_message = "Basic-only must run all basic machines and destroy every GPU, regardless of is_true."
  }
}

run "explicit_stop_keeps_gpu" {
  command = plan
  variables {
    system_state = "basic_on_gpu_stopped"
  }
  assert {
    condition = (
      length(aws_instance.machine) == 6 &&
      aws_ec2_instance_state.machine["gpu_test_001"].state == "stopped" &&
      aws_ec2_instance_state.machine["test_002"].state == "running"
    )
    error_message = "Explicit stop must retain GPU resources and stop them."
  }
}

run "spot_membership" {
  command = plan
  variables {
    system_state = "all_spot_on"
  }
  assert {
    condition = (
      length(aws_instance.machine) == 4 &&
      alltrue([for name in ["test_002", "test_003", "test_004", "gpu_test_003"] : local.desired_states[name] == "running"]) &&
      alltrue([for name in ["gpu_test_001", "gpu_test_002"] : local.desired_states[name] == "destroyed"])
    )
    error_message = "Spot preset must select by is_spot independently of is_true."
  }
}

run "omitted_purchase_option_uses_catalog" {
  command = plan
  variables {
    system_state = "all_on"
    machines = {
      test_002     = { is_true = true }
      gpu_test_001 = { is_true = true }
      gpu_test_003 = { is_true = true, is_spot = null }
    }
  }
  assert {
    condition = (
      aws_instance.machine["test_002"].instance_market_options[0].market_type == "spot" &&
      length(aws_instance.machine["gpu_test_001"].instance_market_options) == 0 &&
      aws_instance.machine["gpu_test_003"].instance_market_options[0].market_type == "spot" &&
      output.machines["test_002"].is_spot == true &&
      output.machines["gpu_test_001"].is_spot == false
    )
    error_message = "Omitted or null purchase options must use catalog defaults in resources and outputs."
  }
}

run "off_destroys_everything" {
  command = plan
  variables {
    system_state = "all_off"
  }
  assert {
    condition     = length(aws_instance.machine) == 0 && length(aws_ec2_instance_state.machine) == 0
    error_message = "Off must remove all instance and power-state resources."
  }
}

run "custom_preserves_controls" {
  command = plan
  variables {
    system_state = "custom"
  }
  assert {
    condition     = alltrue([for name, machine in var.machines : local.desired_states[name] == (machine.is_true ? "running" : "destroyed")])
    error_message = "Custom must respect the existing is_true selections."
  }
}

run "reject_unknown_preset" {
  command = plan
  variables {
    system_state = "typo"
  }
  expect_failures = [var.system_state]
}

run "all_stopped_retains_every_machine" {
  command = plan
  variables {
    system_state = "all_stopped"
  }
  assert {
    condition = (
      length(aws_instance.machine) == 6 &&
      alltrue([for machine in aws_ec2_instance_state.machine : machine.state == "stopped"])
    )
    error_message = "Stopping all must retain all six instances, including those disabled in custom mode."
  }
}

run "purchase_override_and_partial_catalog" {
  command = plan
  variables {
    system_state = "spot_on_on_demand_stopped"
    machines = {
      test_002     = { is_true = false, is_spot = false }
      gpu_test_001 = { is_true = false, is_spot = true }
    }
  }
  assert {
    condition = (
      length(aws_instance.machine) == 2 &&
      aws_ec2_instance_state.machine["test_002"].state == "stopped" &&
      aws_ec2_instance_state.machine["gpu_test_001"].state == "running" &&
      length(aws_instance.machine["test_002"].instance_market_options) == 0 &&
      aws_instance.machine["gpu_test_001"].instance_market_options[0].market_type == "spot"
    )
    error_message = "Presets must honor changed purchase settings and only manage names present in the controls."
  }
}

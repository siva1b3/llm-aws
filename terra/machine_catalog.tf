# Machine defaults. Optional user overrides live in variables.tf.
locals {
  shared_settings = {
    ami                = "ami-02a9cdd4e98bc3c78"
    key_name           = "main"
    subnet_id          = "subnet-c63cabe7" # us-east-1a
    vpc_id             = "vpc-72db7f0f"
    security_group_ids = ["sg-0849de47513ee506e"]
  }

  # Group membership is explicit; CPU-credit support is a separate setting.
  machine_specs = {
    test_002     = { group = "basic", instance_type = "t3a.large", volume_size = 24, burstable = true, is_spot = true }
    test_003     = { group = "basic", instance_type = "t3a.large", volume_size = 24, burstable = true, is_spot = true }
    test_004     = { group = "basic", instance_type = "t3a.large", volume_size = 24, burstable = true, is_spot = true }
    gpu_test_001 = { group = "gpu", instance_type = "g4dn.xlarge", volume_size = 128, burstable = false, is_spot = false }
    gpu_test_002 = { group = "gpu", instance_type = "g4dn.xlarge", volume_size = 128, burstable = false, is_spot = false }
    gpu_test_003 = { group = "gpu", instance_type = "g4dn.xlarge", volume_size = 128, burstable = false, is_spot = true }
  }

  # Resolve purchase options once. Explicit true/false overrides the catalog;
  # an omitted (or null) is_spot inherits the machine's default.
  machine_controls = {
    for name, controls in var.machines : name => {
      is_true = controls.is_true
      is_spot = coalesce(controls.is_spot, local.machine_specs[name].is_spot)
    }
  }
}

# Read these tables to see exactly what each preset does.
# "destroyed" removes the instance; "stopped" retains it and its EBS disk.
locals {
  group_presets = {
    all_on               = { basic = "running", gpu = "running" }
    all_off              = { basic = "destroyed", gpu = "destroyed" }
    all_destroyed        = { basic = "destroyed", gpu = "destroyed" }
    all_stopped          = { basic = "stopped", gpu = "stopped" }
    all_basic_on         = { basic = "running", gpu = "destroyed" }
    all_gpu_on           = { basic = "destroyed", gpu = "running" }
    all_basic_stopped    = { basic = "stopped", gpu = "destroyed" }
    all_gpu_stopped      = { basic = "destroyed", gpu = "stopped" }
    basic_on_gpu_stopped = { basic = "running", gpu = "stopped" }
    gpu_on_basic_stopped = { basic = "stopped", gpu = "running" }
  }

  purchase_presets = {
    all_spot_on               = { spot = "running", on_demand = "destroyed" }
    all_on_demand_on          = { spot = "destroyed", on_demand = "running" }
    all_spot_stopped          = { spot = "stopped", on_demand = "destroyed" }
    all_on_demand_stopped     = { spot = "destroyed", on_demand = "stopped" }
    spot_on_on_demand_stopped = { spot = "running", on_demand = "stopped" }
    on_demand_on_spot_stopped = { spot = "stopped", on_demand = "running" }
  }
}

# Translate each table into the same shape: machine name => desired state.
# No AWS operations occur here.
locals {
  group_states = {
    for preset, states in local.group_presets : preset => {
      for name, machine in local.machine_controls : name => states[local.machine_specs[name].group]
    }
  }

  purchase_states = {
    for preset, states in local.purchase_presets : preset => {
      for name, machine in local.machine_controls : name => states[machine.is_spot ? "spot" : "on_demand"]
    }
  }

  custom_states = {
    for name, machine in local.machine_controls : name => machine.is_true ? "running" : "destroyed"
  }

  # Named presets override is_true; is_spot remains the purchase setting.
  preset_states  = merge(local.group_states, local.purchase_states, { custom = local.custom_states })
  desired_states = local.preset_states[var.system_state]

  # Only retained machines reach the AWS resources in instances.tf.
  enabled_machines = {
    for name, machine in local.machine_controls : name => machine
    if local.desired_states[name] != "destroyed"
  }
}

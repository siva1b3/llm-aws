variable "system_state" {
  description = "Preset for all machines. Unselected machines are destroyed; only explicit stopped presets retain them. custom uses each machine's is_true."
  type        = string
  default     = "all_destroyed"
  nullable    = false

  validation {
    condition = contains([
      "custom", "all_on", "all_off", "all_destroyed", "all_stopped",
      "all_basic_on", "all_gpu_on", "all_spot_on", "all_on_demand_on",
      "all_basic_stopped", "all_gpu_stopped", "all_spot_stopped", "all_on_demand_stopped",
      "basic_on_gpu_stopped", "gpu_on_basic_stopped",
      "spot_on_on_demand_stopped", "on_demand_on_spot_stopped"
    ], var.system_state)
    error_message = "Choose a supported system_state preset listed in SYSTEM_STATE.md."
  }
}

variable "machines" {
  description = "Per-machine controls. is_true applies in custom mode; presets override it. is_spot optionally overrides the catalog default in every mode."
  type = map(object({
    is_true = bool
    is_spot = optional(bool)
  }))
  default = {
    gpu_test_001 = {
      is_true = false
    }
    gpu_test_002 = {
      is_true = false
    }
    gpu_test_003 = {
      is_true = true
    }
    test_002 = {
      is_true = false
    }
    test_003 = {
      is_true = true
    }
    test_004 = {
      is_true = true
    }
  }

  validation {
    condition = alltrue([
      for name in keys(var.machines) : contains(["gpu_test_001", "gpu_test_002", "gpu_test_003", "test_002", "test_003", "test_004"], name)
    ])
    error_message = "Machine names must be gpu_test_001, gpu_test_002, gpu_test_003, test_002, test_003, or test_004."
  }
}

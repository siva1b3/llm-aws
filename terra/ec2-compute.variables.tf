variable "machines" {
  description = "Machines keyed by Name tag. is_true controls management, not detection of existing instances. Disabling a managed machine plans its destruction."
  type = map(object({
    is_true       = optional(bool, true)
    is_spot       = optional(bool, false)
    instance_type = optional(string, "t3a.large")
    # Omit this to inherit the shared root disk size.
    root_volume_size = optional(number)
  }))
  default = {}

  validation {
    condition = alltrue([
      for machine in values(var.machines) : machine.root_volume_size == null ? true : (
        machine.root_volume_size >= 1 && floor(machine.root_volume_size) == machine.root_volume_size
      )
    ])
    error_message = "root_volume_size must be a positive whole number of GiB when specified."
  }
}

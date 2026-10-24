variable "machines" {
  description = "Machines keyed by Name tag. is_true controls management, not detection of existing instances. Disabling a managed machine plans its destruction."
  type = map(object({
    is_true       = optional(bool, true)
    is_spot       = optional(bool, false)
    instance_type = optional(string, "t3a.large")
  }))
  default = {}
}

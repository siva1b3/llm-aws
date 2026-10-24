variable "machines" {
  description = "Per-machine controls. Disabling a managed machine destroys it on apply."
  type = map(object({
    is_true = bool
    is_spot = bool
  }))
  default = {
    test_001 = {
      is_true = false
      is_spot = true
    }
  }
}


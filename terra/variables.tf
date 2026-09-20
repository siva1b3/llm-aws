
variable "create_key_pair" {
  description = "Generate an SSH key, register its public key in AWS, and save a local PEM."
  type        = bool
  default     = false
}

variable "new_key_pair_name" {
  description = "Name for the new AWS key pair and local PEM file; must not already exist in AWS."
  type        = string
  default     = "terraform-generated-key"

  validation {
    condition     = can(regex("^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$", var.new_key_pair_name))
    error_message = "Use 1-64 letters, digits, underscores, or hyphens, starting with a letter or digit."
  }
}

variable "enable_small_instance" {
  description = "Create an additional t3.micro Spot instance."
  type        = bool
  default     = false
}

variable "enable_three_instances" {
  description = "Create three additional t3a.large Spot instances."
  type        = bool
  default     = false
}

variable "enable_single_instances" {
  description = "Manage the original single t3a.large Spot instance."
  type        = bool
  default     = false
}

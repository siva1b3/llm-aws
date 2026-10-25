variable "buckets" {
  description = "Buckets keyed by a logical identifier. Each name must be globally unique. Disabling a managed bucket plans its destruction; non-empty buckets are protected by force_destroy = false."
  type = map(object({
    name       = string
    is_true    = optional(bool, true)
    versioning = optional(bool, false)
  }))
  default = {}
}

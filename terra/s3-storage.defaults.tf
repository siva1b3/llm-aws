# Shared settings for all buckets.
locals {
  s3_defaults = {
    force_destroy           = true
    sse_algorithm           = "AES256"
    block_public_acls       = true
    block_public_policy     = true
    ignore_public_acls      = true
    restrict_public_buckets = true
  }
}

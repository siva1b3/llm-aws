# Each map key becomes the instance Name tag.
machines = {
  "dev-server-1" = {
    is_true          = false
    is_spot          = true
    instance_type    = "t3a.small"
    root_volume_size = 24
  }

  # Same instance type and root disk size as dev-server-1.
  "dev-server-2" = {
    is_true          = false
    is_spot          = true
    instance_type    = "t3a.large"
    root_volume_size = 24
  }

  "dev-server-3" = {
    is_true          = false
    is_spot          = true
    instance_type    = "t3a.large"
    root_volume_size = 24
  }

  # Overrides instance type and root disk size; other settings remain shared.
  "gpu-worker-1" = {
    is_true          = false
    is_spot          = true
    instance_type    = "g4dn.xlarge"
    root_volume_size = 128
  }
}

# Each map key identifies a bucket; name is the globally unique S3 bucket name.
buckets = {
  "dev-storage-1" = {
    name       = "terra-dev-storage-900065085340-1"
    is_true    = false
    versioning = false
  }
}

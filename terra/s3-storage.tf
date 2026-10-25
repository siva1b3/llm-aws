locals {
  enabled_buckets = {
    for name, bucket in var.buckets : name => bucket if bucket.is_true
  }
}

resource "aws_s3_bucket" "bucket" {
  for_each = local.enabled_buckets

  bucket        = each.value.name
  force_destroy = local.s3_defaults.force_destroy

  tags = {
    Name = each.value.name
  }
}

resource "aws_s3_bucket_public_access_block" "bucket" {
  for_each = local.enabled_buckets

  bucket                  = aws_s3_bucket.bucket[each.key].id
  block_public_acls       = local.s3_defaults.block_public_acls
  block_public_policy     = local.s3_defaults.block_public_policy
  ignore_public_acls      = local.s3_defaults.ignore_public_acls
  restrict_public_buckets = local.s3_defaults.restrict_public_buckets
}

resource "aws_s3_bucket_server_side_encryption_configuration" "bucket" {
  for_each = local.enabled_buckets

  bucket = aws_s3_bucket.bucket[each.key].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = local.s3_defaults.sse_algorithm
    }
  }
}

resource "aws_s3_bucket_versioning" "bucket" {
  for_each = local.enabled_buckets

  bucket = aws_s3_bucket.bucket[each.key].id

  # Once enabled, versioning can be suspended but not returned to unversioned.
  versioning_configuration {
    status = each.value.versioning ? "Enabled" : "Suspended"
  }
}

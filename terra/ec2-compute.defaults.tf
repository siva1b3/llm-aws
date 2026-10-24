# Shared settings for all machines. The AMI determines the operating system.
locals {
  ec2_defaults = {
    ami                         = "ami-02a9cdd4e98bc3c78"
    subnet_id                   = "subnet-c63cabe7"
    vpc_id                      = "vpc-72db7f0f"
    security_group_ids          = ["sg-0849de47513ee506e"]
    key_name                    = "main"
    associate_public_ip_address = true
    ebs_optimized               = true
    monitoring                  = false
    disable_api_termination     = false
    disable_api_stop            = false
    root_volume = {
      encrypted             = false
      delete_on_termination = true
      volume_size           = 24
      volume_type           = "gp3"
      iops                  = 3000
      throughput            = 125
    }
    cpu_credits                          = "standard"
    hostname_type                        = "ip-name"
    enable_resource_name_dns_a_record    = true
    enable_resource_name_dns_aaaa_record = false
    auto_recovery                        = "disabled"
  }
}

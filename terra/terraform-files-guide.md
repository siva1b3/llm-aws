# Terraform files guide

This project manages multiple EC2 machines using one resource definition. Common settings are shared, while each machine has its own name, enabled flag, Spot flag, instance type, and root disk size.

## File overview

| File | Purpose | When to edit |
|---|---|---|
| `main.tf` | AWS provider, region, allowed account, and caller identity lookup | Changing provider or account settings |
| `ec2-compute.defaults.tf` | Shared EC2 settings in `local.ec2_defaults` | Changing settings shared by all machines |
| `ec2-compute.variables.tf` | Input structure, optional defaults, and validation for `var.machines` | Adding or changing supported machine inputs |
| `terraform.tfvars` | Machine names and their individual settings | Adding machines or changing their configuration |
| `ec2-compute.tf` | Resolves settings and defines the EC2 resources | Changing how machines are provisioned |
| `outputs.tf` | Instance addresses and AWS identity outputs | Changing what Terraform displays |

Terraform reads the `.tf` files in this directory together as one configuration. Their names organize the code; they do not set execution order. Terraform uses references between resources to determine dependencies.

## 1. main.tf

This file declares the `hashicorp/aws` provider and configures:

- Region: `us-east-1`.
- Allowed AWS account: `900065085340`.
- `data.aws_caller_identity.current`: reads the identity used by Terraform.

The caller identity data source reads information; it does not create an AWS resource. Credentials are supplied separately through the Docker Compose environment, not defined in this file.

## 2. ec2-compute.defaults.tf

This file defines `local.ec2_defaults`, which holds the common settings used by every enabled machine.

| Setting | Current value |
|---|---|
| AMI | `ami-02a9cdd4e98bc3c78` |
| Subnet | `subnet-c63cabe7` |
| Expected VPC | `vpc-72db7f0f` |
| Security group | `sg-0849de47513ee506e` |
| SSH key pair name | `main` |
| Public IP requested | `true` |
| EBS optimized | `true` |
| Detailed monitoring | `false` |
| API termination protection | Disabled |
| API stop protection | Disabled |
| Default root disk size | 24 GiB |
| Root disk type | `gp3` |
| Root disk IOPS | 3000 |
| Root disk throughput | 125 MiB/s |
| Root disk encryption setting | `false` |
| Delete root disk on termination | `true` |
| CPU credit mode | `standard`, applied only to supported T families |
| Hostname type | `ip-name` |
| Resource-name DNS A record | Enabled |
| Resource-name DNS AAAA record | Disabled |
| Automatic recovery | `disabled` |

The AMI determines the operating system and root device. The configuration references an existing subnet, security group, and key pair; it does not create them.

Root disk size can be overridden per machine. The other settings in this file are currently shared. Changing a shared setting affects the desired configuration of all enabled machines, so review the plan before applying.

## 3. ec2-compute.variables.tf

This file declares the `machines` variable as a map of objects. Each map key is a machine name, and each object contains its settings.

| Input | Type | Default when omitted | Meaning |
|---|---|---|---|
| `is_true` | Boolean | `true` | Include this machine in Terraform's desired resources |
| `is_spot` | Boolean | `false` | Use Spot when true; On-Demand when false |
| `instance_type` | String | `t3a.large` | EC2 instance type |
| `root_volume_size` | Number | Shared size from `ec2-compute.defaults.tf` | Root disk size in GiB |

If the entire `machines` input is omitted, its default is an empty map: `{}`.

`root_volume_size` initially defaults to `null`. The compute file replaces that with the shared root disk size. Validation requires an explicitly supplied size to be a positive whole number. AWS and the AMI can impose additional constraints.

Although these inputs are optional in the schema, this project's `terraform.tfvars` lists all four explicitly for each machine so they are easy to review and edit.

## 4. terraform.tfvars

This file supplies actual values for `var.machines`. Terraform loads `terraform.tfvars` automatically when running in this directory.

The `.tf` variable file defines which inputs are supported and their defaults. The `.tfvars` file supplies the values to use instead of those defaults.

Current configuration at the time this guide was written:

| Machine | Enabled | Spot | Instance type | Root disk |
|---|---|---|---|---|
| `dev-server-1` | `false` | `true` | `t3a.large` | 24 GiB |
| `dev-server-2` | `false` | `true` | `t3a.large` | 24 GiB |
| `gpu-worker-1` | `false` | `true` | `g4dn.xlarge` | 128 GiB |

All three entries are disabled in the configuration. This table describes the file, not a live check of AWS.

Example machine entry:

```hcl
"gpu-worker-1" = {
  is_true          = false
  is_spot          = true
  instance_type    = "g4dn.xlarge"
  root_volume_size = 128
}
```

The map key becomes both the instance's `Name` tag and part of its Terraform address, for example:

```text
aws_instance.machine["gpu-worker-1"]
```

## 5. ec2-compute.tf

This file connects the shared settings and machine inputs to the AWS resource.

For each machine, it:

1. Uses the provided root disk size, or falls back to the shared size when omitted.
2. Includes the machine only if `is_true` is true.
3. Reads the selected subnet when at least one machine is enabled.
4. Creates one `aws_instance.machine` resource per enabled map entry using `for_each`.
5. Applies shared settings and the individual instance type and disk size.
6. Adds Spot market options only when `is_spot` is true.
7. Adds CPU credit settings only for the `t2`, `t3`, `t3a`, and `t4g` families.
8. Sets the `Name` tag from the map key.
9. Checks that the subnet belongs to the configured VPC.

The subnet check does not independently verify the security group's VPC. GPU driver installation and AMI compatibility checks are not implemented by these files.

## 6. outputs.tf

| Output | Contents |
|---|---|
| `machines` | Map of managed machine names to instance ID, public IP, private IP, and Spot flag |
| `aws_account_id` | Account ID of the active AWS identity |
| `aws_identity` | ARN of the active AWS identity |
| `aws_user_id` | User or role-session identity ID |
| `aws_summary` | Account ID, ARN, and user ID together |
| `hello_aws` | Greeting containing the account ID |
| `account_console_url` | Account-specific AWS console sign-in URL |

Read saved outputs with:

```sh
terraform output
terraform output machines
```

Outputs come from Terraform state. Editing configuration alone does not update saved output values.

## Adding another machine

Add another entry inside the existing `machines = { ... }` map in `terraform.tfvars`:

```hcl
"dev-server-3" = {
  is_true          = false
  is_spot          = true
  instance_type    = "t3a.large"
  root_volume_size = 24
}
```

Set `is_true = true` when you want it included in the next plan. No new `.tf` file or copied resource block is required.

## Lifecycle behavior

- `is_true = true` includes a machine in the desired configuration. It does not detect or adopt an existing AWS instance by name.
- Setting a managed machine to `is_true = false`, or deleting its map entry, plans its destruction. This is not a stop/start switch.
- Renaming a managed machine's map key changes its Terraform address. Use a deliberate `moved` block or state move to preserve its identity rather than recreate it.
- Changing an instance type, AMI, disk size, or purchasing option can affect the running instance. The plan shows the action Terraform proposes.
- A larger root disk is not a reversible size toggle: do not assume reducing the number later can shrink an existing EBS volume.

## Usual workflow

Run these commands inside the Terraform container, from `/workspace`:

```sh
terraform init
terraform fmt
terraform validate
terraform plan -out=compute.tfplan
```

Review the saved plan, including additions, changes, and destructions. To apply that plan and inspect the saved outputs:

```sh
terraform apply "compute.tfplan"
terraform output machines
```

Generate a fresh plan after editing the configuration. Applying an older saved plan uses that plan's recorded configuration and values.

See [terraform-reference.md](terraform-reference.md) for the original Docker and Terraform command walkthrough.

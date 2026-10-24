# EC2 machine manager

Manage six AWS EC2 machines with Terraform. Choose a `system_state` preset to run, stop, or destroy groups of machines, or use `custom` to control machines individually.

**The current default is `all_destroyed`. Applying it destroys managed machines and their root disks.** Machines outside a selected preset are also destroyed unless that preset explicitly says `stopped`.

## Machines

| Name | Group | Instance type | Root disk | Default purchase option |
| --- | --- | --- | --- | --- |
| `test_002` | Basic | `t3a.large` | 24 GiB | Spot |
| `test_003` | Basic | `t3a.large` | 24 GiB | Spot |
| `test_004` | Basic | `t3a.large` | 24 GiB | Spot |
| `gpu_test_001` | GPU | `g4dn.xlarge` | 128 GiB | On-Demand |
| `gpu_test_002` | GPU | `g4dn.xlarge` | 128 GiB | On-Demand |
| `gpu_test_003` | GPU | `g4dn.xlarge` | 128 GiB | Spot |

Shared configuration:

- Account: `900065085340`; region: `us-east-1`.
- Subnet: `subnet-c63cabe7` in `us-east-1a`, VPC `vpc-72db7f0f`.
- Security group: `sg-0849de47513ee506e`; SSH key pair: `main`.
- AMI: `ami-02a9cdd4e98bc3c78`; root device and source snapshot come from the AMI.
- Public IPv4 requested; root disks use gp3 with 3,000 IOPS and 125 MiB/s throughput.

The key pair, subnet, security group, and AMI must already exist and be accessible. This project does not create them.

## Setup

Run the following PowerShell commands from this directory. Docker Desktop with Docker Compose must be running. Terraform runs inside the container defined in `docker-compose.yaml`.

Create the credentials file only if it does not already exist:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Fill in `.env` with your AWS credentials. Temporary credentials also need `SIVA_AWS_SESSION_TOKEN`. Do not commit credentials; `.env` is ignored by Git. The provider explicitly uses `us-east-1` regardless of the environment's default region.

```powershell
docker compose up -d
docker compose exec -T terraform terraform init
```

After changing credentials in `.env`, recreate the service to load them:

```powershell
docker compose up -d --force-recreate
```

## Choose a preset

Create or edit `terraform.tfvars`. For example:

```hcl
system_state = "all_basic_on"
```

This runs the three basic machines and destroys the three GPU machines on apply.

| Preset | Basic machines | GPU machines |
| --- | --- | --- |
| `all_basic_on` | Running | Destroyed |
| `all_gpu_on` | Destroyed | Running |
| `basic_on_gpu_stopped` | Running | Stopped |
| `gpu_on_basic_stopped` | Stopped | Running |
| `all_on` | Running | Running |
| `all_stopped` | Stopped | Stopped |
| `all_destroyed` / `all_off` | Destroyed | Destroyed |

`all_spot_on` runs machines with `is_spot = true` and destroys the others. With the supplied purchase settings this selects four machines, not three.

See [SYSTEM_STATE.md](SYSTEM_STATE.md) for every supported preset and its exact behavior.

## Control machines individually

Use `system_state = "custom"` to activate each machine's `is_true` control:

```hcl
system_state = "custom"

machines = {
  test_002     = { is_true = true,  is_spot = true }
  test_003     = { is_true = false, is_spot = true }
  test_004     = { is_true = false, is_spot = true }
  gpu_test_001 = { is_true = false, is_spot = false }
  gpu_test_002 = { is_true = false, is_spot = false }
  gpu_test_003 = { is_true = false, is_spot = true }
}
```

This example runs only `test_002` and destroys the other five machines.

| Control | Meaning |
| --- | --- |
| `is_true = true` | In custom mode, create/start this machine |
| `is_true = false` | In custom mode, destroy this machine |
| `is_spot = true` | Use Spot in every mode |
| `is_spot = false` | Use On-Demand in every mode |

Each machine's default `is_spot` is defined in `machine_catalog.tf`, beside its instance type and disk size. In `machines`, omit `is_spot` (or set it to `null`) to inherit that default. Set it explicitly to `true` or `false` to override it. For example, `test_002 = { is_true = true }` uses its catalog Spot default.

Named presets override `is_true`, but keep the resolved `is_spot`. An override of `machines` replaces the entire map; omitted machines are no longer managed by the resource map and will be destroyed if present in state.

The full [terraform.tfvars.example](terraform.tfvars.example) enables **all six machines** in custom mode. Review it before copying. Without a `machines` override, the current defaults enable `gpu_test_003`, `test_003`, and `test_004` in custom mode. Those switches do not override the default `all_destroyed` preset.

## Preview and apply

```powershell
docker compose exec -T terraform terraform validate
docker compose exec -T terraform terraform plan -out=changes.tfplan
docker compose exec -T terraform terraform show -no-color changes.tfplan
```

Review the machine names and all create, update, replace, and destroy actions. When the saved plan matches your intent:

```powershell
docker compose exec -T terraform terraform apply changes.tfplan
docker compose exec -T terraform terraform output
```

Applying a saved plan executes it without another approval prompt. If you edit configuration after saving the plan, generate and review a new plan before applying. Use the `.tfplan` extension so the existing Git ignore rule excludes the saved plan.

Changing a preset alone does not change AWS. Terraform enforces it when applied, not continuously. Keep the local Terraform state: it records the resources managed by this project. Stopping the Docker container does not stop EC2 instances.

## Running, stopped, and destroyed

| Desired state | Instance | Root disk |
| --- | --- | --- |
| Running | Exists and runs | Retained |
| Stopped | Exists but stops | Retained; storage charges continue |
| Destroyed | Terminated | Deleted by the configured termination policy |

A stopped preset is a desired state: if its selected machine does not exist, Terraform creates it and then stops it. Launch quotas and capacity still apply.

Stop protection is disabled. Spot machines use persistent requests with stop interruption behavior. Migrating an existing one-time Spot instance can require replacement. Spot restart depends on available capacity. Check the plan before applying purchase-option changes.

## Code structure

| File | What to change or read there |
| --- | --- |
| [variables.tf](variables.tf) | Input defaults, per-machine controls, accepted preset names |
| [machine_catalog.tf](machine_catalog.tf) | Machine types, disk sizes, explicit groups, shared AWS settings |
| [state_presets.tf](state_presets.tf) | Preset outcome tables and desired-state selection |
| [instances.tf](instances.tf) | EC2 configuration and running/stopped enforcement |
| [outputs.tf](outputs.tf) | Desired states, instance IDs, IP addresses, account identity |
| [main.tf](main.tf) | AWS provider, region, account restriction |
| [docker-compose.yaml](docker-compose.yaml) | Terraform container and credential mapping |
| [tests/system_state.tftest.hcl](tests/system_state.tftest.hcl) | Mocked preset behavior tests |

Read the flow as:

```text
Inputs + machine catalog
          |
          v
Preset outcome table
          |
          v
Machine name -> running / stopped / destroyed
          |
          v
EC2 resources + power-state resources
```

To add a machine, add its catalog entry, controls, and allowed name; update the example and tests. Group presets include machines based on their explicit group. To add a preset, add its outcome-table row and allowed name, then document and test it. The AWS resource code normally stays unchanged.

## Local checks

```powershell
docker compose exec -T terraform terraform fmt -check -recursive
docker compose exec -T terraform terraform validate
docker compose exec -T terraform terraform test -no-color
```

The tests use a mocked AWS provider and do not provision AWS resources. They check selection rules, explicit stopping, purchase overrides, custom controls, and invalid presets. They do not verify live AWS permissions, quotas, AMI availability, or Spot capacity.

[SYSTEM_STATE.md](SYSTEM_STATE.md) contains the full preset reference. [TERRAFORM_COMMANDS.md](TERRAFORM_COMMANDS.md) contains historical session notes; use this README for the current workflow.

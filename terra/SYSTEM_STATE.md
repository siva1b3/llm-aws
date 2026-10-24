# System state presets

Set `system_state` in `terraform.tfvars`, then run `terraform plan` and review before applying.

```hcl
system_state = "all_basic_on"
```

This creates/starts the three basic machines and destroys the GPU machines.
Machines outside a preset's running or explicitly stopped groups are destroyed, including their root disks.

| Preset | Running | Stopped | Destroyed |
| --- | --- | --- | --- |
| `custom` | Machines with `is_true = true` | None | Machines with `is_true = false` |
| `all_on` | All | None | None |
| `all_off`, `all_destroyed` | None | None | All |
| `all_stopped` | None | All | None |
| `all_basic_on` | Basic | None | GPU |
| `all_gpu_on` | GPU | None | Basic |
| `all_spot_on` | Spot | None | On-Demand |
| `all_on_demand_on` | On-Demand | None | Spot |
| `all_basic_stopped` | None | Basic | GPU |
| `all_gpu_stopped` | None | GPU | Basic |
| `all_spot_stopped` | None | Spot | On-Demand |
| `all_on_demand_stopped` | None | On-Demand | Spot |
| `basic_on_gpu_stopped` | Basic | GPU | None |
| `gpu_on_basic_stopped` | GPU | Basic | None |
| `spot_on_on_demand_stopped` | Spot | On-Demand | None |
| `on_demand_on_spot_stopped` | On-Demand | Spot | None |

Basic machines are `test_002`, `test_003`, `test_004`. GPU machines are `gpu_test_001`, `gpu_test_002`, `gpu_test_003`.
Spot membership follows the resolved `is_spot`, so currently includes four machines. Its default lives in `machine_catalog.tf`; an explicit boolean in `machines` overrides it. Omitted or null values inherit the catalog default. Presets override `is_true` but never change the resolved `is_spot`.
Only machine names present in the `machines` map are managed; overriding that map replaces it in full.
The current default is `all_destroyed`. Choose `custom` to use the per-machine selections. Values in `terraform.tfvars` override defaults in `variables.tf`.

## Where to read or change the code

| File | Responsibility |
| --- | --- |
| `variables.tf` | User controls and valid preset names |
| `machine_catalog.tf` | Fixed machine specifications, explicit basic/GPU groups, and shared AWS settings |
| `state_presets.tf` | Readable outcome tables and translation into one desired state per machine |
| `instances.tf` | Create retained instances and enforce running/stopped state |
| `outputs.tf` | Desired states, instance addresses, and account identity |
| `main.tf` | Provider and AWS account configuration |

Read the code in this order: controls → catalog → preset tables → desired states → AWS resources.
The preset tables directly specify running, stopped, or destroyed for each group. Purchase-based presets use `is_spot`; `custom` uses `is_true`. All produce the same machine-name-to-state map.

For example, `basic_on_gpu_stopped` resolves to three basic machines running and three GPU machines stopped. All six are retained. `all_basic_on` retains only the three basic machines; the GPU instances are absent from the resource map and are destroyed on apply if they exist.

To add a preset, add a row to the appropriate table in `state_presets.tf`, allow its name in `variables.tf`, and add its documented behavior and test. AWS resource code does not need to change.
To add a machine, add its specifications in `machine_catalog.tf` and its two controls and allowed name in `variables.tf`. Group presets automatically include it.

File separation does not change Terraform resource addresses. Existing instances continue to use `aws_instance.machine["name"]` and `aws_ec2_instance_state.machine["name"]`.

Stopped presets are desired states, not commands to stop only existing instances. If a selected stopped machine does not exist, Terraform launches it and then stops it. Launch quotas and capacity still apply. EBS storage remains billable while stopped.

Stop protection is disabled and OS-initiated shutdown is configured to stop. Spot requests use persistent requests with stop interruption behavior to support stop/start. Changing existing one-time Spot requests to these settings can replace the instances; review the plan before applying. Spot restart requires capacity, and AWS can resume instances it stopped for an interruption when capacity returns.

Terraform enforces these states when applied; it is not a continuous power-state controller.

References: [AWS Spot management](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-instances-request.html), [Terraform instance state](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/ec2_instance_state).

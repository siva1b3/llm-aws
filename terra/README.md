# Terraform EC2 learning project

Run Terraform through Docker Compose to manage EC2 Spot instances in `us-east-1`. No local Terraform or AWS CLI installation is needed. Docker must be running Linux containers.

## Files and how they fit together

All `.tf` files in this folder form **one configuration**, using one state. Terraform loads them together; it does not run a particular file based on its name. Subfolders are not loaded automatically.

| File | Purpose |
| --- | --- |
| `docker-compose.yaml` | Runs `hashicorp/terraform:1.15.9`. |
| `main.tf` | AWS provider and account ID/alias lookups. |
| `ec2-single.tf` | Up to one `t3a.large` instance. |
| `ec2-small.tf` | Up to one `t3.micro` instance. |
| `ec2-three.tf` | Either zero or three `t3a.large` instances. |
| `variables.tf` | Declares the flags and their fallback defaults. |
| `terraform.tfvars` | Local flag values automatically loaded by Terraform. |
| `terraform.tfvars.example` | Template only; Terraform does not load it. |
| `.env` | Local AWS credentials passed into the container. |
| `.env.example` | Credential placeholders safe to share. |
| `.terraform/` | Downloaded providers and initialization files. |
| `.terraform.lock.hcl` | Selected provider version and verification hashes. |
| `terraform.tfstate` | Local record of managed resource identities and outputs. |

## Docker and credentials

Compose mounts this Windows folder into `/workspace` and runs Terraform there. It loads `.env` using `env_file`:

```dotenv
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_DEFAULT_REGION=us-east-1
# AWS_SESSION_TOKEN=your_session_token
```

Enter real values only in your local `.env`. Temporary credentials require a session token. The explicit `region` in `main.tf` takes precedence over the environment's default region.

Never commit actual credentials. The folder mount also makes `.env` accessible inside the container.

Run commands from:

```cmd
cd C:\main\llm-aws\terra
```

In `docker compose run --rm terraform plan`, `terraform` selects the Compose service and `plan` is passed to Terraform. `--rm` removes the container after it exits, but files in the mounted folder remain on Windows.

## Choose the instances

Edit **`terraform.tfvars`**. For only the single `t3a.large`:

```hcl
enable_single_instances = true
enable_small_instance   = false
enable_three_instances  = false
```

For only the small `t3.micro`:

```hcl
enable_single_instances = false
enable_small_instance   = true
enable_three_instances  = false
```

For only three `t3a.large` instances:

```hcl
enable_single_instances = false
enable_small_instance   = false
enable_three_instances  = true
```

All true requests five instances. All false requests zero instances. Setting an enabled group to false and applying proposes deleting that group's existing managed instances, including their root disks.

**Values in `terraform.tfvars` override defaults in `variables.tf`. Editing `terraform.tfvars.example` has no effect on a run.** Command-line `-var`/`-var-file` arguments and automatically loaded variable files can also override values; the examples here use only `terraform.tfvars`.

At the time this README was recreated, the local file enabled only the small instance. Check the file for the current selection.

The flags control resource counts:

```hcl
count = var.enable_small_instance ? 1 : 0
```

This means one instance when true, zero when false. The three-instance resource uses `3 : 0`. Its instances have indexes `[0]`, `[1]`, and `[2]`.

## Initialize, plan, apply, destroy

### Initialize the folder

```cmd
docker compose run --rm terraform init
```

Installs the AWS provider and prepares state storage. With our local state setup, initialization does not need AWS keys and does not verify them. Provider downloads need internet access.

Run this for a fresh checkout, after deleting `.terraform/`, or after changes requiring initialization, such as new providers or backend settings. You do not need it for each file edit or each new container. `~> 6.0` permits AWS provider versions from 6.0 up to, but excluding, 7.0; the lock file records the selected version.

### Preview changes

```cmd
docker compose run --rm terraform plan
```

Reads the configuration, refreshes relevant AWS information, and compares it with state. It does not execute the proposed infrastructure changes.

| Plan notation | Meaning |
| --- | --- |
| `+` | Create. |
| `~` | Update in place. |
| `-` | Delete. |
| `-/+` | Replace by deleting and creating. |
| `No changes` | No changes are needed to match the configuration. |

Review the counts of additions, changes, and deletions.

### Apply changes

```cmd
docker compose run --rm terraform apply
```

Generates a fresh plan and, when confirmation is required, waits for `yes`. It then carries out the changes and updates state. Running `plan` separately is optional because this command shows its own plan. A previous unsaved plan is not the exact plan used by `apply`.

For normal work: edit the configuration or flags, review the plan, then apply. Applying unchanged configuration does not create duplicate instances.

### Delete all managed resources

```cmd
docker compose run --rm terraform destroy
```

Review the destruction plan, then confirm with `yes` if correct. This deletes resources managed in this configuration's state, not everything in your AWS account.

The code remains. A later `apply` recreates any enabled instances. Alternatively, disable a group or remove its resource block and use `plan` followed by `apply` to delete that group's managed resources.

## What state tracks

Terraform maps its resource addresses to AWS IDs, for example:

```text
aws_instance.single[0] -> i-example123
```

Changing the Name tag manually does not change the instance ID. Terraform still recognizes the instance and usually proposes restoring the tag to the value in the code. To keep a manual tag change, update the code to match.

| Situation | Expected behavior |
| --- | --- |
| Enabled in code, instance missing in AWS | Propose creating it. |
| In code and state, settings changed | Propose an update or replacement. |
| In state, removed from code or count reduced to zero | Propose deleting it if it still exists. |
| Existing AWS resource in neither code nor state | Leave it alone. |

The existing key pair and security group are referenced, not created or owned by this configuration. Destroying these instances does not delete those shared resources.

The `moved` block in `ec2-single.tf` maps the old address `aws_instance.test` to `aws_instance.single[0]`. It lets Terraform retain the existing state association through that code rename instead of treating it as an unrelated resource. Do not casually rename resource addresses without considering state migration.

Keep state while managing resources. Deleting state does not delete the real infrastructure; it removes Terraform's record of it.

## EC2 settings and Spot replacement

The definitions share the supplied AMI `ami-0f8a61b66d1accaee`, key pair `terraform-ec2-key`, and security group ID `sg-0849de47513ee506e`. They request a public IPv4 address and use a default subnet because no subnet ID is specified. That requires a compatible default VPC/subnet and security group in `us-east-1`.

Each definition requests a 60 GiB gp3 root disk with 3,000 IOPS and 125 MiB/s throughput, with encryption set to false. AWS account encryption policies may affect the effective encryption. Root disks are deleted on termination. Instance, disk, and public IPv4 usage may incur charges.

These are one-time Spot instances configured for termination on interruption. If an instance is terminated, its next normal plan refresh detects that it is missing. If its flag remains true, Terraform proposes creating a replacement. Terraform is not a continuously running recovery service; you must apply the plan. Replacement depends on Spot capacity and starts with a new instance ID, public IP, and fresh root disk.

There is no Git, Docker, Ollama installation script or Ansible configuration yet.

## Outputs and local checks

```cmd
docker compose run --rm terraform output
docker compose run --rm terraform state list
docker compose run --rm terraform fmt -check
docker compose run --rm terraform validate
```

`output` shows saved outputs, not a fresh AWS lookup. Account outputs come from read-only data sources. The IAM alias lookup requires `iam:ListAccountAliases` and may fail if no alias is configured. The alias can differ from the billing account name.

The single-instance outputs use `one(...)`, returning a value when enabled and null when absent. Null root outputs are omitted from saved output displays. The small and three-instance outputs use `[*]` to return lists, which are empty when those groups are disabled. All output names must be unique across the folder.

`validate` checks configuration consistency; it does not prove that the AMI, permissions, subnet, or Spot capacity will allow a launch.

## What belongs in Git

Commit `.tf` files, Compose configuration, `.env.example`, `terraform.tfvars.example`, `.gitignore`, this README, and `.terraform.lock.hcl`.

The current `.gitignore` excludes `.env`, `terraform.tfvars`, `.terraform/`, state files and backups, saved `*.tfplan` plans, and `*.pem`/`*.key` files. State and plans can contain sensitive data. Ignoring files does not delete them from disk or untrack files already committed.

## Generate a new SSH key pair

`key-pair.tf` optionally generates an RSA key using the TLS provider, registers its public key using the AWS provider, and saves the private key using the Local provider. These are SSH keys, not AWS API access keys.

Add these settings to your actual `terraform.tfvars` to enable creation:

```hcl
create_key_pair   = true
new_key_pair_name = "terraform-generated-key"
```

Run `docker compose run --rm terraform init` if the new providers have not yet been installed, then `plan` and `apply` as usual. This adds three managed resources for the key workflow. The plan also includes any changes requested by your EC2 flags.

After applying, the PEM appears at `keys/terraform-generated-key.pem` in this Windows project folder. No private key is printed as an output. Converting this PEM to PPK is a separate PuTTYgen step.

The private key is stored in Terraform state as well as the PEM. Both must remain private. `keys/`, PEM/PPK files, and state are ignored by Git. Linux file permissions on a Docker bind mount do not guarantee equivalent Windows ACL restrictions.

Existing EC2 definitions still use `terraform-ec2-key`. Creating this new key does not automatically change those instances. Wiring the new key into EC2 is a separate step; changing an existing instance's `key_name` can require replacement.

Setting `create_key_pair` back to false and applying, or running `destroy`, removes the Terraform-managed AWS key pair and local PEM file. Removing the AWS key pair does not remove public keys already installed inside running instances. Keep the private key securely if you still need access to those instances.

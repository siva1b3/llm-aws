# Terraform commands in Docker

Notes from the Terraform session. Run the Docker command in PowerShell and the remaining commands inside the container shell.

## 1. Attach to the container shell

```powershell
& "C:\Program Files\Docker\Docker\resources\bin\docker.exe" exec -it 344d46c8a2ba31a428811d48c6d0e7a7bad845825c78d3e51037bbf725b8935b sh
```

This opens a shell in the running container. The working directory is `/workspace`.

The container ID above is from this session. From the folder containing `docker-compose.yaml`, you can also use the Compose service name:

```powershell
docker compose exec terraform sh
```

## 2. Check the Terraform version

```sh
terraform --version
```

Output from the session:

```text
Terraform v1.15.9
on linux_amd64
```

The session also displayed an update notice reporting `1.17.0` as the latest version. This records the session output, not a separate check of the current release.

## 3. Initialize Terraform

```sh
terraform init
```

Initializes the backend and installs the providers required by the configuration.

In this session, Terraform installed `hashicorp/aws v6.70.0` and created `.terraform.lock.hcl` to record the provider selection. Commit this lock file to version control so subsequent initialization uses the same provider selection by default.

Successful result:

```text
Terraform has been successfully initialized!
```

## 4. Check the version after initialization

```sh
terraform --version
```

The output now includes the installed provider:

```text
Terraform v1.15.9
on linux_amd64
+ provider registry.terraform.io/hashicorp/aws v6.70.0
```

## 5. Run initialization again

```sh
terraform init
```

In this session, the second run reused the lock file selection and the installed provider:

```text
- Reusing previous version of hashicorp/aws from the dependency lock file
- Using previously-installed hashicorp/aws v6.70.0
```

Rerun initialization when you change modules or backend configuration. You do not need to repeat it before every plan.

## 6. Preview the changes

```sh
terraform plan
```

Terraform read `data.aws_caller_identity.current` and proposed these outputs:

```text
Changes to Outputs:
  + aws_account_id = "900065085340"
  + aws_identity   = "arn:aws:iam::900065085340:user/sivanaga"
```

This plan only added output values to Terraform state. It did not propose infrastructure changes.

This command did not save a plan file. A later `terraform apply` without a saved plan would generate a new plan.

## 7. Save the plan

```sh
terraform plan -out=siva_tfplan
```

Saves the proposed changes to `siva_tfplan` in the current directory.

```text
Saved the plan to: siva_tfplan
```

## 8. Apply the saved plan

```sh
terraform apply siva_tfplan
```

Applies the saved plan. Passing a saved plan does not prompt for interactive approval.

Result from the session:

```text
Apply complete! Resources: 0 added, 0 changed, 0 destroyed.

Outputs:

aws_account_id = "900065085340"
aws_identity = "arn:aws:iam::900065085340:user/sivanaga"
```

Terraform saved the outputs to state. No infrastructure resources were added, changed, or destroyed.

## Command reference

Run inside the container, in this order:

```sh
terraform --version
terraform init
terraform --version
terraform plan
terraform plan -out=siva_tfplan
terraform apply siva_tfplan
```

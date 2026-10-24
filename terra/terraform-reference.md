# Terraform Docker workflow reference

This records a successful Terraform initialization, plan, and apply session in the Docker container's `/workspace` directory.

## 1. Open a shell in the container

From PowerShell:

```powershell
& "C:\Program Files\Docker\Docker\resources\bin\docker.EXE" exec -it 344d46c8a2ba31a428811d48c6d0e7a7bad845825c78d3e51037bbf725b8935b sh
```

The container ID above is from the recorded session. If the container has been recreated, use `docker ps` to find its current ID or name.

Run the following Terraform commands inside the container at `/workspace`.

## 2. Check the Terraform version

```sh
terraform --version
```

Recorded output:

```text
Terraform v1.15.9
on linux_amd64

Your version of Terraform is out of date! The latest version
is 1.17.0. You can update by downloading from https://developer.hashicorp.com/terraform/install
```

The version notice above is historical output from this session, not a current version check.

## 3. Initialize Terraform

```sh
terraform init
```

Recorded output:

```text
Initializing the backend...

Initializing provider plugins...
- Finding latest version of hashicorp/aws...
- Installing hashicorp/aws v6.70.0...
- Installed hashicorp/aws v6.70.0 (signed by HashiCorp)

Terraform has created a lock file .terraform.lock.hcl to record the provider
selections it made above. Include this file in your version control repository
so that Terraform can guarantee to make the same selections by default when
you run "terraform init" in the future.

Terraform has been successfully initialized!
```

Rerun `terraform init` after changing modules or backend configuration. Keep `.terraform.lock.hcl` in version control to record provider selections.

## 4. Preview the changes

```sh
terraform plan
```

Recorded output:

```text
data.aws_caller_identity.current: Reading...
data.aws_caller_identity.current: Read complete after 1s [id=900065085340]

Changes to Outputs:
  + aws_account_id = "900065085340"
  + aws_identity   = "arn:aws:iam::900065085340:user/sivanaga"

You can apply this plan to save these new output values to the Terraform state, without changing any real
infrastructure.
```

This command did not save the plan. Terraform noted that a subsequent `terraform apply` could produce a different plan.

## 5. Save the plan

```sh
terraform plan -out=siva_tf_plan
```

Recorded output:

```text
data.aws_caller_identity.current: Reading...
data.aws_caller_identity.current: Read complete after 0s [id=900065085340]

Changes to Outputs:
  + aws_account_id = "900065085340"
  + aws_identity   = "arn:aws:iam::900065085340:user/sivanaga"

You can apply this plan to save these new output values to the Terraform state, without changing any real
infrastructure.

Saved the plan to: siva_tf_plan

To perform exactly these actions, run the following command to apply:
    terraform apply "siva_tf_plan"
```

## 6. Apply the saved plan

```sh
terraform apply "siva_tf_plan"
```

Recorded output:

```text
Apply complete! Resources: 0 added, 0 changed, 0 destroyed.

Outputs:

aws_account_id = "900065085340"
aws_identity = "arn:aws:iam::900065085340:user/sivanaga"
```

The apply saved the AWS account ID and caller identity outputs to Terraform state. No infrastructure resources were added, changed, or destroyed in this session.

## 7. View saved outputs

```sh
terraform output
```

Displays the output values saved in Terraform state after applying.

## Quick command reference

Inside the container, from `/workspace`:

```sh
terraform --version
terraform init
terraform plan
terraform plan -out=siva_tf_plan
# Review the plan output before applying it.
terraform apply "siva_tf_plan"
terraform output
```

For future changes, generate and review a fresh plan before applying. The results above describe this recorded session; future plans depend on the configuration, credentials, and state at that time.

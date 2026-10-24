# Ubuntu Development Environment Setup

Run `dev-setup.yaml` to configure the Ubuntu hosts in the `ec2` inventory group. It installs development utilities, configures Docker, and prints installation checks.

## Requirements

- Docker Desktop and Docker Compose on your Windows computer.
- An Ubuntu target with systemd and access to Ubuntu and Docker package repositories.
- A regular SSH user with sudo access; the existing inventory uses `ubuntu`.
- The correct host address in `inventory.ini` and the SSH key at `keys/main.pem`.
- Network access to the target's SSH port.

The Ansible container runs as root, but the remote SSH user must be a regular user. Administrative tasks use `become: true`; Git LFS configuration runs as the SSH user.

## Files and Execution Order

| Order | File | Task prefix | Purpose |
| --- | --- | --- | --- |
| Entry point | `dev-setup.yaml` | `UBUNTU-SETUP` | Imports the five stages below |
| 1 | `tasks/01-check-ubuntu.yaml` | `PRECHECK / PRE` | Check Ubuntu and the SSH user |
| 2 | `tasks/02-install-dev-tools.yaml` | `DEV-TOOLS / DEV` | Refresh APT, install utilities, initialize Git LFS |
| 3 | `tasks/03-install-docker.yaml` | `DOCKER-INSTALL / DKI` | Detect Docker, check conflicts, configure its repository, install packages |
| 4 | `tasks/04-configure-docker.yaml` | `DOCKER-CONFIG / DKC` | Enable Docker and configure user membership |
| 5 | `tasks/05-verify.yaml` | `VERIFY / VER` | Check tool versions and Docker service status |

Run the main playbook, not the individual task files. The `import_tasks` order controls execution; numeric filenames are for navigation. A failed task stops subsequent tasks on that host.

Each task has a unique ID:

```text
TASK [[PRECHECK | PRE002] Identify the regular SSH login user]
TASK [[DOCKER-INSTALL | DKI001] Check whether Docker is already available]
```

Search for the ID in the task files to locate a step. IDs ending in `000` identify imports in the main playbook. Blocks also have IDs, so displayed tasks can have gaps in numbering.

Multiple hosts can run concurrently, with task order preserved per host. To complete one host before starting another, add `serial: 1` at the play level:

```yaml
  hosts: ec2
  serial: 1
  become: true
```

## Run from Windows PowerShell

Open PowerShell in `C:\main\llm-aws\ansible`.

### 1. Start the Ansible runner

```powershell
docker compose up -d --build
```

### 2. Validate syntax and inspect task order

These commands do not connect to EC2:

```powershell
docker compose exec -T ansible ansible-playbook -i inventory.ini dev-setup.yaml --syntax-check
docker compose exec -T ansible ansible-playbook -i inventory.ini dev-setup.yaml --list-tasks
```

### 3. Run the setup

This command installs and configures software on the selected EC2 hosts:

```powershell
docker compose exec -T ansible ansible-playbook -i inventory.ini dev-setup.yaml
```

To select one host, append `--limit dev-server-1`.

If sudo requires a password, use an interactive prompt:

```powershell
docker compose exec ansible ansible-playbook -i inventory.ini dev-setup.yaml --ask-become-pass
```

## Run from Inside the Ansible Container

If you already have a container shell:

```bash
cd /workspace
ansible-playbook -i inventory.ini dev-setup.yaml
```

To open that shell from PowerShell:

```powershell
docker compose exec ansible sh
```

## View and Save Output

Progress appears in the terminal. Example format only; this is not an actual deployment result:

```text
TASK [[DEV-TOOLS | DEV001] Update APT package index] ****************
changed: [dev-server-1]

TASK [[DOCKER-INSTALL | DKI001] Check whether Docker is already available] ***
ok: [dev-server-1]

TASK [[VERIFY | VER002] Show verification output] ******************
ok: [dev-server-1] => ...

PLAY RECAP ********************************************************
dev-server-1 : ok=... changed=... unreachable=0 failed=0 skipped=...
```

| Status | Meaning |
| --- | --- |
| `ok` | Succeeded without reporting a change |
| `changed` | Succeeded and reported a change |
| `skipping` | A condition excluded the task |
| `failed` | Failed; inspect the task ID and error details |
| `unreachable` | Ansible could not connect to the host |

Verification prints Git, Python, Docker, Compose, and Buildx versions, plus Docker service status. Ansible does not stream every line of APT output. Append `-v` for additional detail.

To show and save output from PowerShell:

```powershell
docker compose exec -T ansible ansible-playbook -i inventory.ini dev-setup.yaml 2>&1 | Tee-Object -FilePath dev-setup.log
```

## Installation Behavior

- The complete development package list is in `tasks/02-install-dev-tools.yaml`.
- Git LFS configuration is checked and initialized for the SSH user.
- Existing Docker skips repository setup and Docker package installation. Configuration and verification still run.
- When Docker is absent, installed conflicts stop setup for review: `docker.io`, `docker-compose`, `docker-compose-v2`, `podman-docker`, `containerd`, or `runc`. No conflicting packages are automatically removed.
- New installations use Docker's official Ubuntu repository and install `docker-ce`, `docker-ce-cli`, `containerd.io`, `docker-buildx-plugin`, and `docker-compose-plugin`.
- Docker is started and enabled at boot.
- The SSH user is added to the Docker group without removing other group memberships. This group grants root-equivalent privileges.
- Repeated runs reuse installed packages and configuration. APT metadata is refreshed each run.

## Troubleshooting

| Problem | Explanation or action |
| --- | --- |
| Docker detection returns `1` or `127` | Both are accepted as Docker absent; installation proceeds after conflict checks. Other unexpected codes fail. |
| Conflicting container package | Review the existing installation before deciding whether to remove or retain it. |
| Existing Docker lacks Compose, Buildx, or its service | Review that installation. The playbook skips Docker installation when the Docker command already exists. |
| SSH unreachable | Check the host address, key, network access, and EC2 security group. |
| Sudo password required | Use the interactive `--ask-become-pass` command above. |
| Docker access denied after setup | Log out of the remote session and log back in to activate group membership. |

After fixing an error, rerun the full playbook so prerequisite facts and registered values are available.

## After Setup

Log out of the remote Ubuntu session and back in. Then run on that host:

```bash
docker run hello-world
```

The playbook does not run this container automatically.

## Validation and Check Mode

Local syntax validation and task-order listing have passed. They do not prove successful EC2 installation; a normal run provides that validation.

`--check` is a partial preview. It skips Git LFS initialization, Docker package installation, and verification. Docker configuration is also skipped when Docker is absent. Repository changes are not applied, so check mode cannot validate an entire fresh installation.

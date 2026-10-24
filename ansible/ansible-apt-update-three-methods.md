# Three Ways to Run `apt update` on EC2 Using Ansible

**Learning lab:** Ansible controller in Docker → two Ubuntu EC2 instances  
**Objective:** Refresh APT package indexes on both servers, understand three execution methods, and choose the appropriate one for future automation.

> **Key distinction:** `apt update` refreshes the local package *index*. It does **not** upgrade installed software. `apt upgrade` is a different operation.

## 1. Lab configuration

```text
Windows PC (PowerShell)
  └── Docker container: Ansible controller
         ├── inventory.ini
         ├── packages.yml
         └── SSH with the EC2 private key
                ├── dev-server-1 (52.90.40.209)
                └── dev-server-2 (3.92.68.233)
```

Your `inventory.ini` targets both instances through the `ec2` group:

```ini
[ec2]
dev-server-1 ansible_host=52.90.40.209
dev-server-2 ansible_host=3.92.68.233

[ec2:vars]
ansible_user=ubuntu
ansible_connection=ssh
ansible_ssh_private_key_file=/root/.ssh/main.pem
ansible_ssh_common_args='-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/root/.ssh/known-hosts/known_hosts'
```

**Before running package commands**, test the connection:

```bash
ansible -i inventory.ini ec2 -m ansible.builtin.ping
```

A `SUCCESS` with `"ping": "pong"` for each server confirms that Ansible connected and executed its ping module. This is **not** an ICMP/network ping: it uses SSH and the remote Python interpreter.

---

## 2. Method 1 — Run the familiar Linux command through Ansible

### Command

```bash
ansible -i inventory.ini ec2 -a "sudo apt update"
```

### What happens?

1. Ansible reads `inventory.ini` and selects members of the `ec2` group.
2. It connects to both EC2 instances over SSH (normally in parallel).
3. Its default `ansible.builtin.command` module runs `sudo` with arguments `apt update` on each instance.
4. Each instance updates its APT package lists and sends its output and exit status back to Ansible.

**Equivalent remote action:**

```bash
sudo apt update
```

### Important details

- `-a` supplies arguments to the selected module. With no `-m` specified, the default module is normally `ansible.builtin.command`.
- The command module does **not** invoke a shell by default; shell features such as pipes, `&&`, and output redirects will not work unless you explicitly use the shell module.
- `sudo` is included in the command itself, so this example depends on the remote user's sudo privileges.
- Ansible generally reports `changed: true` when the command succeeds, even if package indexes were already current. It cannot infer the command's idempotent intent.
- For automation, prefer `apt-get` over the human-oriented `apt` CLI if you must use raw commands; better still, use Ansible's APT module.

**Good for:** quick experiments and learning how Ansible runs remote commands.  
**Not ideal for:** durable, repeatable configuration management.

---

## 3. Method 2 — Use an Ansible playbook

A playbook stores automation in a YAML file, making the steps readable, reviewable, repeatable, and version-controllable.

### Example: `apt-update.yml`

```yaml
---
- name: Refresh APT package indexes on EC2
  hosts: ec2
  become: true

  tasks:
    - name: Update APT package cache
      ansible.builtin.apt:
        update_cache: true
```

### Run it

```bash
ansible-playbook -i inventory.ini apt-update.yml
```

### What happens?

1. Ansible loads the YAML playbook.
2. `hosts: ec2` selects the two instances.
3. `become: true` elevates privileges, ordinarily through `sudo`, for the tasks.
4. `ansible.builtin.apt` refreshes the package index on each server.
5. The play recap summarizes successful, changed, failed, and unreachable tasks by host.

**Good for:** reusable automation, multiple ordered tasks, code review, teamwork, and infrastructure configuration.

### Your existing `packages.yml`

You **do not need to replace it** to add more commands. It already manages more than the cache refresh:

```yaml
---
- name: Configure EC2 Ubuntu servers
  hosts: ec2
  become: true
  gather_facts: true

  tasks:
    - name: Update APT package cache
      ansible.builtin.apt:
        update_cache: true
        cache_valid_time: 3600

    - name: Install required packages
      ansible.builtin.apt:
        name:
          - git
          - curl
          - htop
        state: present
```

Run the existing playbook with:

```bash
ansible-playbook -i inventory.ini packages.yml
```

**Behavior to notice:**

- `cache_valid_time: 3600` allows Ansible to **skip refreshing** the cache if it considers the last update recent enough (less than one hour old). This is useful for repeated executions, but less suitable for a demonstration intended to force a refresh every time.
- `state: present` ensures the specified packages are installed. If they are already installed, Ansible normally makes no package changes.
- In future you can add tasks for folders, users, services, application deployments, and more. When the playbook becomes large, split tasks into separate playbooks or reusable roles.

---

## 4. Method 3 — Call the APT module directly (ad-hoc)

### Command

```bash
ansible -i inventory.ini ec2 -b -m ansible.builtin.apt -a "update_cache=yes"
```

### Arguments explained

| Part | Meaning |
|---|---|
| `ansible` | Run an ad-hoc Ansible task |
| `-i inventory.ini` | Read the inventory file |
| `ec2` | Target the inventory group containing both servers |
| `-b` | Become an elevated user (typically root via sudo) |
| `-m ansible.builtin.apt` | Choose Ansible's APT module |
| `-a "update_cache=yes"` | Give the module its parameter: refresh the APT cache |

**Equivalent intention:** perform `sudo apt update` on both machines, but via an Ansible module rather than a raw command.

### What happens?

1. Ansible connects to the selected hosts.
2. `-b` enables privilege escalation.
3. The APT module refreshes package indexes.
4. Ansible returns structured module results for each host.

**Good for:** quick package-management tasks that do not need a saved playbook.

---

## 5. Side-by-side comparison

| Characteristic | Method 1: Linux command | Method 2: Playbook | Method 3: Ad-hoc APT module |
|---|---|---|---|
| Entry point | `ansible` | `ansible-playbook` | `ansible` |
| Executes raw Linux command? | Yes, via command module | No, uses APT module | No, uses APT module |
| Uses Ansible APT module? | No | Yes | Yes |
| Sudo handling | Explicit `sudo` inside command | `become: true` | `-b` |
| Stored as reusable YAML? | No | Yes | No |
| Supports ordered multi-step configuration? | Not as a single task | Yes | One ad-hoc operation at a time |
| Reports structured module data? | Generic command results | Yes | Yes |
| Best use | One-off command testing | Standard configuration management | One-off package management |

### How the three commands relate

```text
Method 1: ansible ... -a "sudo apt update"
          └── command module → sudo apt update

Method 2: ansible-playbook ... apt-update.yml
          └── playbook task → ansible.builtin.apt(update_cache=true)

Method 3: ansible ... -b -m ansible.builtin.apt -a "update_cache=yes"
          └── direct module invocation → ansible.builtin.apt(update_cache=true)
```

**The underlying distinction:** Methods 2 and 3 use the **same APT module**, but one is defined in a YAML playbook and the other is invoked from the CLI. Method 1 invokes a Linux command rather than the APT module.

---

## 6. How to verify success

Regardless of method, check the output for **both** `dev-server-1` and `dev-server-2`.

- For **Method 1**, check each host's `rc=0` exit status and output from APT.
- For **Method 2**, check the play recap: `failed=0` and `unreachable=0` for both servers.
- For **Method 3**, check that both hosts return `SUCCESS` with no module failure.

`changed` can be `true` or `false` depending on method and whether Ansible actually refreshed the cache. **Do not use `changed: true` alone as a success criterion.**

### Common problems

| Error | Likely cause | What to inspect |
|---|---|---|
| `Host key verification failed` | SSH server key is unknown or changed | Trusted `known_hosts` entries and fingerprints |
| `UNPROTECTED PRIVATE KEY FILE` | Private key readable by others | File permissions inside the Linux container; `0600` is appropriate |
| `Permission denied (publickey)` | SSH authentication failed | Remote username, EC2 key pair, and key path |
| `Missing sudo password` / privilege escalation failure | User cannot elevate non-interactively | Remote sudo policy and Ansible `become` configuration |
| APT errors | Repository, DNS, lock, or connectivity issues | Error text on the specific EC2 host |

> **Security reminder:** Don't bypass host-key verification as a permanent solution. A key collected by `ssh-keyscan` still needs fingerprint verification through a trusted channel before being trusted.

---

## 7. What to use in a real project

**Default to Method 2 (playbooks)** for regular server management. Store the files in version control, group related tasks, and extend them as your infrastructure grows. Use **Method 3** for quick administrative package operations. Use **Method 1** for exploring commands or actions that do not have a suitable module, preferably with `become` handled by Ansible and explicit change-status handling.

### Minimal command reference

```bash
# Test connectivity
ansible -i inventory.ini ec2 -m ansible.builtin.ping

# 1: Raw command
ansible -i inventory.ini ec2 -a "sudo apt update"

# 2: YAML playbook
ansible-playbook -i inventory.ini apt-update.yml

# 3: Direct APT module
ansible -i inventory.ini ec2 -b -m ansible.builtin.apt -a "update_cache=yes"

# Existing package-installation playbook
ansible-playbook -i inventory.ini packages.yml
```

**Takeaway:** A playbook is not a different package-management engine. It is a reusable way to describe and orchestrate tasks. The APT module can be called either from a playbook or as a one-line ad-hoc command.

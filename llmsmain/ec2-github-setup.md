# EC2 Spot Instance — GitHub CLI Setup Guide

Setup workflow for a fresh Ubuntu 24.04 EC2 spot instance, with GitHub CLI authentication for cloning, pushing, and pulling repositories. Includes verification checks at each stage.

---

## Context

- **Instance:** Ubuntu 24.04 (Noble), spot lifecycle
- **Auth method:** GitHub CLI (`gh auth login`) via device flow
- **Reason:** Spot instances are short-lived; SSH key setup is wasted effort. `gh auth login` is one command per instance.
- **Workflow:** Login once → clone, push, pull normally.

---

## Step 1 — Install GitHub CLI

```bash
sudo apt install -y gh
```

### Check
```bash
gh --version
# Expected: gh version 2.x.x
```

If the package is not found, use the official install script from `cli.github.com`.

---

## Step 2 — Verify git is installed

```bash
git --version
```

If missing:
```bash
sudo apt install -y git
```

---

## Step 3 — Authenticate with GitHub

Run as the `ubuntu` user (NOT as root):

```bash
gh auth login
```

Answer the prompts:
- **Account:** GitHub.com
- **Protocol:** HTTPS
- **Authenticate Git with your GitHub credentials:** Yes
- **How to authenticate:** Login with a web browser

You will see a one-time code, e.g., `897C-5BDD`.

### What to do with the code

The EC2 server has no browser. The device flow is designed for this — the browser part runs on your **laptop**, not the server.

1. On your laptop, open: `https://github.com/login/device`
2. Enter the code shown in the EC2 terminal.
3. Approve the authorization on GitHub.
4. The EC2 terminal automatically detects the approval and completes login.

### Check
```bash
gh auth status
```

Expected output:
```
github.com
  ✓ Logged in to github.com as <your-username>
  ✓ Git operations protocol: https
  ✓ Token: gho_************************
```

> **Caveat:** If you ran `gh auth login` as `root`, the credentials are stored in `/root/.config/gh/`. When you SSH in as `ubuntu` later, those credentials won't be available. Always run `gh auth login` as the same user you'll use for git operations.

---

## Step 4 — Set git commit identity

`gh auth login` handles authentication (who can push). Git still needs the commit author identity (whose name appears in the commit log). These are separate.

```bash
git config --global user.name "siva1b3"
git config --global user.email "siva1b3@gmail.com"
```

### Check
```bash
git config --global --list
```

Expected output (at minimum):
```
user.name=your-github-username
user.email=your-github-email@example.com
credential.helper=...
```

### View your GitHub profile (to confirm correct email)
```bash
gh api user --jq '.login, .email'
```

If email is `null` (because you made it private on GitHub):
```bash
gh api user/emails
```
Use the email marked `"primary": true`.

---

## Step 7 — Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### Check
```bash
git remote -v
# Expected: origin  https://github.com/<your-username>/<your-repo>.git (fetch)
#           origin  https://github.com/<your-username>/<your-repo>.git (push)
```

---

## Step 8 — Verify push works

Make a trivial change and test the full cycle:

```bash
echo "test" >> README.md
git add README.md
git commit -m "test commit from EC2"
git push
```

If `gh auth login` configured the credential helper correctly, no password prompt appears. If you do get a prompt, run:

```bash
gh auth setup-git
```

This explicitly tells git to use `gh` as the credential helper.

### Check
```bash
git log -1
# Confirms your commit is in history with correct author name and email
```

After successful push, undo the test change if you don't want it:
```bash
git reset --hard HEAD~1
git push --force-with-lease
```

---

## Step 9 — Connect VS Code to the instance

On your **laptop**:

1. Install the **Remote - SSH** extension in VS Code.
2. Edit `~/.ssh/config` (create if it doesn't exist):

```
Host trailq
    HostName <public-ip>
    User ubuntu
    IdentityFile /path/to/main.pem
```

3. In VS Code: `Ctrl+Shift+P` → "Remote-SSH: Connect to Host" → select `trailq`.
4. Open the cloned folder via VS Code's File → Open Folder.

### Check
- VS Code's bottom-left status bar shows: `SSH: trailq`
- Terminal inside VS Code drops you in `/home/ubuntu` as user `ubuntu`.

---

## Step 10 — Install project runtime

Depends on the project. Common cases:

**Python:**
```bash
sudo apt install -y python3-pip python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Node.js (via nvm):**
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
source ~/.bashrc
nvm install --lts
```

**Docker:**
```bash
sudo apt install -y docker.io
sudo usermod -aG docker ubuntu
# Log out and back in for group change to take effect
```

---

## Operational reminders for spot instances

1. **Spot interruption.** AWS can reclaim the instance with 2-minute notice. Always:
   - Commit and push changes frequently.
   - Treat the instance disk as throwaway storage.
   - Do not store unique data only on the instance.

2. **Public IP changes on stop/start.** No Elastic IP attached. Stopping the instance and starting it later assigns a new public IP, which breaks your VS Code SSH config.

3. **Memory pressure on `t3a.small`.** 2 GB RAM is tight. VS Code Remote server alone uses 300–700 MB. For ML workloads, this size is insufficient. Watch with:
   ```bash
   free -h
   htop
   ```

4. **Re-login per instance.** Every fresh spot instance starts with no credentials. You'll repeat Steps 1–6 each time. This is the cost of using spot.

5. **Token cleanup.** Periodically revoke old tokens at: GitHub → Settings → Applications → Authorized OAuth Apps. Not urgent, but good hygiene.

---

## Quick verification checklist

Run these in order on a fresh instance to confirm everything is working:

```bash
whoami                          # ubuntu
git --version                   # git version 2.x
gh --version                    # gh version 2.x
gh auth status                  # Logged in to github.com as <username>
git config --global user.name   # <username>
git config --global user.email  # <email>
gh api user --jq '.login'       # <username>
```

If all of these return expected values, you are ready to clone, push, and pull.

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `Permission denied (publickey)` on SSH | Wrong key file or wrong user | Use `ubuntu`, not `ec2-user`. Check key path and `chmod 400`. |
| `gh auth login` opens no browser | Server has no GUI (expected) | Open the device URL on your laptop instead. |
| `git push` asks for password | Credential helper not configured | Run `gh auth setup-git`. |
| `gh auth status` shows logged out as `ubuntu` but logged in as `root` | Auth was done as root | Re-run `gh auth login` as `ubuntu`. |
| `error: key does not contain a section` | Missing `--` before flag | Use `git config --global --list`, not `--global list`. |
| Commits show author `ubuntu@ip-xxx` | `user.name` / `user.email` not set | Run `git config --global user.name/email` commands. |

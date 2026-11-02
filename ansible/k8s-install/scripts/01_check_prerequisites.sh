#!/usr/bin/env bash
# Run with bash. Stop if a command fails; do not source this file.
set -euo pipefail

# Authenticate normally from a terminal before making system changes.
sudo -v

# This installer must not modify an initialized or joined node.
for CONFIG_FILE in /etc/kubernetes/admin.conf /etc/kubernetes/kubelet.conf /var/lib/kubelet/config.yaml /etc/kubernetes/manifests/kube-apiserver.yaml; do
  if sudo test -e "$CONFIG_FILE"; then
    echo "[ERROR] Cluster configuration exists: $CONFIG_FILE"
    echo "[INFO] Installation stopped. Use verification only."
    exit 1
  fi
done

echo "[01] Checking host prerequisites"

# This file is the only OS metadata file read as shell variables.
. /etc/os-release
if [ "$ID" != "ubuntu" ] || [ "$VERSION_ID" != "24.04" ]; then
  echo "[ERROR] These scripts target Ubuntu 24.04."
  exit 1
fi

CPU_ARCHITECTURE=$(uname -m)
CPU_COUNT=$(nproc)
echo "[INFO] OS: $PRETTY_NAME"
echo "[INFO] Architecture: $CPU_ARCHITECTURE"
echo "[INFO] CPUs: $CPU_COUNT"
free -h
df -h /

if [ "$CPU_ARCHITECTURE" != "x86_64" ]; then
  echo "[ERROR] These scripts download amd64 binaries only."
  exit 1
fi

if [ "$CPU_COUNT" -lt 2 ]; then
  echo "[ERROR] At least two CPUs are required for this setup."
  exit 1
fi

ACTIVE_SWAP=$(swapon --show --noheadings)
if [ -n "$ACTIVE_SWAP" ]; then
  echo "[ERROR] Active swap found. Disable it before installing."
  echo "$ACTIVE_SWAP"
  exit 1
fi

# Read fstab columns; ignore comments and blank lines.
while read -r DEVICE MOUNT_POINT FILESYSTEM OPTIONS REST; do
  if [ "$FILESYSTEM" = "swap" ] && [[ "$DEVICE" != \#* ]]; then
    echo "[ERROR] Swap is configured in /etc/fstab. Resolve it first."
    exit 1
  fi
done < /etc/fstab

if ! command -v systemctl >/dev/null; then
  echo "[ERROR] systemd tools are required."
  exit 1
fi

# Avoid mixing APT-managed runtimes with the upstream binary installation.
for PACKAGE in docker.io docker-ce containerd containerd.io runc; do
  if dpkg-query -W -f='${Status}\n' "$PACKAGE" 2>/dev/null | grep -qx 'install ok installed'; then
    echo "[ERROR] Conflicting runtime package installed: $PACKAGE"
    exit 1
  fi
done

for TOOL in git curl wget gpg unzip zip tar jq make gcc g++ openssl; do
  if command -v "$TOOL" >/dev/null; then
    echo "[OK] $TOOL is available."
  else
    echo "[MISSING] $TOOL will be installed in step 02."
  fi
done

echo "[PASS] Host prerequisite checks passed."

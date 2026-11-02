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

echo "[03] Preparing the host"

ACTIVE_SWAP=$(swapon --show --noheadings)
if [ -n "$ACTIVE_SWAP" ]; then
  echo "[ERROR] Active swap found. Stop and resolve it first."
  exit 1
fi

sudo modprobe overlay
echo overlay | sudo tee /etc/modules-load.d/containerd.conf

cat <<'EOF' | sudo tee /etc/sysctl.d/99-kubernetes.conf
net.ipv4.ip_forward = 1
EOF

sudo sysctl --load=/etc/sysctl.d/99-kubernetes.conf
grep -w overlay /proc/filesystems
sysctl net.ipv4.ip_forward

echo "[INFO] CNI-specific kernel and firewall settings will be handled later."
echo "[PASS] Host preparation completed."

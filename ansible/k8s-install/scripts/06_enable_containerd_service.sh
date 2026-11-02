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

echo "[06] Enabling the containerd service"

CONTAINERD_VERSION="2.3.7"
SERVICE_FILE=$(mktemp)
trap 'rm -f -- "$SERVICE_FILE"' EXIT

curl -fSL --retry 3 -o "$SERVICE_FILE" \
  "https://raw.githubusercontent.com/containerd/containerd/v$CONTAINERD_VERSION/containerd.service"

EXISTING_SERVICE=$(systemctl show containerd -p FragmentPath --value)
if [ -n "$EXISTING_SERVICE" ]; then
  if sudo cmp -s "$SERVICE_FILE" "$EXISTING_SERVICE"; then
    echo "[INFO] Existing service definition matches; preserving it."
  else
    echo "[ERROR] Existing service definition differs: $EXISTING_SERVICE"
    echo "[INFO] Review it manually; this script will not overwrite it."
    exit 1
  fi
else
  sudo install -D -m 0644 "$SERVICE_FILE" /usr/local/lib/systemd/system/containerd.service
fi

sudo systemctl daemon-reload
sudo systemctl enable --now containerd
sudo systemctl is-active containerd
sudo systemctl is-enabled containerd
sudo /usr/local/bin/ctr version

echo "[INFO] An already running containerd service was not restarted."
echo "[PASS] Containerd service is running and enabled."

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

echo "[08] Installing CRI diagnostic tools"

# The repository must already be configured by step 07.
if CRI_VERSION=$(dpkg-query -W -f='${Version}' cri-tools 2>/dev/null); then
  if [[ "$CRI_VERSION" != 1.37.* ]]; then
    echo "[ERROR] Existing cri-tools version differs: $CRI_VERSION"
    exit 1
  fi
else
  CRI_VERSION=$(apt-cache madison cri-tools |
    awk '$3 ~ /^1[.]37[.]/ {print $3}' | sort -Vr | sed -n '1p')
fi

if [ -z "$CRI_VERSION" ]; then
  echo "[ERROR] No cri-tools 1.37 package found. Run step 07 first."
  exit 1
fi

echo "[INFO] Selected cri-tools version: $CRI_VERSION"
sudo apt-get install -y "cri-tools=$CRI_VERSION"
sudo apt-mark hold cri-tools

command -v crictl
crictl --version
echo "[PASS] CRI tools installed. Verification uses explicit endpoint flags."

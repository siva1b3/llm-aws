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

echo "[07] Installing Kubernetes components"

KUBERNETES_MINOR="1.37"
TEMP_DIRECTORY=$(mktemp -d)
trap 'rm -rf -- "$TEMP_DIRECTORY"' EXIT

sudo install -d -m 0755 /etc/apt/keyrings
curl -fsSL --retry 3 \
  "https://pkgs.k8s.io/core:/stable:/v$KUBERNETES_MINOR/deb/Release.key" \
  -o "$TEMP_DIRECTORY/Release.key"

gpg --batch --yes --dearmor -o "$TEMP_DIRECTORY/key.gpg" "$TEMP_DIRECTORY/Release.key"
sudo install -m 0644 "$TEMP_DIRECTORY/key.gpg" /etc/apt/keyrings/kubernetes-apt-keyring.gpg

echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v$KUBERNETES_MINOR/deb/ /" |
  sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update

# Preserve an installed matching patch; do not silently upgrade held packages.
KUBERNETES_VERSION=""
for PACKAGE in kubeadm kubelet kubectl; do
  if INSTALLED_VERSION=$(dpkg-query -W -f='${Version}' "$PACKAGE" 2>/dev/null); then
    if [[ "$INSTALLED_VERSION" != "$KUBERNETES_MINOR".* ]]; then
      echo "[ERROR] $PACKAGE has another minor version: $INSTALLED_VERSION"
      exit 1
    fi
    if [ -n "$KUBERNETES_VERSION" ] && [ "$KUBERNETES_VERSION" != "$INSTALLED_VERSION" ]; then
      echo "[ERROR] Installed Kubernetes package versions differ."
      exit 1
    fi
    KUBERNETES_VERSION="$INSTALLED_VERSION"
  fi
done

if [ -z "$KUBERNETES_VERSION" ]; then
  # madison column 3 contains package versions. Select the newest 1.37 patch.
  KUBERNETES_VERSION=$(apt-cache madison kubeadm |
    awk '$3 ~ /^1[.]37[.]/ {print $3}' | sort -Vr | sed -n '1p')
fi

if [ -z "$KUBERNETES_VERSION" ]; then
  echo "[ERROR] No Kubernetes 1.37 package version is available."
  exit 1
fi

echo "[INFO] Selected version: $KUBERNETES_VERSION"
sudo apt-get install -y \
  "kubeadm=$KUBERNETES_VERSION" \
  "kubelet=$KUBERNETES_VERSION" \
  "kubectl=$KUBERNETES_VERSION"
sudo apt-mark hold kubeadm kubelet kubectl

# Kubeadm will generate kubelet configuration during init or join.
sudo systemctl stop kubelet
sudo systemctl enable kubelet

kubeadm version -o short
kubelet --version
kubectl version --client
echo "[PASS] Kubernetes installed; kubelet enabled but intentionally stopped."

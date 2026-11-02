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

echo "[04] Installing containerd and runc"

CONTAINERD_VERSION="2.3.7"
RUNC_VERSION="1.5.2"
INSTALL_CONTAINERD=true
INSTALL_RUNC=true

if [ "$(uname -m)" != "x86_64" ]; then
  echo "[ERROR] These downloads require x86_64."
  exit 1
fi

if command -v containerd >/dev/null; then
  CONTAINERD_PATH=$(command -v containerd)
  if [ "$CONTAINERD_PATH" != "/usr/local/bin/containerd" ]; then
    echo "[ERROR] Unexpected containerd location: $CONTAINERD_PATH"
    exit 1
  fi
  if containerd --version | grep -Fq " v$CONTAINERD_VERSION "; then
    INSTALL_CONTAINERD=false
    echo "[INFO] Matching containerd already installed; preserving it."
  else
    echo "[ERROR] Different containerd version found. No automatic upgrade."
    exit 1
  fi
fi

if command -v runc >/dev/null; then
  RUNC_PATH=$(command -v runc)
  if [ "$RUNC_PATH" != "/usr/local/sbin/runc" ]; then
    echo "[ERROR] Unexpected runc location: $RUNC_PATH"
    exit 1
  fi
  if runc --version | grep -Fxq "runc version $RUNC_VERSION"; then
    INSTALL_RUNC=false
    echo "[INFO] Matching runc already installed; preserving it."
  else
    echo "[ERROR] Different runc version found. No automatic upgrade."
    exit 1
  fi
fi

if [ "$INSTALL_CONTAINERD" = false ] && [ "$INSTALL_RUNC" = false ]; then
  echo "[PASS] Both requested runtime versions are installed."
  exit 0
fi

if systemctl is-active --quiet containerd; then
  echo "[ERROR] Stop and review: containerd is running but runtime binaries need installation."
  exit 1
fi

DOWNLOAD_DIRECTORY=$(mktemp -d /tmp/k8s-runtime.XXXXXX)
# Remove only this script's temporary downloads when it exits.
trap 'rm -rf -- "$DOWNLOAD_DIRECTORY"' EXIT
cd "$DOWNLOAD_DIRECTORY"

if [ "$INSTALL_CONTAINERD" = true ]; then
  ARCHIVE="containerd-$CONTAINERD_VERSION-linux-amd64.tar.gz"
  DOWNLOAD_URL="https://github.com/containerd/containerd/releases/download/v$CONTAINERD_VERSION"
  curl -fSL --retry 3 -o "$ARCHIVE" "$DOWNLOAD_URL/$ARCHIVE"
  curl -fSL --retry 3 -o "$ARCHIVE.sha256sum" "$DOWNLOAD_URL/$ARCHIVE.sha256sum"
  sha256sum --check "$ARCHIVE.sha256sum"
fi

if [ "$INSTALL_RUNC" = true ]; then
  DOWNLOAD_URL="https://github.com/opencontainers/runc/releases/download/v$RUNC_VERSION"
  curl -fSL --retry 3 -o runc.amd64 "$DOWNLOAD_URL/runc.amd64"
  curl -fSL --retry 3 -o runc.sha256sum "$DOWNLOAD_URL/runc.sha256sum"
  grep -E '[[:space:]]+\*?runc[.]amd64$' runc.sha256sum > runc-amd64.sha256sum
  sha256sum --check runc-amd64.sha256sum
fi

# Install only after all requested downloads pass checksum verification.
if [ "$INSTALL_CONTAINERD" = true ]; then
  sudo tar -C /usr/local -xzf "$ARCHIVE"
fi
if [ "$INSTALL_RUNC" = true ]; then
  sudo install -D -m 0755 runc.amd64 /usr/local/sbin/runc
fi

/usr/local/bin/containerd --version
/usr/local/sbin/runc --version
echo "[PASS] Runtime binaries installed."

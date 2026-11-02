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

echo "[05] Configuring containerd"

sudo mkdir -p /etc/containerd
TEMP_DIRECTORY=$(mktemp -d)
trap 'rm -rf -- "$TEMP_DIRECTORY"' EXIT

if sudo test -f /etc/containerd/config.toml; then
  echo "[INFO] Preserving existing containerd configuration."
else
  /usr/local/bin/containerd config default > "$TEMP_DIRECTORY/config.toml"
  sed -i 's/SystemdCgroup = false/SystemdCgroup = true/g' "$TEMP_DIRECTORY/config.toml"
  sudo install -m 0644 "$TEMP_DIRECTORY/config.toml" /etc/containerd/config.toml
fi

# Parse the effective TOML rather than trusting a text match in any section.
sudo /usr/local/bin/containerd --config /etc/containerd/config.toml config dump > "$TEMP_DIRECTORY/effective.toml"

python3 - "$TEMP_DIRECTORY/effective.toml" <<'PYTHON'
import sys
import tomllib

with open(sys.argv[1], "rb") as config_file:
    config = tomllib.load(config_file)

runtime = config.get("plugins", {}).get("io.containerd.cri.v1.runtime", {}).get("containerd", {})
runtime_name = runtime.get("default_runtime_name", "runc")
options = runtime.get("runtimes", {}).get(runtime_name, {}).get("options", {})
disabled_plugins = config.get("disabled_plugins", [])

print("[INFO] Default runtime:", runtime_name)
print("[INFO] SystemdCgroup:", options.get("SystemdCgroup"))
print("[INFO] Disabled plugins:", disabled_plugins)

if options.get("SystemdCgroup") is not True:
    sys.exit("[ERROR] SystemdCgroup must be true. Existing config was not overwritten.")

for plugin in disabled_plugins:
    if "cri" in plugin:
        sys.exit("[ERROR] CRI is disabled. Existing config was not overwritten.")
PYTHON

echo "[PASS] Containerd configuration validated."

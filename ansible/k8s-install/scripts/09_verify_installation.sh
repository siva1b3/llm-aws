#!/usr/bin/env bash
# Read-only verification. This script does not install or restart anything.
# Continue after failed checks so the final summary includes every result.
set -uo pipefail

if ! sudo -v; then
  echo "[ERROR] Sudo authentication failed."
  exit 1
fi

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

# These local helpers only print results and count them. Checks remain visible below.
pass() {
  echo "[PASS] $1"
  PASS_COUNT=$((PASS_COUNT + 1))
}
warn() {
  echo "[WARN] $1"
  WARN_COUNT=$((WARN_COUNT + 1))
}
fail() {
  echo "[FAIL] $1"
  FAIL_COUNT=$((FAIL_COUNT + 1))
}

echo "========== 01 | Host and basic tools =========="
date -Is
hostname
cat /etc/os-release
uname -m
nproc
free -h
df -h /

. /etc/os-release
if [ "$ID" = ubuntu ] && [ "$VERSION_ID" = 24.04 ] && [ "$(uname -m)" = x86_64 ]; then
  pass "Ubuntu 24.04 on x86_64"
else
  fail "Expected Ubuntu 24.04 on x86_64"
fi

if [ "$(nproc)" -ge 2 ]; then
  pass "At least two CPUs available"
else
  fail "Fewer than two CPUs available"
fi

for TOOL in git curl wget gpg unzip zip tar jq make gcc g++ openssl python3; do
  if command -v "$TOOL"; then
    pass "$TOOL available"
  else
    fail "$TOOL missing"
  fi
done

if dpkg-query -W -f='${Status}\n' ca-certificates 2>/dev/null | grep -qx 'install ok installed'; then
  pass "CA certificates installed"
else
  fail "CA certificates missing"
fi

echo
echo "========== 02 | Swap and host settings =========="
if ACTIVE_SWAP=$(swapon --show --noheadings); then
  if [ -z "$ACTIVE_SWAP" ]; then
    pass "No active swap"
  else
    echo "$ACTIVE_SWAP"
    fail "Swap is active"
  fi
else
  fail "Could not inspect swap"
fi

SWAP_IN_FSTAB=false
while read -r DEVICE MOUNT_POINT FILESYSTEM OPTIONS REST; do
  if [ "$FILESYSTEM" = swap ] && [[ "$DEVICE" != \#* ]]; then
    SWAP_IN_FSTAB=true
  fi
done < /etc/fstab

if [ "$SWAP_IN_FSTAB" = false ]; then
  pass "No enabled swap entry in fstab"
else
  fail "Swap is configured in fstab"
fi

if grep -w overlay /proc/filesystems; then
  pass "OverlayFS available"
else
  fail "OverlayFS unavailable"
fi

if grep -Ex '[[:space:]]*overlay[[:space:]]*' /etc/modules-load.d/containerd.conf; then
  pass "Overlay module configured at boot"
else
  fail "Overlay boot configuration missing"
fi

if [ "$(sysctl -n net.ipv4.ip_forward)" = 1 ]; then
  pass "IPv4 forwarding enabled"
else
  fail "IPv4 forwarding disabled"
fi

if grep -Ex '[[:space:]]*net[.]ipv4[.]ip_forward[[:space:]]*=[[:space:]]*1[[:space:]]*' /etc/sysctl.d/99-kubernetes.conf; then
  pass "Persistent forwarding setting exists"
else
  fail "Persistent forwarding setting missing"
fi

echo "[INFO] Cgroup filesystem type:"
stat -fc %T /sys/fs/cgroup

echo
echo "========== 03 | Runtime versions and services =========="
if /usr/local/bin/containerd --version | grep -F ' v2.3.7 '; then
  pass "containerd 2.3.7 installed"
else
  fail "Expected containerd 2.3.7"
fi

if /usr/local/sbin/runc --version | grep -Fx 'runc version 1.5.2'; then
  pass "runc 1.5.2 installed"
else
  fail "Expected runc 1.5.2"
fi

if sudo systemctl is-active containerd; then
  pass "containerd active"
else
  fail "containerd inactive"
fi

if sudo systemctl is-enabled containerd; then
  pass "containerd enabled at boot"
else
  fail "containerd not enabled"
fi

if sudo test -S /run/containerd/containerd.sock; then
  pass "containerd socket exists"
else
  fail "containerd socket missing"
fi

if sudo /usr/local/bin/ctr version; then
  pass "containerd server responds"
else
  fail "containerd server did not respond"
fi

if PLUGIN_LIST=$(sudo /usr/local/bin/ctr plugins ls); then
  echo "$PLUGIN_LIST"
  # Compare the TYPE, ID and STATUS columns for each required plugin.
  for REQUIRED_PLUGIN in \
    'io.containerd.snapshotter.v1 overlayfs' \
    'io.containerd.cri.v1 images' \
    'io.containerd.cri.v1 runtime' \
    'io.containerd.grpc.v1 cri'
  do
    read -r PLUGIN_TYPE PLUGIN_NAME <<< "$REQUIRED_PLUGIN"
    PLUGIN_STATUS=$(echo "$PLUGIN_LIST" |
      awk -v type="$PLUGIN_TYPE" -v name="$PLUGIN_NAME" '$1 == type && $2 == name {print $NF}')
    if [ "$PLUGIN_STATUS" = ok ]; then
      pass "$REQUIRED_PLUGIN loaded"
    else
      fail "$REQUIRED_PLUGIN missing or unhealthy"
    fi
  done
else
  fail "Could not list runtime plugins"
fi

echo "[INFO] Optional unused snapshotters may show skip."

echo
echo "========== 04 | Effective containerd configuration =========="
# Containerd expands defaults/imports; Python reads the resulting TOML.
if sudo /usr/local/bin/containerd --config /etc/containerd/config.toml config dump |
  python3 -c '
import sys
import tomllib

config = tomllib.loads(sys.stdin.read())
runtime = config.get("plugins", {}).get("io.containerd.cri.v1.runtime", {}).get("containerd", {})
runtime_name = runtime.get("default_runtime_name", "runc")
options = runtime.get("runtimes", {}).get(runtime_name, {}).get("options", {})
print("[INFO] Runtime:", runtime_name)
print("[INFO] SystemdCgroup:", options.get("SystemdCgroup"))

if options.get("SystemdCgroup") is not True:
    sys.exit("[ERROR] SystemdCgroup is not true")

for plugin in config.get("disabled_plugins", []):
    if "cri" in plugin:
        sys.exit("[ERROR] CRI plugin is disabled")
'
then
  pass "Disk configuration enables systemd cgroups and CRI"
else
  fail "Containerd configuration validation failed"
fi

echo
echo "========== 05 | Kubernetes versions and package holds =========="
if kubeadm version -o short | grep -Ex 'v1[.]37[.][0-9]+'; then
  pass "kubeadm 1.37.x installed"
else
  fail "kubeadm version check failed"
fi

if kubelet --version | grep -Ex 'Kubernetes v1[.]37[.][0-9]+'; then
  pass "kubelet 1.37.x installed"
else
  fail "kubelet version check failed"
fi

if kubectl version --client -o json | jq -er '.clientVersion.gitVersion' | grep -Ex 'v1[.]37[.][0-9]+'; then
  pass "kubectl 1.37.x installed"
else
  fail "kubectl version check failed"
fi

if PACKAGE_VERSIONS=$(dpkg-query -W -f='${Version}\n' kubeadm kubelet kubectl); then
  echo "$PACKAGE_VERSIONS"
  UNIQUE_VERSION_COUNT=$(echo "$PACKAGE_VERSIONS" | sort -u | wc -l)
  if [ "$UNIQUE_VERSION_COUNT" -eq 1 ]; then
    pass "All three Kubernetes packages have the same version"
  else
    fail "Kubernetes package versions differ"
  fi
else
  fail "Could not read Kubernetes package versions"
fi

HELD_PACKAGES=$(apt-mark showhold)
for PACKAGE in kubeadm kubelet kubectl cri-tools; do
  if echo "$HELD_PACKAGES" | grep -Fxq "$PACKAGE"; then
    pass "$PACKAGE held against automatic upgrades"
  else
    warn "$PACKAGE is not held"
  fi
done

echo
echo "========== 06 | Kubelet and cluster state =========="
if systemctl is-enabled kubelet; then
  pass "kubelet enabled at boot"
else
  fail "kubelet not enabled"
fi

# is-active returns a nonzero status for inactive; that is expected here.
KUBELET_STATE=$(systemctl is-active kubelet)
if [ "$KUBELET_STATE" = inactive ]; then
  pass "kubelet intentionally stopped before init/join"
else
  warn "Expected inactive kubelet; found: $KUBELET_STATE"
fi

for CONFIG_FILE in /etc/kubernetes/admin.conf /etc/kubernetes/kubelet.conf /var/lib/kubelet/config.yaml /etc/kubernetes/manifests/kube-apiserver.yaml; do
  if sudo test -e "$CONFIG_FILE"; then
    warn "Cluster configuration exists: $CONFIG_FILE"
  else
    pass "Absent before init/join: $CONFIG_FILE"
  fi
done

echo
echo "========== 07 | CRI API readiness =========="
if command -v crictl >/dev/null; then
  if crictl --version | grep -Ex 'crictl version v?1[.]37[.][0-9]+'; then
    pass "crictl 1.37.x installed"
  else
    fail "crictl version check failed"
  fi

  if CRI_INFO=$(sudo crictl \
    --runtime-endpoint unix:///run/containerd/containerd.sock \
    --image-endpoint unix:///run/containerd/containerd.sock \
    --timeout 10s info)
  then
    echo "$CRI_INFO"
    if echo "$CRI_INFO" | jq -e '.status.conditions | any(.type == "RuntimeReady" and .status == true)'; then
      pass "CRI RuntimeReady is true"
    else
      fail "CRI RuntimeReady is not true or the response is invalid"
    fi
  else
    fail "CRI API did not respond successfully"
  fi
else
  fail "crictl missing; install it using step 08"
fi

echo "[INFO] NetworkReady=false is expected before CNI configuration."

echo
echo "========== 08 | Recent runtime warnings and errors =========="
if ! sudo journalctl -u containerd -b -p warning -n 20 --no-pager; then
  warn "Could not read containerd logs"
fi

echo
echo "========== 09 | Summary =========="
echo "PASS: $PASS_COUNT"
echo "WARN: $WARN_COUNT"
echo "FAIL: $FAIL_COUNT"
echo "[INFO] Registry pulls, container execution, CNI and AWS connectivity were not tested."
echo "[INFO] Persistent settings were inspected, not tested through a reboot."
echo "[INFO] Enabled kubelet may attempt to start after reboot before init/join."

if [ "$FAIL_COUNT" -gt 0 ]; then
  echo "[RESULT] Verification failed. Review the failed checks."
  exit 1
fi

echo "[RESULT] All executed mandatory checks passed. Review warnings separately."

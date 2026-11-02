



for tool in git curl wget gpg unzip zip tar jq make gcc g++ openssl; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf '[OK]      %s\n' "$tool"
  else
    printf '[MISSING] %s\n' "$tool"
  fi
done

for package in ca-certificates gnupg build-essential; do
  if dpkg-query -W -f='${Status}\n' "$package" 2>/dev/null | grep -qx 'install ok installed'; then
    printf '[OK]      %s\n' "$package"
  else
    printf '[MISSING] %s\n' "$package"
  fi
done





sudo apt-get update && sudo apt-get install -y \
  ca-certificates \
  curl \
  wget \
  gnupg \
  git \
  unzip \
  zip \
  tar \
  jq \
  build-essential \
  openssl






echo "=== CPU architecture ==="
uname -m

echo "=== CPU count ==="
nproc

echo "=== Existing container tools ==="
for tool in containerd runc docker kubeadm kubelet kubectl; do
  command -v "$tool" || true
done

echo "=== Swap ==="
swapon --show

echo "=== Kernel modules ==="
lsmod | grep -E '^(overlay|br_netfilter)\b' || true

echo "=== IPv4 forwarding ==="
sysctl net.ipv4.ip_forward


sudo modprobe overlay

echo overlay | sudo tee /etc/modules-load.d/containerd.conf

cat <<'EOF' | sudo tee /etc/sysctl.d/99-kubernetes.conf
net.ipv4.ip_forward = 1
EOF

sudo sysctl --load=/etc/sysctl.d/99-kubernetes.conf


lsmod | grep '^overlay'
sysctl net.ipv4.ip_forward








  set -euo pipefail

  INSTALL_DIR=$(mktemp -d /tmp/containerd-install.XXXXXX)
  cd "$INSTALL_DIR"

  CONTAINERD_VERSION="2.3.7"
  RUNC_VERSION="1.5.2"

  CONTAINERD_URL="https://github.com/containerd/containerd/releases/download/v${CONTAINERD_VERSION}"
  RUNC_URL="https://github.com/opencontainers/runc/releases/download/v${RUNC_VERSION}"
  ARCHIVE="containerd-${CONTAINERD_VERSION}-linux-amd64.tar.gz"

  echo "[1] Downloading containerd and its checksum"
  curl -fSL --retry 3 -o "$ARCHIVE" "$CONTAINERD_URL/$ARCHIVE"
  curl -fSL --retry 3 -o "$ARCHIVE.sha256sum" \
    "$CONTAINERD_URL/$ARCHIVE.sha256sum"

  echo "[2] Verifying containerd"
  sha256sum --check "$ARCHIVE.sha256sum"

  echo "[3] Downloading runc and its checksums"
  curl -fSL --retry 3 -o runc.amd64 "$RUNC_URL/runc.amd64"
  curl -fSL --retry 3 -o runc.sha256sum "$RUNC_URL/runc.sha256sum"

  echo "[4] Verifying runc"
  grep -E '[[:space:]]+\*?runc\.amd64$' runc.sha256sum \
    > runc-amd64.sha256sum
  sha256sum --check runc-amd64.sha256sum

  echo "[5] Installing verified binaries"
  sudo tar -C /usr/local -xzf "$ARCHIVE"
  sudo install -D -m 0755 runc.amd64 /usr/local/sbin/runc

  echo "[6] Checking installed versions"
  /usr/local/bin/containerd --version
  /usr/local/sbin/runc --version




  set -euo pipefail

  sudo mkdir -p /etc/containerd

  # Preserve any existing configuration.
  if sudo test -f /etc/containerd/config.toml; then
    sudo cp -a /etc/containerd/config.toml \
      "/etc/containerd/config.toml.bak.$(date +%Y%m%d%H%M%S)"
  fi

  /usr/local/bin/containerd config default |
    sudo tee /etc/containerd/config.toml >/dev/null

  sudo sed -i \
    's/SystemdCgroup = false/SystemdCgroup = true/g' \
    /etc/containerd/config.toml

  # Verify the setting and validate that containerd can read the configuration.
  sudo grep 'SystemdCgroup = true' /etc/containerd/config.toml
  sudo /usr/local/bin/containerd \
    --config /etc/containerd/config.toml config dump >/dev/null





  set -euo pipefail

  SERVICE_FILE=$(mktemp)

  curl -fSL --retry 3 \
    https://raw.githubusercontent.com/containerd/containerd/v2.3.7/containerd.service \
    -o "$SERVICE_FILE"

  sudo install -D -m 0644 "$SERVICE_FILE" \
    /usr/local/lib/systemd/system/containerd.service

  sudo systemctl daemon-reload
  sudo systemctl enable --now containerd




sudo systemctl is-active containerd
sudo systemctl is-enabled containerd

sudo /usr/local/bin/ctr version

sudo /usr/local/bin/ctr plugins ls |
  grep -E '(^TYPE|cri|overlayfs)'




  set -euo pipefail

  echo "[1] Adding the Kubernetes repository signing key"
  sudo install -d -m 0755 /etc/apt/keyrings

  curl -fsSL --retry 3 \
    https://pkgs.k8s.io/core:/stable:/v1.37/deb/Release.key |
    sudo gpg --batch --yes --dearmor \
      -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg

  sudo chmod 0644 /etc/apt/keyrings/kubernetes-apt-keyring.gpg

  echo "[2] Adding the Kubernetes 1.37 repository"
  echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.37/deb/ /' |
    sudo tee /etc/apt/sources.list.d/kubernetes.list

  sudo apt-get update

  echo "[3] Selecting an available 1.37 patch version"
  K8S_VERSION=$(
    apt-cache madison kubeadm |
      awk '$3 ~ /^1\.37\./ {print $3}' |
      sort -Vr |
      sed -n '1p'
  )

  if [ -z "$K8S_VERSION" ]; then
    echo "[ERROR] No Kubernetes 1.37 package version found."
    exit 1
  fi

  echo "[INFO] Installing version: $K8S_VERSION"

  sudo apt-get install -y \
    "kubeadm=$K8S_VERSION" \
    "kubelet=$K8S_VERSION" \
    "kubectl=$K8S_VERSION"

  echo "[4] Holding package versions"
  sudo apt-mark hold kubeadm kubelet kubectl

  echo "[5] Keeping kubelet stopped until init or join"
  sudo systemctl stop kubelet
  sudo systemctl enable kubelet





kubeadm version -o short
kubelet --version
kubectl version --client

apt-mark showhold

systemctl is-enabled kubelet
systemctl is-active kubelet


sudo apt-get update &&
sudo apt-get install -y 'cri-tools=1.37.*'

command -v crictl
crictl --version















































  #!/usr/bin/env bash
  set -euo pipefail

  echo "========== 01 | Check required tools =========="

  for TOOL in kubeadm ip; do
    if command -v "$TOOL" >/dev/null; then
      echo "[OK] $TOOL is available."
    else
      echo "[ERROR] $TOOL is missing."
      exit 1
    fi
  done

  echo
  echo "========== 02 | Check containerd =========="

  if sudo systemctl is-active --quiet containerd; then
    echo "[OK] Containerd is running."
  else
    echo "[ERROR] Containerd is not running."
    exit 1
  fi

  echo
  echo "========== 03 | Download Kubernetes images =========="

  KUBERNETES_VERSION=$(kubeadm version -o short)
  echo "[INFO] Kubernetes version: $KUBERNETES_VERSION"

  if sudo kubeadm config images pull \
    --kubernetes-version "$KUBERNETES_VERSION" \
    --cri-socket unix:///run/containerd/containerd.sock
  then
    echo "[PASS] Required Kubernetes images are available locally."
  else
    echo "[ERROR] Image download failed. Review the output above."
    exit 1
  fi

  echo
  echo "========== 04 | Show IPv4 routes =========="

  ip -4 route

  echo
  echo "========== 05 | Summary =========="
  echo "[PASS] Containerd is running and Kubernetes images are available."
  echo "[INFO] No cluster was initialized or joined."
  echo "[INFO] Pod and Service network ranges still need to be selected."
  echo "[INFO] Host routes do not show all AWS VPC, VPN, or peering routes."











  set -euo pipefail

  API_SERVER_IP="172.31.84.225"
  POD_NETWORK="192.168.0.0/16"
  SERVICE_NETWORK="10.96.0.0/12"
  KUBERNETES_VERSION=$(kubeadm version -o short)

  echo "========== 01 | Check current server =========="

  if ip -4 address show dev ens5 | grep -Fq "inet $API_SERVER_IP/"; then
    echo "[PASS] Expected private IP is present."
  else
    echo "[ERROR] Expected private IP is missing. Stop and review."
    exit 1
  fi

  if sudo test -e /etc/kubernetes/kubelet.conf ||
     sudo test -e /etc/kubernetes/manifests/kube-apiserver.yaml
  then
    echo "[ERROR] Cluster configuration already exists."
    echo "[INFO] Do not repeat initialization without investigating."
    exit 1
  fi

  echo
  echo "========== 02 | Initialize control plane =========="

  if sudo kubeadm init \
    --kubernetes-version "$KUBERNETES_VERSION" \
    --apiserver-advertise-address "$API_SERVER_IP" \
    --pod-network-cidr "$POD_NETWORK" \
    --service-cidr "$SERVICE_NETWORK" \
    --cri-socket unix:///run/containerd/containerd.sock
  then
    echo "[PASS] Control-plane initialization completed."
  else
    echo "[ERROR] Initialization failed."
    echo "[INFO] Preserve the output. Do not automatically reset or rerun."
    exit 1
  fi

  echo
  echo "========== 03 | Configure kubectl for this user =========="

  mkdir -p "$HOME/.kube"

  if [ -f "$HOME/.kube/config" ]; then
    cp -p "$HOME/.kube/config" \
      "$HOME/.kube/config.backup.$(date +%Y%m%d%H%M%S)"
  fi

  sudo install \
    -o "$(id -u)" \
    -g "$(id -g)" \
    -m 0600 \
    /etc/kubernetes/admin.conf \
    "$HOME/.kube/config"

  echo
  echo "========== 04 | Inspect initial cluster state =========="

  kubectl --kubeconfig "$HOME/.kube/config" get nodes -o wide
  kubectl --kubeconfig "$HOME/.kube/config" get pods -n kube-system -o wide

  echo
  echo "[INFO] Calico has not been installed yet."
  echo "[INFO] The node may be NotReady and CoreDNS may be Pending."
  echo "[NEXT] Install Calico using the same Pod network: $POD_NETWORK"














  set -euo pipefail

  echo "========== 01 | API server readiness =========="
  kubectl get --raw='/readyz?verbose'

  echo
  echo "========== 02 | Wait for control-plane Pods =========="

  if kubectl wait \
    --namespace kube-system \
    --selector tier=control-plane \
    --for=condition=Ready \
    pod \
    --timeout=120s
  then
    echo "[PASS] Control-plane Pods are ready."
  else
    echo "[ERROR] Control-plane Pods did not become ready."
    kubectl get pods -n kube-system -o wide
    kubectl get events -n kube-system \
      --sort-by=.metadata.creationTimestamp
    exit 1
  fi

  echo
  echo "========== 03 | Check default addons =========="
  kubectl get deployment coredns -n kube-system
  kubectl get daemonset kube-proxy -n kube-system

  echo
  echo "========== 04 | Current cluster state =========="
  kubectl get nodes -o wide
  kubectl get pods -n kube-system -o wide

  echo
  echo "[PASS] Control-plane checks completed."
  echo "[NEXT] Install Calico for Pod networking."













  set -euo pipefail

  CALICO_VERSION="v3.33.0"
  MANIFEST_URL="https://raw.githubusercontent.com/projectcalico/calico/$CALICO_VERSION/manifests"

  echo "========== 01 | Check Kubernetes API =========="
  kubectl get --raw='/readyz'

  echo
  echo "========== 02 | Install Calico resource definitions =========="

  kubectl apply --server-side \
    -f "$MANIFEST_URL/v3_projectcalico_org.yaml"

  echo
  echo "========== 03 | Install Tigera Operator =========="

  kubectl apply --server-side \
    -f "$MANIFEST_URL/tigera-operator.yaml"

  kubectl wait \
    --for=condition=Established \
    crd/installations.operator.tigera.io \
    --timeout=120s

  kubectl rollout status \
    deployment/tigera-operator \
    --namespace tigera-operator \
    --timeout=180s

  echo
  echo "========== 04 | Configure Calico networking =========="

  kubectl apply -f - <<'EOF'
apiVersion: operator.tigera.io/v1
kind: Installation
metadata:
  name: default
spec:
  variant: Calico
  calicoNetwork:
    linuxDataplane: Iptables
    bgp: Disabled
    nodeAddressAutodetectionV4:
      kubernetes: NodeInternalIP
    ipPools:
      - cidr: 192.168.0.0/16
        blockSize: 26
        encapsulation: VXLAN
        natOutgoing: Enabled
        nodeSelector: all()
EOF

  echo
  echo "========== 05 | Wait for node readiness =========="

  if kubectl wait \
    --for=condition=Ready \
    node/ip-172-31-84-225 \
    --timeout=300s
  then
    echo "[PASS] Node is Ready."
  else
    echo "[ERROR] Node did not become Ready within five minutes."
    kubectl get pods -A -o wide
    kubectl get tigerastatus
    kubectl get events -A --sort-by=.metadata.creationTimestamp
    exit 1
  fi

  echo
  echo "========== 06 | Check DNS and cluster state =========="

  kubectl rollout status \
    deployment/coredns \
    --namespace kube-system \
    --timeout=180s

  kubectl get nodes -o wide
  kubectl get pods -A -o wide
  kubectl get tigerastatus

  echo
  echo "[INFO] Review Calico status for remaining progressing or degraded components."







































bash <<'VERIFY'
set -uo pipefail

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

section() {
  printf '\n============================================================\n'
  printf ' %s\n' "$1"
  printf '============================================================\n'
}

pass() {
  printf '[PASS] %s\n' "$*"
  PASS_COUNT=$((PASS_COUNT + 1))
}

warn() {
  printf '[WARN] %s\n' "$*"
  WARN_COUNT=$((WARN_COUNT + 1))
}

fail() {
  printf '[FAIL] %s\n' "$*"
  FAIL_COUNT=$((FAIL_COUNT + 1))
}

check() {
  local description="$1"
  local output
  shift

  printf '\n[CHECK] %s\n' "$description"
  printf '[COMMAND]'
  printf ' %q' "$@"
  printf '\n'

  if output=$("$@" 2>&1); then
    [ -z "$output" ] || printf '%s\n' "$output"
    pass "$description"
  else
    [ -z "$output" ] || printf '%s\n' "$output"
    fail "$description"
  fi
}

section "00 | Verification scope"
printf 'Target: Kubernetes 1.37.x + containerd 2.3.x\n'
printf 'Stage: Installed only; before kubeadm init or join\n'
printf 'Time: %s\n' "$(date -Is)"
printf 'User: %s\n' "$(id -un)"
printf 'Host: %s\n' "$(hostname)"
printf 'Actions: Read-only checks; sudo is needed for runtime access\n'

if sudo -n true; then
  pass "Sudo access is available without prompting"
else
  fail "Sudo requires authentication; verification stopped"
  printf '\nRun sudo true separately to authenticate, then rerun this script.\n'
  exit 1
fi

section "01 | Operating system and resources"
cat /etc/os-release
printf '\nKernel: '
uname -r
printf 'Architecture: '
uname -m
printf 'CPU count: '
nproc
printf '\nMemory:\n'
free -h
printf '\nRoot filesystem:\n'
df -h /

check "Architecture is x86_64" \
  bash -c '[ "$(uname -m)" = x86_64 ]'

check "At least 2 CPUs are available" \
  bash -c '[ "$(nproc)" -ge 2 ]'

section "02 | Basic tools"
for tool in git curl wget gpg tar unzip zip jq make gcc g++ openssl; do
  if tool_path=$(command -v "$tool"); then
    pass "$tool: $tool_path"
  else
    warn "$tool is missing; not every development tool is required for Kubernetes"
  fi
done

check "CA certificates package is installed" \
  bash -c 'dpkg-query -W -f="\${Status}\n" ca-certificates |
    grep -x "install ok installed"'

section "03 | Swap and kernel preparation"
check "No swap is currently active" \
  bash -c '
    output=$(swapon --show --noheadings) || exit 1
    if [ -n "$output" ]; then
      printf "%s\n" "$output"
      exit 1
    fi
    echo "No active swap"
  '

printf '\n[INFO] Active swap entries in /etc/fstab:\n'
if awk '
  $0 !~ /^[[:space:]]*#/ && NF >= 3 && $3 == "swap" {
    print
    found=1
  }
  END {exit !found}
' /etc/fstab; then
  warn "fstab contains swap entries; swap may return after reboot"
else
  pass "No active swap entries found in /etc/fstab"
fi

check "OverlayFS is available in the running kernel" \
  grep -w overlay /proc/filesystems

check "Overlay module is configured to load at boot" \
  grep -Ex '[[:space:]]*overlay[[:space:]]*' \
  /etc/modules-load.d/containerd.conf

check "IPv4 forwarding is enabled now" \
  bash -c '
    sysctl net.ipv4.ip_forward
    [ "$(sysctl -n net.ipv4.ip_forward)" = 1 ]
  '

check "IPv4 forwarding is recorded in our persistent configuration" \
  grep -Ex '[[:space:]]*net\.ipv4\.ip_forward[[:space:]]*=[[:space:]]*1[[:space:]]*' \
  /etc/sysctl.d/99-kubernetes.conf

printf '\n[INFO] Cgroup filesystem type:\n'
stat -fc %T /sys/fs/cgroup

printf '\n[INFO] Bridge filtering module, if loaded:\n'
if lsmod | grep '^br_netfilter'; then
  pass "br_netfilter is loaded"
else
  printf '[INFO] br_netfilter is not loaded; requirements depend on the future CNI choice\n'
fi

section "04 | Container runtime binaries"
check "containerd is in the requested 2.3 series" \
  bash -o pipefail -c \
  '/usr/local/bin/containerd --version | grep -E " v2\.3\.[0-9]+"'

check "runc binary runs successfully" \
  /usr/local/sbin/runc --version

section "05 | Containerd service and socket"
check "containerd is active" \
  sudo systemctl is-active containerd

check "containerd is enabled at boot" \
  sudo systemctl is-enabled containerd

check "Containerd runtime socket exists" \
  sudo test -S /run/containerd/containerd.sock

check "Containerd client can contact the server" \
  sudo /usr/local/bin/ctr \
  --address /run/containerd/containerd.sock version

printf '\n[INFO] Service configuration and process details:\n'
sudo systemctl show containerd \
  -p FragmentPath -p ExecStart -p MainPID \
  -p ActiveState -p SubState -p UnitFileState

section "06 | Containerd configuration"
check "Containerd configuration parses successfully" \
  sudo /usr/local/bin/containerd \
  --config /etc/containerd/config.toml config dump >/dev/null
# The command output above is suppressed because the full default
# configuration is large. Print its outcome explicitly.
printf '[INFO] Configuration validation completed; see final counters for failures\n'

check "Systemd cgroups are enabled and CRI is not explicitly disabled" \
  sudo python3 -c '
import tomllib

with open("/etc/containerd/config.toml", "rb") as file:
    config = tomllib.load(file)

runtime = (
    config.get("plugins", {})
    .get("io.containerd.cri.v1.runtime", {})
    .get("containerd", {})
)
default_runtime = runtime.get("default_runtime_name", "runc")
options = runtime.get("runtimes", {}).get(default_runtime, {}).get("options", {})
disabled = config.get("disabled_plugins", [])

print("Config version:", config.get("version"))
print("Default runtime:", default_runtime)
print("SystemdCgroup:", options.get("SystemdCgroup"))
print("Disabled plugins:", disabled)

assert options.get("SystemdCgroup") is True, "SystemdCgroup is not true"
assert not any("cri" in item for item in disabled), "A CRI plugin is disabled"
'

section "07 | Loaded runtime plugins"
PLUGIN_OUTPUT=$(sudo /usr/local/bin/ctr plugins ls 2>&1)
PLUGIN_RC=$?
printf '%s\n' "$PLUGIN_OUTPUT"

if [ "$PLUGIN_RC" -ne 0 ]; then
  fail "Could not query containerd plugins"
else
  for plugin in \
    "io.containerd.snapshotter.v1 overlayfs" \
    "io.containerd.cri.v1 images" \
    "io.containerd.cri.v1 runtime" \
    "io.containerd.grpc.v1 cri"
  do
    read -r plugin_type plugin_id <<< "$plugin"

    if awk -v type="$plugin_type" -v id="$plugin_id" '
      $1 == type && $2 == id && $NF == "ok" {found=1}
      END {exit !found}
    ' <<< "$PLUGIN_OUTPUT"; then
      pass "$plugin is loaded successfully"
    else
      fail "$plugin is missing or not healthy"
    fi
  done
fi

printf '\n[INFO] Optional snapshotters may show "skip"; that is not automatically a failure\n'

section "08 | Kubernetes versions and package holds"
check "kubeadm is version 1.37.x" \
  bash -o pipefail -c \
  'kubeadm version -o short | grep -E "^v1\.37\.[0-9]+$"'

check "kubelet is version 1.37.x" \
  bash -o pipefail -c \
  'kubelet --version | grep -E "^Kubernetes v1\.37\.[0-9]+$"'

check "kubectl client is version 1.37.x" \
  bash -o pipefail -c '
    kubectl version --client -o json |
      jq -er ".clientVersion.gitVersion" |
      grep -Ex "v1[.]37[.][0-9]+"
  '

printf '\n[INFO] Installed package versions:\n'
dpkg-query -W -f='${Package}\t${Version}\t${Status}\n' \
  kubeadm kubelet kubectl

check "All three Kubernetes packages have the same version" \
  bash -o pipefail -c '
    versions=$(dpkg-query -W -f="\${Version}\n" kubeadm kubelet kubectl) || exit 1
    printf "%s\n" "$versions"
    [ "$(printf "%s\n" "$versions" | sort -u | wc -l)" -eq 1 ]
  '

HELD_PACKAGES=$(apt-mark showhold)
for package in kubeadm kubelet kubectl; do
  if grep -Fxq "$package" <<< "$HELD_PACKAGES"; then
    pass "$package is held against automatic upgrades"
  else
    fail "$package is not held"
  fi
done

section "09 | Kubelet and cluster initialization state"
check "kubelet is enabled at boot" \
  systemctl is-enabled kubelet

KUBELET_STATE=$(systemctl is-active kubelet 2>/dev/null || true)
printf '[INFO] Kubelet state: %s\n' "$KUBELET_STATE"

if [ "$KUBELET_STATE" = inactive ]; then
  pass "kubelet is intentionally stopped before init/join"
else
  warn "Expected inactive at this stage; found $KUBELET_STATE"
fi

for config_file in \
  /etc/kubernetes/admin.conf \
  /etc/kubernetes/kubelet.conf \
  /var/lib/kubelet/config.yaml \
  /etc/kubernetes/manifests/kube-apiserver.yaml
do
  if sudo test -e "$config_file"; then
    warn "Cluster-related configuration already exists: $config_file"
  else
    pass "Not present, as expected before init/join: $config_file"
  fi
done

section "10 | CRI API check, if crictl is installed"
if command -v crictl >/dev/null 2>&1; then
  check "Kubernetes CRI API responds" \
    sudo crictl \
    --runtime-endpoint unix:///run/containerd/containerd.sock \
    --image-endpoint unix:///run/containerd/containerd.sock \
    --timeout 10s info
else
  warn "crictl is unavailable; direct CRI API verification was skipped"
fi

printf '\n[INFO] NetworkReady=false can be expected before CNI configuration\n'
printf '[INFO] CRI plugin status alone does not verify that a container can start\n'

section "11 | Recent containerd warnings and errors"
sudo journalctl -u containerd -b -p warning -n 20 --no-pager
printf '\n[INFO] Logs above are diagnostic; review messages in context\n'
printf '[INFO] Missing CNI configuration is expected at this stage\n'

section "12 | Final summary"
printf 'PASS : %d\n' "$PASS_COUNT"
printf 'WARN : %d\n' "$WARN_COUNT"
printf 'FAIL : %d\n' "$FAIL_COUNT"

if [ "$FAIL_COUNT" -gt 0 ]; then
  printf '\n[RESULT] One or more installation checks failed.\n'
else
  printf '\n[RESULT] All executed installation checks passed.\n'
fi

printf '\nScope limits:\n'
printf '  - No cluster was initialized or joined by this script.\n'
printf '  - Image pulls and actual container execution were not tested.\n'
printf '  - CNI networking, AWS security groups, and inter-node connectivity were not tested.\n'
printf '  - Persistent settings were inspected, not tested through a reboot.\n'

[ "$FAIL_COUNT" -eq 0 ]
VERIFY











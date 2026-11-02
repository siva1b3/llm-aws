#!/usr/bin/env bash
# Run with bash. Stop if a command fails; do not source this file.
set -euo pipefail

# Change to this script's directory so relative child paths work.
SCRIPT_DIRECTORY=$(dirname "$0")
cd "$SCRIPT_DIRECTORY"

if [ "$#" -gt 1 ]; then
  echo "Usage: bash master.sh [--verify-only]"
  exit 1
fi

if [ "${1:-}" = "--verify-only" ]; then
  bash scripts/09_verify_installation.sh
  exit 0
fi

if [ "$#" -ne 0 ]; then
  echo "Usage: bash master.sh [--verify-only]"
  exit 1
fi

echo "[INFO] Installing tools and preparing this server before init/join."
echo "[INFO] This is not a Kubernetes upgrade script."

# Run each step in order. A failed step stops the master script.
for SCRIPT_FILE in \
  scripts/01_check_prerequisites.sh \
  scripts/02_install_basic_tools.sh \
  scripts/03_prepare_host.sh \
  scripts/04_install_containerd_runc.sh \
  scripts/05_configure_containerd.sh \
  scripts/06_enable_containerd_service.sh \
  scripts/07_install_kubernetes.sh \
  scripts/08_install_cri_tools.sh \
  scripts/09_verify_installation.sh
do
  echo
  echo "============================================================"
  echo "[START] $SCRIPT_FILE"
  echo "============================================================"

  if bash "$SCRIPT_FILE"; then
    echo "[PASS] $SCRIPT_FILE completed."
  else
    echo "[ERROR] $SCRIPT_FILE failed. Remaining steps were not run."
    exit 1
  fi
done

echo "[DONE] Installation and verification completed."
echo "[INFO] No kubeadm init or kubeadm join was run."

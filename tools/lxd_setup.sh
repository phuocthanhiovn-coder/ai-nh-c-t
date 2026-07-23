#!/bin/bash
# Install LXD + zfs, init storage. Base for multi-user fake-spec containers.
export DEBIAN_FRONTEND=noninteractive
export PATH=$PATH:/snap/bin

echo "[1/4] Cai zfs + snapd..."
apt-get update -y >/dev/null 2>&1
apt-get install -y zfsutils-linux snapd >/dev/null 2>&1

echo "[2/4] Cai LXD (snap)..."
snap install lxd 2>&1 | tail -1
/snap/bin/lxd waitready --timeout=90

echo "[3/4] Init LXD (auto, zfs loop pool)..."
/snap/bin/lxd init --auto 2>&1 | tail -3

echo "[4/4] Kiem tra..."
echo -n "version: "; /snap/bin/lxc version 2>&1 | head -2 | tr '\n' ' '; echo
echo "storage:"; /snap/bin/lxc storage list 2>&1
echo "LXD_READY"

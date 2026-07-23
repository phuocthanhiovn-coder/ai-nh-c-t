#!/bin/bash
# Create zfs pool + "trial" profile (fake 4 vCPU / real 0.5 core / 1GB RAM / 10GB disk) + test container.
export PATH=$PATH:/snap/bin

echo "[1] Tao zfs pool 60GB (thin, cho disk quota)..."
lxc storage create zpool zfs size=60GiB 2>&1 | tail -1 || echo "  (zpool da ton tai)"

echo "[2] Tao profile 'trial'..."
lxc profile create trial 2>/dev/null || echo "  (trial da ton tai)"
lxc profile set trial limits.cpu 4                       # LXCFS hien nproc=4 (FAKE)
lxc profile set trial limits.cpu.allowance 50ms/100ms    # THAT: 0.5 core CPU time
lxc profile set trial limits.memory 1GiB                 # RAM that = 1GB
lxc profile device remove trial root 2>/dev/null || true
lxc profile device add trial root disk path=/ pool=zpool size=10GiB   # disk that 10GB
lxc profile device remove trial eth0 2>/dev/null || true
lxc profile device add trial eth0 nic network=lxdbr0 name=eth0 2>/dev/null || true

echo "[3] Launch container test 'u1' (keo image ubuntu 22.04 ~1-2 phut)..."
lxc delete -f u1 2>/dev/null || true
lxc launch ubuntu:22.04 u1 --profile trial 2>&1 | tail -2
sleep 10

echo "=== BEN TRONG CONTAINER u1 (thong so FAKE user thay) ==="
lxc exec u1 -- bash -c 'echo -n "nproc (fake CPU): "; nproc; echo -n "RAM: "; free -h | awk "/Mem/{print \$2}"; echo -n "Disk: "; df -h / | tail -1 | awk "{print \$2}"'
echo "=== CPU THAT (cgroup quota, 50000/100000 = 0.5 core) ==="
lxc exec u1 -- cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo "(cgroup v1?)"
echo "PROFILE_DONE"

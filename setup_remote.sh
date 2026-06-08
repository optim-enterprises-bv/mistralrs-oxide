#!/bin/bash
# setup_remote.sh - Setup cuda-oxide environment on remote GPU server

set -e

SERVER="dingo@192.168.50.180"

echo "=== Setting up remote GPU environment ==="

# Check if we can connect
ssh ${SERVER} "echo 'Connected'"

# Install Rust if not present
ssh ${SERVER} "which rustc || (curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && source ~/.cargo/env)"

# Install cuda-oxide
ssh ${SERVER} "which cargo-oxide || (cargo +nightly-2026-04-03 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide 2>&1 || echo 'Note: cargo-oxide may require specific toolchain')"

# Check GPU
ssh ${SERVER} "nvidia-smi"

# Clone cuda-oxide examples
ssh ${SERVER} "test -d ~/cuda-oxide || git clone https://github.com/NVlabs/cuda-oxide.git ~/cuda-oxide"

echo "=== Remote setup complete ==="
echo "Next steps:"
echo "1. SSH to server: ssh ${SERVER}"
echo "2. Test cuda-oxide: cd ~/cuda-oxide && cargo oxide run vecadd"
echo "3. Build mistralrs-oxide: cd ~/mistralrs-oxide && cargo oxide build"

#!/bin/bash
# deploy.sh - Deploy mistralrs-oxide to remote GPU server

set -e

SERVER="dingo@192.168.50.180"
REMOTE_DIR="~/mistralrs-oxide"

echo "=== Deploying mistralrs-oxide to ${SERVER} ==="

# Check SSH connection
echo "Checking SSH connection..."
ssh -o ConnectTimeout=5 ${SERVER} "echo 'Connected to remote server'" || {
    echo "Failed to connect to ${SERVER}"
    echo "Make sure the server is reachable and SSH keys are configured"
    exit 1
}

# Create remote directory
echo "Creating remote directory..."
ssh ${SERVER} "mkdir -p ${REMOTE_DIR}"

# Sync code (excluding .git and target)
echo "Syncing code..."
rsync -avz --exclude='.git' --exclude='target' \
    --exclude='*.ptx' --exclude='*.o' --exclude='*.so' \
    ./ ${SERVER}:${REMOTE_DIR}/

# Check CUDA on remote
echo "Checking CUDA on remote server..."
ssh ${SERVER} "nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv || echo 'nvidia-smi not available'"
ssh ${SERVER} "nvcc --version 2>/dev/null || echo 'NVCC not found'"

# Build on remote (optional)
read -p "Build on remote server? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Building on remote server..."
    ssh ${SERVER} "cd ${REMOTE_DIR} && cargo build --release 2>&1 || echo 'Build may require cuda-oxide toolchain'"
fi

echo "=== Deployment complete ==="
echo "To SSH into the server: ssh ${SERVER}"
echo "Remote directory: ${REMOTE_DIR}"

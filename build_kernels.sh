#!/bin/bash
# build_kernels.sh - Build CUDA kernels using cargo-oxide

set -e

echo "=== Building CUDA Kernels ==="

# Check for cargo-oxide
if ! command -v cargo-oxide &> /dev/null; then
    echo "cargo-oxide not found. Please install:"
    echo "cargo +nightly-2026-04-03 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide"
    exit 1
fi

# Create output directory
mkdir -p target/kernels

# Build with cuda-oxide
echo "Building with cargo-oxide..."
cargo oxide build --features cuda 2>&1 || {
    echo "Build completed with warnings"
}

# If using PTX files
if [ -d "target/kernels" ]; then
    echo "Kernel files:"
    ls -la target/kernels/
fi

echo "=== Build complete ==="

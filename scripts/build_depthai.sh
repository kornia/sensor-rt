#!/usr/bin/env bash
# Build depthai-core v3 from source into vendor/depthai (shared lib + headers).
# depthai-core is not packaged on conda-forge, so we vendor a pinned tag and build
# it with the system compiler (ABI-compatible with the cargo link). Run via
# `pixi run depthai-build` so cmake/ninja/libusb/pkg-config come from the pixi env.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/vendor"
SRC="$VENDOR/depthai-core"
PREFIX="$VENDOR/depthai"
TAG="v3.7.1"

mkdir -p "$VENDOR"

# Precompiled-once: if the prefix is already installed, do nothing — no clone, no
# vcpkg, no compile. (DEPTHAI_FORCE=1 to rebuild; `pixi run depthai-unpack` to
# restore a packaged prefix on a fresh machine instead of building.)
if ls "$PREFIX"/lib/libdepthai-core.so* >/dev/null 2>&1 && [ -z "${DEPTHAI_FORCE:-}" ]; then
    echo "[depthai] already installed at $PREFIX — skipping (set DEPTHAI_FORCE=1 to rebuild)"
    exit 0
fi

# Source lives in the pinned git submodule (vendor/depthai-core @ $TAG). Init it
# (with depthai-core's own nested submodules: xtensor, xtl, …) if the working
# tree is empty — e.g. on a fresh clone that skipped `--recursive`.
if [ ! -e "$SRC/CMakeLists.txt" ]; then
    echo "[depthai] initializing depthai-core submodule ($TAG) ..."
    git -C "$ROOT" submodule update --init --recursive vendor/depthai-core
fi

# Portability patches (idempotent: skip when already applied). See patches/*.patch.
for patch in "$ROOT"/patches/depthai-core-*.patch; do
    [ -e "$patch" ] || continue
    if git -C "$SRC" apply --check "$patch" 2>/dev/null; then
        echo "[depthai] applying $(basename "$patch")"
        git -C "$SRC" apply "$patch"
    fi
done

echo "[depthai] configuring (Release, shared) ..."
cmake -S "$SRC" -B "$SRC/build" -G Ninja \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_SHARED_LIBS=ON \
    -D DEPTHAI_BUILD_EXAMPLES=OFF \
    -D DEPTHAI_BUILD_TESTS=OFF \
    -D DEPTHAI_BUILD_DOCS=OFF \
    -D CMAKE_INSTALL_PREFIX="$PREFIX" \
    -D CMAKE_INSTALL_RPATH='$ORIGIN'

# Parallelism is RAM-bound here, NOT core-bound: depthai-core's TUs (xtensor /
# nlohmann-json / spdlog templates) peak at ~1.5-2 GB each in cc1plus. On a small
# Jetson (e.g. 7.4 GB Orin Nano) `--parallel $(nproc)` OOM-kills the box. Cap jobs
# so peak ≈ JOBS*2 GB stays under physical RAM, leaving headroom for the kernel.
# Override with DEPTHAI_JOBS=N if you have more RAM.
mem_gb=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo)
jobs="${DEPTHAI_JOBS:-$(( mem_gb >= 16 ? 4 : 2 ))}"
echo "[depthai] building (-j $jobs; ${mem_gb}GB RAM detected) ..."
cmake --build "$SRC/build" --target install -j "$jobs"

echo "[depthai] installed to $PREFIX"
ls -la "$PREFIX/lib" 2>/dev/null | grep -i depthai || true

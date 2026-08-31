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
# The stamp keys the install on the $TAG it was built from: the old guard was "a .so
# exists", so a TAG bump silently kept the previous library.
STAMP="$PREFIX/.build-stamp"
WANT="$TAG"
if ls "$PREFIX"/lib/libdepthai-core.so* >/dev/null 2>&1 && [ -z "${DEPTHAI_FORCE:-}" ] \
   && [ "$(cat "$STAMP" 2>/dev/null)" = "$WANT" ]; then
    echo "[depthai] already installed at $PREFIX ($TAG) — skipping"
    exit 0
fi
if ls "$PREFIX"/lib/libdepthai-core.so* >/dev/null 2>&1 && [ -z "${DEPTHAI_FORCE:-}" ]; then
    echo "[depthai] prefix exists but was built from a different tag — rebuilding"
fi

# Source lives in the pinned git submodule (vendor/depthai-core @ $TAG). Init it
# (with depthai-core's own nested submodules: xtensor, xtl, …) if the working
# tree is empty — e.g. on a fresh clone that skipped `--recursive`.
if [ ! -e "$SRC/CMakeLists.txt" ]; then
    echo "[depthai] initializing depthai-core submodule ($TAG) ..."
    git -C "$ROOT" submodule update --init --recursive vendor/depthai-core
fi

# NO patches are applied here: this repo builds depthai-core AS PINNED. A configuration
# that needs source changes (e.g. DEPTHAI_OPENCV_SUPPORT=OFF, which does not link upstream
# as of v3.7.1) is the CALLER's choice, so the caller owns the fixes — see
# flux-xlerobot/deploy/depthai-patches/ and its laptop_setup.sh. The real fix is upstreaming
# them; until then, nothing in this library repo modifies vendored third-party source.

echo "[depthai] configuring (Release, shared) ..."
cmake -S "$SRC" -B "$SRC/build" -G Ninja \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_SHARED_LIBS=ON \
    -D DEPTHAI_BUILD_EXAMPLES=OFF \
    -D DEPTHAI_BUILD_TESTS=OFF \
    -D DEPTHAI_BUILD_DOCS=OFF \
    -D CMAKE_INSTALL_PREFIX="$PREFIX" \
    -D CMAKE_INSTALL_RPATH='$ORIGIN' \
    ${DEPTHAI_CMAKE_EXTRA:-}

# Parallelism is RAM-bound here, NOT core-bound: depthai-core's TUs (xtensor /
# nlohmann-json / spdlog templates) peak at ~1.5-2 GB each in cc1plus. On a small
# Jetson (e.g. 7.4 GB Orin Nano) `--parallel $(nproc)` OOM-kills the box. Cap jobs
# so peak ≈ JOBS*2 GB stays under physical RAM, leaving headroom for the kernel.
# Override with DEPTHAI_JOBS=N if you have more RAM.
mem_gb=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo)
jobs="${DEPTHAI_JOBS:-$(( mem_gb >= 16 ? 4 : 2 ))}"
echo "[depthai] building (-j $jobs; ${mem_gb}GB RAM detected) ..."
cmake --build "$SRC/build" --target install -j "$jobs"

echo "$WANT" > "$STAMP"
echo "[depthai] installed to $PREFIX"
ls -la "$PREFIX/lib" 2>/dev/null | grep -i depthai || true

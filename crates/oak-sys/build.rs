//! Build the C++ shim (oak_bridge.cpp) via CMake so `find_package(depthai)`
//! supplies depthai-core's include + link flags, then link the result. depthai-core
//! is built into `vendor/depthai` by `pixi run depthai-build`; the prefix is found
//! by walking up to the workspace root (or `DEPTHAI_PREFIX` if set).

use std::path::PathBuf;
use std::process::Command;

/// Locate libusb's libdir. depthai-core records `libusb-1.0.so` in its DT_NEEDED
/// but expects it provided externally (the pixi/conda env), so the final link
/// needs that dir on the search path to resolve the transitive symbols. Try
/// pkg-config first (pixi sets PKG_CONFIG_PATH), then fall back to scanning the
/// in-repo pixi env.
fn find_libusb_dir(workspace_root: &std::path::Path) -> Option<PathBuf> {
    if let Ok(out) = Command::new("pkg-config")
        .args(["--variable=libdir", "libusb-1.0"])
        .output()
    {
        if out.status.success() {
            let dir = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !dir.is_empty() && PathBuf::from(&dir).is_dir() {
                return Some(PathBuf::from(dir));
            }
        }
    }
    let pixi = workspace_root.join(".pixi/envs/default/lib");
    pixi.join("libusb-1.0.so").exists().then_some(pixi)
}

fn find_depthai_prefix() -> PathBuf {
    if let Ok(p) = std::env::var("DEPTHAI_PREFIX") {
        return PathBuf::from(p);
    }
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        let cand = dir.join("vendor/depthai");
        if cand.join("lib/cmake/depthai").is_dir() {
            return cand;
        }
        if !dir.pop() {
            return PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vendor/depthai");
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=oak_bridge.cpp");
    println!("cargo:rerun-if-changed=oak_bridge.h");
    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-env-changed=DEPTHAI_PREFIX");

    let prefix = find_depthai_prefix();

    // Build liboak_bridge.a with depthai's include/link flags resolved by CMake.
    let dst = cmake::Config::new(".")
        .define("CMAKE_PREFIX_PATH", prefix.to_string_lossy().as_ref())
        .define("CMAKE_BUILD_TYPE", "Release")
        .build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=oak_bridge");

    // depthai-core (shared, self-contained via vcpkg-static deps) + C++ runtime.
    println!("cargo:rustc-link-search=native={}/lib", prefix.display());
    println!("cargo:rustc-link-lib=dylib=depthai-core");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", prefix.display());
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // libusb: a transitive (DT_NEEDED) dep of libdepthai-core.so provided by the
    // pixi env — needed on the search path to satisfy the link, and on the rpath
    // so the loader finds it at runtime.
    let ws_root = prefix
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or(&prefix)
        .to_path_buf();
    if let Some(usb) = find_libusb_dir(&ws_root) {
        println!("cargo:rustc-link-search=native={}", usb.display());
        println!("cargo:rustc-link-lib=dylib=usb-1.0");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", usb.display());
    } else {
        println!("cargo:warning=oak-sys: libusb-1.0 not found (build inside the pixi env: `pixi run cargo build`)");
    }
}

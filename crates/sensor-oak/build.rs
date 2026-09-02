//! Bake the depthai-core (+ bundled libusb) directories into the rpath of this
//! crate's examples, from the metadata depthai-sys exports. Cargo does not propagate
//! a dependency's `rustc-link-arg`, so every crate with binaries repeats this.
fn main() {
    println!("cargo:rerun-if-env-changed=DEP_DEPTHAI_CORE_RPATH");
    if let Ok(rpath) = std::env::var("DEP_DEPTHAI_CORE_RPATH") {
        for dir in rpath.split(':').filter(|d| !d.is_empty()) {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        }
    }
}

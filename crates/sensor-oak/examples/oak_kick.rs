//! Recover a PoE OAK wedged in bootloader state (the "X_LINK_BOOTLOADER" failure where a camera drops
//! off and no in-process reconnect brings it back). Reboots it via a bootloader open+drop so the next
//! open succeeds — the in-code version of the old manual `depthai.DeviceBootloader` recovery.
//!
//!   pixi run -- cargo run -p sensor-oak --example oak_kick -- <ip-or-deviceId>
//!   pixi run -- cargo run -p sensor-oak --example oak_kick            # first wedged device found
//!
//! Exit: 0 = kicked (wait ~8s before re-opening the camera) or nothing to kick; 1 = driver error.

use sensor_oak::kick_wedged_oak;

fn main() -> std::process::ExitCode {
    let target = std::env::args().nth(1);
    match kick_wedged_oak(target.as_deref()) {
        Ok(true) => {
            println!(
                "kicked {} — waiting ~8s for the reboot",
                target.as_deref().unwrap_or("(first wedged)")
            );
            std::thread::sleep(std::time::Duration::from_secs(8));
            println!("done — the camera should re-open now");
            std::process::ExitCode::SUCCESS
        }
        Ok(false) => {
            println!("nothing to kick (target absent or not wedged)");
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("kick failed: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

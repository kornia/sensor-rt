/* Pure-C ABI over depthai-core v3 (C++). The Rust side never sees any C++ —
 * opaque handle + plain types; C++ exceptions are caught and converted to return
 * codes + oak_last_error().
 *
 * Scope: the STEREO + IMU pipeline only. The colour/depth and H.264 paths were
 * removed while that modality is the one under development. */
#ifndef OAK_BRIDGE_H
#define OAK_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct oak_device oak_device;

/* One IMU reading: an accelerometer + gyroscope pair, timestamped on the SAME
 * host-synced epoch timeline as the image frames (so IMU and frames can be
 * interpolated against each other directly). */
typedef struct {
    uint64_t ts_ns;    /* accelerometer packet capture time, epoch ns */
    float ax, ay, az;  /* accelerometer, m/s^2 */
    float gx, gy, gz;  /* gyroscope, rad/s */
} oak_imu_sample;

/* Open an OAK in the STEREO+IMU modality: the two mono cameras (CAM_B = left,
 * CAM_C = right) streamed as a time-synced RGB888 pair, plus the on-board IMU on
 * its own queue. NO colour camera, NO StereoDepth, NO encoder — this is the raw
 * stereo + inertial source for VIO / stereo-feature work, not the depth path
 * (for aligned depth use oak_open with enable_depth).
 *
 * A separate entry point rather than more flags on oak_open: the pipeline shares
 * no nodes with it, and the RGBD/H.264 paths must not regress.
 *
 *   width/height : per-eye output size. The mono sensors are requested as RGB888i
 *                  (gray replicated to 3 channels) so the frames drop straight
 *                  into a 3-channel consumer with no host conversion.
 *   fps          : stereo pair rate.
 *   imu_hz       : accelerometer + gyroscope report rate (e.g. 200-400). The IMU
 *                  is OPTIONAL — a device without one (or whose IMU fails to
 *                  start) still streams stereo, with oak_has_imu() == 0.
 *
 * Returns NULL on failure (reason via oak_last_error). */
oak_device *oak_open_stereo(const char *device_id, int width, int height,
                            int fps, int imu_hz);

/* True (1) if this device runs the stereo pair pipeline (oak_poll_stereo yields
 * frames), i.e. it was opened with oak_open_stereo. */
int oak_has_stereo(const oak_device *dev);

/* True (1) if the on-board IMU is running (oak_poll_imu yields samples). 0 on a
 * device with no IMU, or when the IMU node failed to start — the stereo pair is
 * unaffected either way, so callers should degrade rather than abort. */
int oak_has_imu(const oak_device *dev);

/* Pull the next time-synced stereo pair. On success (return 1) both out-pointers
 * alias device-internal buffers VALID UNTIL THE NEXT oak_poll_stereo:
 *   left/right -> width*height*3 bytes each, interleaved R,G,B (tightly packed)
 *   len        -> that byte length (width*height*3), same for both eyes
 *   ts_ns      -> capture time of the LEFT frame, epoch ns
 * Blocks up to ~1s for the pair. Returns 1 on a pair, 0 on timeout/no-frame,
 * -1 on error. */
int oak_poll_stereo(oak_device *dev,
                    const uint8_t **left, const uint8_t **right,
                    int *width, int *height, int *len, uint64_t *ts_ns);

/* Retain one eye of the CURRENT stereo pair so its pixel buffer stays valid past the
 * next oak_poll_stereo. Returns an opaque handle, or NULL if there is no current pair.
 *
 * The span handed out by oak_poll_stereo is only guaranteed until the next poll. A
 * caller that needs to keep a frame longer — buffering, VIO windows, async work —
 * retains it, which copies the underlying shared_ptr so depthai cannot recycle the
 * buffer. The pixel pointer from oak_poll_stereo stays valid for as long as the
 * handle is alive.
 *
 *   eye : 0 = left (CAM_B), 1 = right (CAM_C)
 *
 * Every successful retain MUST be matched by exactly one oak_frame_release. */
void *oak_stereo_retain(oak_device *dev, int eye);

/* Release a handle from oak_stereo_retain. NULL is a no-op. */
void oak_frame_release(void *handle);

/* Drain queued IMU samples into the caller's array. NON-BLOCKING: writes up to
 * `max` samples, sets *n to how many were written (0 when none are queued or the
 * IMU isn't running). Returns 1 / 0 / -1 (error).
 *
 * The IMU reports far faster than the frame rate, so call this in a loop until
 * *n == 0 (or with a generous `max`) each iteration, otherwise the batch queue
 * overflows and samples are silently dropped. */
int oak_poll_imu(oak_device *dev, oak_imu_sample *out, int max, int *n);

/* Recover a PoE OAK wedged in bootloader state: reboot it via a bootloader open+drop so the next
 * oak_open succeeds. `target` = IP/name or deviceId (NULL = first wedged device). 1 = kicked (wait ~8s),
 * 0 = nothing to kick, -1 = error. Blocking. */
int oak_kick(const char *target);

/* Stop the pipeline and free the device. */
void oak_close(oak_device *dev);

/* Last error message on the calling thread (empty string if none). */
const char *oak_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* OAK_BRIDGE_H */

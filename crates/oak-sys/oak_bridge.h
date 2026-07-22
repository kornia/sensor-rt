/* Pure-C ABI over depthai-core v3 (C++). bindgen / the Rust side never see any
 * C++ — same discipline as trt-sys's trt_bridge.h. Opaque handle + plain types;
 * C++ exceptions are caught and converted to return codes + oak_last_error(). */
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

/* Open an OAK device (USB or PoE) and start an RGB(+aligned depth) pipeline at the
 * requested size/fps. Returns NULL on failure (reason via oak_last_error).
 *
 *   device_id   : NULL or "" -> the first available device (USB auto). Otherwise a
 *                 device identifier that selects a SPECIFIC camera — an MxId (works
 *                 for USB or PoE) or an IP address (PoE). depthai resolves the
 *                 transport, so the same call opens a USB or a PoE OAK.
 *   enable_h264 : 1 -> the device ALSO encodes the colour stream to H.264 on-device,
 *                 drained as a standalone compressed video stream via oak_poll_video.
 *                 0 -> synced RGB-D only.
 *   enable_depth: 1 -> build the StereoDepth node (synced RGB + aligned depth). 0 -> RGB only,
 *                 NO stereo node — for cameras with no factory calibration (whose StereoDepth would
 *                 fail at runtime and crash the pipeline) or when depth simply isn't needed.
 *   video_only  : 1 -> build the H.264 encoder + a raw RGB888 output, but NO stereo/depth — for a
 *                 camera with no stereo pair or no usable factory calibration, whose StereoDepth
 *                 would fail at runtime and crash the pipeline. Overrides enable_h264/enable_depth.
 *                 0 -> the normal raw-RGBD(+optional H.264) pipeline.
 *
 *                 NOTE the name is now narrower than the behaviour: this path still ships raw RGB
 *                 (w*h*3 per frame over XLink), because these cameras feed ONBOARD COMPUTE, which
 *                 needs real frames — not only a bitstream to look at. So oak_poll DOES return
 *                 frames here (RGB, no depth) and oak_poll_video runs alongside it.
 *
 * The synced group (oak_poll) carries raw RGB888 + (when enable_depth) aligned uint16 depth — the
 * H.264 stream is separate (a second camera output), not part of the sync. */
oak_device *oak_open(const char *device_id, int width, int height, int fps,
                     int enable_h264, int enable_depth, int video_only);

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

/* Drain queued IMU samples into the caller's array. NON-BLOCKING: writes up to
 * `max` samples, sets *n to how many were written (0 when none are queued or the
 * IMU isn't running). Returns 1 / 0 / -1 (error).
 *
 * The IMU reports far faster than the frame rate, so call this in a loop until
 * *n == 0 (or with a generous `max`) each iteration, otherwise the batch queue
 * overflows and samples are silently dropped. */
int oak_poll_imu(oak_device *dev, oak_imu_sample *out, int max, int *n);

/* True (1) if the device produced a depth stream (stereo present), else 0. */
int oak_has_depth(const oak_device *dev);

/* True (1) if an on-device H.264 video stream is running (oak_open enable_h264=1). */
int oak_has_video(const oak_device *dev);

/* Factory intrinsics of the streamed (depth-aligned) RGB camera. Returns 0 on
 * success, -1 on error. */
int oak_intrinsics(const oak_device *dev,
                   float *fx, float *fy, float *cx, float *cy);

/* Pull the next time-synced frame. On success (return 1) the out-pointers alias
 * device-internal buffers VALID UNTIL THE NEXT oak_poll on this device:
 *   rgb       -> width*height*3 bytes, interleaved R,G,B (RGB888, tightly packed)
 *   depth_mm  -> depth_w*depth_h uint16, millimetres, aligned to rgb (NULL if no depth)
 *   depth_w/h -> depth grid size; may be SMALLER than width/height (downscaled on-device),
 *                still aligned to the rgb frame, so scale coords by width/depth_w.
 * Returns 1 on a frame, 0 on timeout/no-frame, -1 on error. */
int oak_poll(oak_device *dev,
             const uint8_t **rgb, const uint16_t **depth_mm,
             int *width, int *height, int *rgb_len, uint64_t *ts_ns,
             int *depth_w, int *depth_h);

/* Pull the next H.264 access unit (one encoded colour frame). NON-BLOCKING: returns
 * 0 immediately when none is ready (or H.264 wasn't enabled). On success (return 1)
 * *data aliases the device bitstream buffer (valid until the next oak_poll_video),
 * *len is its byte length, *ts_ns the capture timestamp. Returns 1 / 0 / -1 (error).
 * Drain in a loop until 0 each iteration so the encoder queue never overflows (a
 * dropped P-frame glitches the stream until the next keyframe, ~1 s). */
int oak_poll_video(oak_device *dev,
                   const uint8_t **data, int *len, uint64_t *ts_ns);

/* Decoupled polls: raw RGB and depth on their OWN queues (no on-device sync), each non-blocking,
 * zero-copy-aliased, and timestamped so the caller pairs them by timestamp. 1 / 0 / -1. */
int oak_poll_rgb(oak_device *dev,
                 const uint8_t **rgb, int *width, int *height, int *rgb_len, uint64_t *ts_ns);
int oak_poll_depth(oak_device *dev,
                   const uint16_t **depth_mm, int *depth_w, int *depth_h, uint64_t *ts_ns);

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

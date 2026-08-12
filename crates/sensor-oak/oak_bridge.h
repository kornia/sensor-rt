/* Pure-C ABI over depthai-core v3 (C++). The Rust side never sees any C++ —
 * opaque handle + plain types; C++ exceptions are caught and converted to return
 * codes + oak_last_error().
 *
 * Two independent modalities behind ONE opaque handle, each its own oak_open_*
 * entry point that builds a wholly separate pipeline (shared nodes would let one
 * modality's changes regress the other):
 *   - STEREO + IMU  (oak_open_stereo): the two mono cameras as a Sync'd GRAY8 pair
 *     + the on-board IMU. Raw stereo/inertial source for VIO.
 *   - RGBD + H.264  (oak_open_rgbd): CAM_A colour (RGB888) + StereoDepth aligned to
 *     it (uint16 mm) + an on-device H.264 colour stream. The colour, depth, and
 *     video are DECOUPLED — each on its own queue, pulled + timestamped
 *     independently (oak_poll_rgb / oak_poll_depth / oak_poll_video), paired
 *     downstream by their shared host-synced timeline. This is the site's camera
 *     producer path (flux-oak). */
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
 *   width/height : per-eye output size. CAM_B/CAM_C are MONOCHROME, so the frames
 *                  are GRAY8 (one byte per pixel). Requesting RGB888i would make
 *                  depthai replicate gray across three channels and ship 3x the
 *                  bytes for no extra information — consumers that need RGB expand
 *                  it on the GPU instead.
 *   fps          : stereo pair rate.
 *   imu_hz       : accelerometer + gyroscope report rate (e.g. 200-400). The IMU
 *                  is OPTIONAL — the IMU node is only built after a
 *                  getConnectedIMU() preflight confirms the board carries one, so
 *                  an IMU-less device still streams stereo (oak_has_imu() == 0)
 *                  and a missing IMU never reaches pipeline start. When the EEPROM
 *                  carries the IMU extrinsics, samples are rotated into the LEFT
 *                  (CAM_B) optical frame — see oak_imu_aligned().
 *
 * Returns NULL on failure (reason via oak_last_error). */
oak_device *oak_open_stereo(const char *device_id, int width, int height,
                            int fps, int imu_hz);

/* True (1) if the on-board IMU is running (oak_poll_imu yields samples). 0 on a
 * device with no IMU, or when the IMU node failed to start — the stereo pair is
 * unaffected either way, so callers should degrade rather than abort. */
int oak_has_imu(const oak_device *dev);

/* Open an OAK in the RGBD + H.264 modality: CAM_A colour (RGB888) + StereoDepth
 * (CAM_B/CAM_C) aligned to the colour grid (uint16 mm) + an on-device H.264 colour
 * stream. Colour, depth, and video are DECOUPLED — each on its own queue, pulled
 * independently (oak_poll_rgb / oak_poll_depth / oak_poll_video) and paired
 * downstream by their shared host-synced timestamps.
 *
 *   width/height : CAM_A colour output size (the depth is aligned to this grid, then
 *                  optionally downscaled on-device — see OAK_DEPTH_DIV — so it stays
 *                  aligned but ships fewer bytes; its own dims come back from
 *                  oak_poll_depth).
 *   fps          : colour + encoder rate. The raw-RGB pull rate (OAK_RGB_FPS, default
 *                  10) and depth rate (OAK_DEPTH_FPS, default = fps) are throttled
 *                  separately to spare XLink; the H.264 stream runs at full fps.
 *   enable_h264  : build the on-device H.264 encoder (oak_poll_video). Usually 1.
 *   enable_depth : build StereoDepth. Set 0 for an uncalibrated camera — the stereo
 *                  node would otherwise fail at runtime and crash the pipeline.
 *   video_only   : build ONLY the H.264 encoder (no RGB888/depth): the device ships
 *                  just the small bitstream for low-bandwidth viewing. oak_poll_rgb/
 *                  _depth yield nothing; drain oak_poll_video.
 *   imu_hz       : accelerometer + gyroscope rate (oak_poll_imu), same semantics as
 *                  oak_open_stereo's. 0 disables the IMU node; the node is only built
 *                  after a getConnectedIMU() preflight confirms the board carries an
 *                  IMU, so a missing IMU never costs the image streams
 *                  (oak_has_imu()==0, streams run on) and never reaches pipeline
 *                  start. Works in video_only mode too.
 *
 * Auto-fall-back: if enable_depth is set but the device can't actually produce depth
 * (no stereo pair, or a wiped/blank calibration → fx=0), the pipeline silently drops
 * to video-only rather than pull raw RGB for a depth that would be garbage —
 * reflected by oak_has_sync()==0.
 *
 * Returns NULL on failure (reason via oak_last_error). */
oak_device *oak_open_rgbd(const char *device_id, int width, int height, int fps,
                          int enable_h264, int enable_depth, int video_only,
                          int imu_hz);

/* True (1) if StereoDepth is running (oak_poll_depth can yield aligned depth). */
int oak_has_depth(const oak_device *dev);

/* True (1) if the on-device H.264 colour stream is running (oak_poll_video yields). */
int oak_has_video(const oak_device *dev);

/* True (1) if this device runs the RGBD colour(+depth) pipeline (oak_poll_rgb yields
 * frames). 0 when it auto-fell-back to video-only (mono / uncalibrated): the caller
 * then advertises only the H.264 stream and never polls colour/depth. */
int oak_has_sync(const oak_device *dev);

/* Pull the next raw RGB888 colour frame from its own queue. NON-BLOCKING. On success
 * (return 1) `rgb` aliases a device-internal buffer VALID UNTIL THE NEXT oak_poll_rgb
 * (the caller copies it out before the next poll):
 *   rgb    -> width*height*3 bytes, RGB888 (tightly packed, 3 B/px)
 *   len    -> that byte length (width*height*3)
 *   ts_ns  -> capture time, epoch ns (shared timeline with depth + video)
 * Returns 1 on a frame, 0 when none is queued, -1 on error. Drain in a loop until 0. */
int oak_poll_rgb(oak_device *dev, const uint8_t **rgb,
                 int *width, int *height, int *len, uint64_t *ts_ns);

/* Pull the next aligned depth frame from its own queue. NON-BLOCKING. On success
 * (return 1) `depth_mm` aliases a device-internal buffer VALID UNTIL THE NEXT
 * oak_poll_depth:
 *   depth_mm         -> depth_w*depth_h uint16 values, millimetres (0 = no return)
 *   depth_w/depth_h  -> depth grid size (may be < colour size: downscaled on-device,
 *                       but aligned to the colour grid — scale coords by rgb_w/depth_w)
 *   ts_ns            -> capture time, epoch ns (shared timeline with colour + video)
 * Returns 1 / 0 (none) / -1 (error). Drain in a loop until 0. */
int oak_poll_depth(oak_device *dev, const uint16_t **depth_mm,
                   int *depth_w, int *depth_h, uint64_t *ts_ns);

/* Pull the next on-device H.264 access unit (Annex-B NAL units) from its own queue.
 * NON-BLOCKING. On success (return 1) `data` aliases a device-internal buffer VALID
 * UNTIL THE NEXT oak_poll_video:
 *   data   -> `len` bytes of H.264 bitstream
 *   ts_ns  -> capture time, epoch ns (shared timeline with colour + depth)
 * Returns 1 / 0 (none) / -1 (error). Drain in a loop until 0 so the encoder queue
 * never overflows (a dropped P-frame glitches the stream until the next keyframe). */
int oak_poll_video(oak_device *dev, const uint8_t **data, int *len, uint64_t *ts_ns);

/* Factory intrinsics of the LEFT (CAM_B) camera at the streamed size — the stereo
 * reference frame. Returns 0 on success, -1 on error. */
int oak_intrinsics(const oak_device *dev,
                   float *fx, float *fy, float *cx, float *cy);

/* Pull the next time-synced stereo pair. On success (return 1) both out-pointers
 * alias device-internal buffers VALID UNTIL THE NEXT oak_poll_stereo:
 *   left/right -> width*height bytes each, GRAY8 (tightly packed, 1 byte/px)
 *   len        -> that byte length (width*height), same for both eyes
 *   ts_ns      -> capture time of the LEFT frame, epoch ns
 *   l_hnd/r_hnd-> owned retain handles for the two eyes; the pixel buffers stay valid
 *                 for as long as these live, independently of later polls. Release each
 *                 with oak_frame_release exactly once. Never NULL on success.
 * Blocks up to ~1s for the pair. Returns 1 on a pair, 0 on timeout/no-frame,
 * -1 on error. */
int oak_poll_stereo(oak_device *dev,
                    const uint8_t **left, const uint8_t **right,
                    int *width, int *height, int *len, uint64_t *ts_ns,
                    void **l_hnd, void **r_hnd);

/* Release a retain handle from oak_poll_stereo. NULL is a no-op. */
void oak_frame_release(void *handle);

/* Drain queued IMU samples into the caller's array. NON-BLOCKING: writes up to
 * `max` samples, sets *n to how many were written (0 when none are queued or the
 * IMU isn't running). Returns 1 / 0 / -1 (error).
 *
 * Sample frame: when the device calibration carries the IMU extrinsics
 * (oak_imu_aligned() == 1), samples are rotated into the modality's reference
 * camera optical frame — CAM_A (colour) on the RGBD modality, CAM_B (left) on the
 * stereo modality. Otherwise they are the raw IMU-chip frame, which is
 * axis-permuted vs the camera by the board mounting.
 *
 * The IMU reports far faster than the frame rate, so call this in a loop until
 * *n == 0 (or with a generous `max`) each iteration, otherwise the batch queue
 * overflows and samples are silently dropped. */
int oak_poll_imu(oak_device *dev, oak_imu_sample *out, int max, int *n);

/* True (1) when oak_poll_imu samples are calibration-rotated into the camera
 * optical frame (CAM_A on the RGBD modality, CAM_B/left on the stereo modality —
 * IMU extrinsics present in the EEPROM and validated as a proper rotation). 0 =
 * raw IMU-chip frame; the shim logs WHY to stderr (no extrinsics vs rejected). */
int oak_imu_aligned(const oak_device *dev);

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

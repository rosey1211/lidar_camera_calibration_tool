#!/usr/bin/env python3
"""
Camera-LiDAR Extrinsic Calibration Tool
========================================
Reads ROS1 (.bag) or ROS2 bag files directly — no ROS runtime required.
Displays LiDAR points projected onto the camera image with a 6-DOF overlay.

Usage:
  python calibrate.py <bag>                  # auto-select topics interactively
  python calibrate.py <bag> --image /cam/image_raw --lidar /velodyne_points --info /cam/camera_info

Controls:
  Trackbars  — adjust translation (±3 m, 1 cm steps) and rotation (±180°, 0.5° steps)
  n / b      — next / previous frame
  a          — ADJUST mode (trackbars)
  p          — PICK mode (click a LiDAR dot to add a 3D-2D correspondence)
  s          — solve PnP from collected correspondences (needs ≥ 6 pairs)
  c          — clear all correspondence pairs
  r          — reset extrinsic to zero
  w          — write calibration.yaml
  q / Esc    — quit
"""

import sys
import argparse
import time
import numpy as np
import cv2
import yaml
from pathlib import Path

# ─── X11 cursor control ───────────────────────────────────────────────────────
# Uses python-xlib to set the cursor shape on the OpenCV window.
# XC_left_ptr=68 (normal arrow), XC_crosshair=34, XC_tcross=108
#
# OpenCV's zoom mode sets a hand cursor on its *internal* GTK child window,
# which overrides any cursor set only on the parent.  The fix is to push our
# cursor onto every X11 window in the subtree so there is nowhere for
# OpenCV's hand cursor to survive.  Cursor objects and the window reference
# are cached so the per-frame overhead is cheap.

_xlib_ok      = False
_xdisplay     = None
_xcursor_font = None
_xcursor_cache: dict = {}   # cursor_shape (int) → Xlib cursor object
_xcal_win     = None        # cached reference to the top-level OpenCV window

def _xlib_init():
    global _xlib_ok, _xdisplay, _xcursor_font
    try:
        from Xlib import display
        _xdisplay     = display.Display()
        _xcursor_font = _xdisplay.open_font('cursor')
        _xlib_ok = True
    except Exception:
        pass

def _xlib_find_window(root, title):
    """Recursively search the X11 window tree for a window whose title contains title."""
    try:
        name = root.get_wm_name()
        if name and title in str(name):
            return root
    except Exception:
        pass
    try:
        for child in root.query_tree().children:
            result = _xlib_find_window(child, title)
            if result:
                return result
    except Exception:
        pass
    return None

def _apply_cursor_tree(win, cur):
    """Set cursor on win and every X11 descendant so child windows can't override."""
    try:
        win.change_attributes(cursor=cur)
    except Exception:
        pass
    try:
        for child in win.query_tree().children:
            _apply_cursor_tree(child, cur)
    except Exception:
        pass

def _set_cursor(cursor_shape: int):
    """Set the X11 cursor on every window in the calibration window's subtree."""
    global _xcal_win
    if not _xlib_ok:
        return
    try:
        from Xlib import X
        if _xcal_win is None:
            _xcal_win = _xlib_find_window(_xdisplay.screen().root, WIN)
        if _xcal_win is None:
            return
        if cursor_shape not in _xcursor_cache:
            if cursor_shape < 0:
                _xcursor_cache[cursor_shape] = X.NONE
            else:
                _xcursor_cache[cursor_shape] = _xcursor_font.create_glyph_cursor(
                    _xcursor_font, cursor_shape, cursor_shape + 1,
                    (0xFFFF, 0xFFFF, 0xFFFF), (0x0000, 0x0000, 0x0000))
        _apply_cursor_tree(_xcal_win, _xcursor_cache[cursor_shape])
        _xdisplay.sync()
    except Exception:
        pass

_XC_CROSSHAIR = 34   # standard X11 cursor font shape
_XC_ARROW     = 68

# ─── Bag reading ──────────────────────────────────────────────────────────────

def _is_ros1(path: str) -> bool:
    return Path(path).is_file() and Path(path).suffix == '.bag'

def _make_reader_and_store(path: str):
    from rosbags.typesys import get_typestore, Stores
    if _is_ros1(path):
        from rosbags.rosbag1 import Reader
        return Reader(path), get_typestore(Stores.ROS1_NOETIC), True
    else:
        from rosbags.rosbag2 import Reader
        return Reader(path), get_typestore(Stores.ROS2_HUMBLE), False

def list_topics(path: str) -> tuple[dict, bool]:
    reader, _, is_ros1 = _make_reader_and_store(path)
    topics = {}
    with reader:
        for conn in reader.connections:
            topics[conn.topic] = conn.msgtype
    return topics, is_ros1

def iter_messages(path: str, wanted: set):
    """Yield (topic, timestamp_ns, msg) for every message on a wanted topic."""
    reader, typestore, is_ros1 = _make_reader_and_store(path)
    with reader:
        conns = [c for c in reader.connections if c.topic in wanted]
        for conn, ts, raw in reader.messages(connections=conns):
            if is_ros1:
                msg = typestore.deserialize_ros1(raw, conn.msgtype)
            else:
                msg = typestore.deserialize_cdr(raw, conn.msgtype)
            yield conn.topic, ts, msg

# ─── Message decoders ─────────────────────────────────────────────────────────

_PC2_DTYPES = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2',
               5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}

def pc2_to_xyz(msg) -> np.ndarray:
    """Return (N, 3) float32 XYZ from a PointCloud2 message."""
    dt = np.dtype({
        'names':    [f.name for f in msg.fields],
        'formats':  [_PC2_DTYPES.get(f.datatype, 'u1') for f in msg.fields],
        'offsets':  [f.offset for f in msg.fields],
        'itemsize': msg.point_step,
    })
    raw = np.frombuffer(bytes(msg.data), dtype=dt)
    pts = np.stack([raw['x'].astype(np.float32),
                    raw['y'].astype(np.float32),
                    raw['z'].astype(np.float32)], axis=1)
    return pts[np.isfinite(pts).all(axis=1)]

def decode_image(msg) -> np.ndarray:
    """Return a BGR uint8 array from a sensor_msgs/Image or CompressedImage."""
    if hasattr(msg, 'format'):  # CompressedImage
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    enc = msg.encoding.lower().replace('/', '')
    h, w = msg.height, msg.width
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc in ('rgb8', '8uc3'):
        return cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if enc == 'bgr8':
        return raw.reshape(h, w, 3).copy()
    if enc in ('mono8', '8uc1'):
        return cv2.cvtColor(raw.reshape(h, w), cv2.COLOR_GRAY2BGR)
    bayer = {'bayer_rggb8': cv2.COLOR_BayerRGGB2BGR, 'bayer_bggr8': cv2.COLOR_BayerBGGR2BGR,
             'bayer_gbrg8': cv2.COLOR_BayerGBRG2BGR, 'bayer_grbg8': cv2.COLOR_BayerGRBG2BGR}
    if enc in bayer:
        return cv2.cvtColor(raw.reshape(h, w), bayer[enc])
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")

def parse_camera_info(msg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = np.asarray(msg.K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(msg.D, dtype=np.float64).flatten()
    P = np.asarray(msg.P, dtype=np.float64).reshape(3, 4)
    return K, D, P


def _check_rectification(K: np.ndarray, D: np.ndarray, P: np.ndarray,
                          img_topic: str) -> bool:
    """
    Inspect camera_info fields and topic name to determine whether the image
    is rectified (undistorted).  Returns True if image appears rectified.

    Three independent signals are checked:
      1. D all zeros   → distortion removed by image_proc (strong indicator)
      2. Topic name contains 'rect'  → ROS naming convention
      3. P[:3,:3] ≈ K  with D≈0     → P is the rectified projection matrix

    If the image IS rectified you should use D=0 when projecting LiDAR points
    (the distortion was already removed from the pixels you are clicking on).
    Using non-zero D on a rectified image will make near/far alignment wrong.
    """
    d_nonzero  = np.any(np.abs(D) > 1e-9)
    topic_rect = 'rect' in img_topic.lower()
    # For a rectified monocular image, P[:3,:3] should equal K exactly
    P_K_match  = np.allclose(P[:, :3], K, atol=0.5) if P.shape == (3, 4) else False

    print("\n  ┌─ Rectification check ──────────────────────────────────────┐")
    print(f"  │ Image topic : {img_topic}")
    print(f"  │ D           : {np.round(D, 6).tolist()}")
    print(f"  │ K[fx,fy,cx,cy]: {K[0,0]:.2f}, {K[1,1]:.2f}, {K[0,2]:.2f}, {K[1,2]:.2f}")
    print(f"  │ P[:3,:3]≈K  : {P_K_match}   topic says rect: {topic_rect}")

    if not d_nonzero:
        verdict = 'RECTIFIED'
        reason  = "D is all zeros — distortion already removed by image pipeline"
    elif topic_rect:
        verdict = 'RECTIFIED'
        reason  = "topic name contains 'rect'"
    else:
        verdict = 'RAW (unrectified)'
        reason  = "D has non-zero values and topic name does not contain 'rect'"

    print(f"  │")
    print(f"  │ ► Image appears: {verdict}")
    print(f"  │   {reason}")

    if verdict == 'RECTIFIED':
        print(f"  │")
        print(f"  │   Projection will use D=0 (correct for rectified images).")
        print(f"  │   Using non-zero D on a rectified image causes depth-dependent")
        print(f"  │   misalignment — near and far objects cannot both align.")
    else:
        print(f"  │")
        print(f"  │   Projection will apply the distortion model from camera_info.")
        print(f"  │   If near/far alignment is inconsistent, verify D is correct")
        print(f"  │   and the image has not been rectified upstream.")

    print("  └────────────────────────────────────────────────────────────┘\n")

    return verdict == 'RECTIFIED'

# ─── Geometry ─────────────────────────────────────────────────────────────────

def euler_to_R(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0,           1, 0          ],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0,           0,           1]])
    return Rz @ Ry @ Rx

def R_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        rx = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        ry = np.degrees(np.arctan2(-R[2, 0], sy))
        rz = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        rx = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        ry = np.degrees(np.arctan2(-R[2, 0], sy))
        rz = 0.0
    return rx, ry, rz

def project(pts3d: np.ndarray, K: np.ndarray, D: np.ndarray,
            R: np.ndarray, t: np.ndarray):
    """
    Project LiDAR points into the camera image.

    Returns
    -------
    pts2d   : (M, 2) float32 — pixel coordinates of visible points
    ranges  : (M,)  float32 — distance from camera origin in metres
    idx_front: boolean mask of pts3d that are in front of the camera
    """
    pts_cam = (R @ pts3d.T + t.reshape(3, 1)).T          # (N, 3)
    in_front = pts_cam[:, 2] > 0.1
    if not in_front.any():
        return np.zeros((0, 2), np.float32), np.zeros(0, np.float32), in_front

    front = pts3d[in_front].astype(np.float32)
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    pts2d, _ = cv2.projectPoints(front, rvec, t.reshape(3, 1).astype(np.float64),
                                  K.astype(np.float64), D.astype(np.float64))
    pts2d    = pts2d.reshape(-1, 2)
    ranges   = np.linalg.norm(pts_cam[in_front], axis=1).astype(np.float32)
    return pts2d, ranges, in_front

# ─── Trackbars ────────────────────────────────────────────────────────────────

WIN    = 'LiDAR-Camera Calibration'
# Trackbar names include range description; OpenCV shows the raw integer beside
# each bar so we encode the scale in the name so the user knows the mapping.
_TNAMES = (
    'tx  range=-3.0..+3.0 m  (1 step = 1 cm)',
    'ty  range=-3.0..+3.0 m  (1 step = 1 cm)',
    'tz  range=-3.0..+3.0 m  (1 step = 1 cm)',
    'rx  range=-180..+180 deg  (1 step = 0.5 deg)',
    'ry  range=-180..+180 deg  (1 step = 0.5 deg)',
    'rz  range=-180..+180 deg  (1 step = 0.5 deg)',
)
_T_CTR    = 300   # ±3.00 m in 1 cm steps → 601 positions, centre = 300
_R_CTR    = 360   # ±180° in 0.5° steps  → 721 positions, centre = 360
_TB_RANGE = 'LiDAR color range (max meters)'

def make_trackbars():
    for name in _TNAMES[:3]:
        cv2.createTrackbar(name, WIN, _T_CTR, _T_CTR * 2, lambda _: None)
    for name in _TNAMES[3:]:
        cv2.createTrackbar(name, WIN, _R_CTR, _R_CTR * 2, lambda _: None)
    cv2.createTrackbar(_TB_RANGE, WIN, 30, 80, lambda _: None)
    cv2.setTrackbarMin(_TB_RANGE, WIN, 1)

def read_trackbars() -> tuple:
    tx = (cv2.getTrackbarPos(_TNAMES[0], WIN) - _T_CTR) * 0.01
    ty = (cv2.getTrackbarPos(_TNAMES[1], WIN) - _T_CTR) * 0.01
    tz = (cv2.getTrackbarPos(_TNAMES[2], WIN) - _T_CTR) * 0.01
    rx = (cv2.getTrackbarPos(_TNAMES[3], WIN) - _R_CTR) * 0.5
    ry = (cv2.getTrackbarPos(_TNAMES[4], WIN) - _R_CTR) * 0.5
    rz = (cv2.getTrackbarPos(_TNAMES[5], WIN) - _R_CTR) * 0.5
    return tx, ty, tz, rx, ry, rz

def set_trackbars(tx, ty, tz, rx, ry, rz):
    cv2.setTrackbarPos(_TNAMES[0], WIN, int(round(tx / 0.01)) + _T_CTR)
    cv2.setTrackbarPos(_TNAMES[1], WIN, int(round(ty / 0.01)) + _T_CTR)
    cv2.setTrackbarPos(_TNAMES[2], WIN, int(round(tz / 0.01)) + _T_CTR)
    cv2.setTrackbarPos(_TNAMES[3], WIN, int(round(rx / 0.5))  + _R_CTR)
    cv2.setTrackbarPos(_TNAMES[4], WIN, int(round(ry / 0.5))  + _R_CTR)
    cv2.setTrackbarPos(_TNAMES[5], WIN, int(round(rz / 0.5))  + _R_CTR)

# ─── Frame loading ────────────────────────────────────────────────────────────

def load_frames(cam_path, lidar_path, img_topic, lidar_topic, info_topic, max_frames=60):
    """
    Extract up to max_frames time-synced (bgr, xyz, K, D) tuples.
    cam_path and lidar_path may be the same bag or separate bags.
    Time-sync: for each image frame the nearest LiDAR frame within 200 ms is used.
    """
    K = D = None
    imgs   = {}   # ts_ns → bgr array
    clouds = {}   # ts_ns → (N,3) float32

    # ── Camera bag (image + camera_info) ──────────────────────────────────────
    cam_wanted = {img_topic}
    if info_topic:
        cam_wanted.add(info_topic)

    n_msgs = 0
    print(f"  Scanning camera bag: {cam_path}")
    for topic, ts, msg in iter_messages(cam_path, cam_wanted):
        n_msgs += 1
        if n_msgs % 200 == 0:
            print(f"    ... {n_msgs} msgs, {len(imgs)} images", end='\r')
        if topic == info_topic and K is None:
            K, D, P = parse_camera_info(msg)
            is_rect = _check_rectification(K, D, P, img_topic)
            if is_rect:
                D = np.zeros(5, dtype=np.float64)
        elif topic == img_topic:
            imgs[ts] = decode_image(msg)
    print(f"\n  Camera bag: {len(imgs)} images")

    # ── LiDAR bag ─────────────────────────────────────────────────────────────
    print(f"  Scanning lidar bag:  {lidar_path}")
    n_msgs = 0
    for topic, ts, msg in iter_messages(lidar_path, {lidar_topic}):
        n_msgs += 1
        if n_msgs % 200 == 0:
            print(f"    ... {n_msgs} msgs, {len(clouds)} clouds", end='\r')
        clouds[ts] = pc2_to_xyz(msg)
    print(f"\n  LiDAR bag:  {len(clouds)} clouds")

    print(f"  Bag scan complete: {len(imgs)} images, {len(clouds)} clouds")

    if K is None:
        raise RuntimeError(
            "No camera_info found in bag.\n"
            "Pass the correct topic with --info, or check that it was recorded.")

    img_ts   = sorted(imgs)
    cloud_ts = np.array(sorted(clouds), dtype=np.int64)

    frames = []
    for it in img_ts:
        if len(frames) >= max_frames:
            break
        diffs = np.abs(cloud_ts - np.int64(it))
        ci    = int(np.argmin(diffs))
        if diffs[ci] > 200_000_000:   # 200 ms in nanoseconds
            continue
        frames.append((imgs[it], clouds[cloud_ts[ci]], K, D))

    print(f"  {len(frames)} synced frames loaded")
    return frames

# ─── Mouse state ──────────────────────────────────────────────────────────────

_st = {
    'mode':        'ADJUST',  # 'ADJUST' | 'PICK'
    'pick_state':  'cam',     # 'cam' = waiting for camera click, 'lid' = waiting for lidar click
    'pending_cam': None,      # 2D float32[2] camera point awaiting its lidar pair
    'pairs':       [],        # list of (pt3d float32[3], pt2d_cam float32[2])
    'proj2d':      None,      # (M,2) float32 — projected LiDAR points this frame
    'proj3d':      None,      # (M,3) float32 — corresponding 3D LiDAR points
    'K':           None,      # stored so mouse callback can trigger auto-solve
    'D':           None,
    'needs_solve':       False,
    'has_loaded_calib':  False,  # True after --load; lowers auto-solve threshold to 4
    'mouse_xy':          (0, 0), # current mouse position for crosshair drawing
}

def _on_mouse(event, x, y, flags, param):
    _st['mouse_xy'] = (x, y)   # track position for crosshair drawing
    if _st['mode'] != 'PICK' or event != cv2.EVENT_LBUTTONDOWN:
        return

    if _st['pick_state'] == 'cam':
        # First click: record the camera image feature location
        _st['pending_cam'] = np.array([x, y], dtype=np.float32)
        _st['pick_state']  = 'lid'
        _set_cursor(_XC_CROSSHAIR + 1)   # slightly different shape for lidar-pick state
        print(f"  [{len(_st['pairs'])+1}] Camera point: ({x}, {y})"
              f"  — now click the corresponding LiDAR dot")

    elif _st['pick_state'] == 'lid':
        # Second click: snap to the nearest projected LiDAR point
        pts2d = _st['proj2d']
        pts3d = _st['proj3d']
        if pts2d is None or len(pts2d) == 0:
            print("  No LiDAR points visible — check gross alignment first")
            _st['pending_cam'] = None
            _st['pick_state']  = 'cam'
            _set_cursor(_XC_CROSSHAIR)
            return
        dists   = np.linalg.norm(pts2d - [x, y], axis=1)
        nearest = int(np.argmin(dists))
        if dists[nearest] > 30:
            print(f"  No LiDAR point within 30 px of click ({x},{y}) "
                  f"(closest = {dists[nearest]:.1f} px) — camera point cancelled")
            _st['pending_cam'] = None
            _st['pick_state']  = 'cam'
            _set_cursor(_XC_CROSSHAIR)
            return
        pt3d = pts3d[nearest].copy()
        pt2d = _st['pending_cam']
        _st['pairs'].append((pt3d, pt2d))
        _st['pending_cam'] = None
        _st['pick_state']  = 'cam'
        _set_cursor(_XC_CROSSHAIR)   # back to green cam-pick cursor
        print(f"  [{len(_st['pairs'])}] Pair complete: "
              f"3D=({pt3d[0]:.3f}, {pt3d[1]:.3f}, {pt3d[2]:.3f})  "
              f"2D=({pt2d[0]:.0f}, {pt2d[1]:.0f})")
        # Trigger auto-solve: threshold is 4 when a calibration was loaded
        # (good initial guess lets PnP converge with fewer points), 6 otherwise.
        min_pairs = 4 if _st['has_loaded_calib'] else 6
        if len(_st['pairs']) >= min_pairs:
            _st['needs_solve'] = True

# ─── Rendering ────────────────────────────────────────────────────────────────

def render(bgr, pts3d, K, D, R, t, frame_idx, n_frames, err_px=float('inf')):
    canvas = bgr.copy()
    h, w   = canvas.shape[:2]

    pts2d, ranges, in_front = project(pts3d, K, D, R, t)

    # Clip to image bounds
    if len(pts2d):
        in_bounds  = ((pts2d[:, 0] >= 0) & (pts2d[:, 0] < w) &
                      (pts2d[:, 1] >= 0) & (pts2d[:, 1] < h))
        vis2d      = pts2d[in_bounds].astype(np.int32)
        vis_ranges = ranges[in_bounds]
        vis3d      = pts3d[in_front][in_bounds]
    else:
        vis2d = np.zeros((0, 2), np.int32)
        vis_ranges = np.zeros(0, np.float32)
        vis3d      = np.zeros((0, 3), np.float32)

    # Apply max-range filter: hide points beyond the slider value so the full
    # colour spectrum stretches across only the visible depth band, making it
    # easier to distinguish features when picking point correspondences.
    max_range = max(1, cv2.getTrackbarPos(_TB_RANGE, WIN))
    if len(vis2d):
        in_range   = vis_ranges <= max_range
        vis2d      = vis2d[in_range]
        vis_ranges = vis_ranges[in_range]
        vis3d      = vis3d[in_range]

    _st['proj2d'] = vis2d.astype(np.float32)
    _st['proj3d'] = vis3d

    # Draw LiDAR dots — JET colormap stretched across 0..max_range
    if len(vis2d):
        norm    = np.clip(vis_ranges / max_range, 0.0, 1.0)
        lut_idx = (norm * 255).astype(np.uint8).reshape(-1, 1, 1)
        colors  = cv2.applyColorMap(lut_idx, cv2.COLORMAP_JET).reshape(-1, 3)
        for (px, py), (b, g, r_) in zip(vis2d, colors):
            cv2.circle(canvas, (int(px), int(py)), 2,
                       (int(b), int(g), int(r_)), -1, cv2.LINE_AA)

    # Draw completed pairs: camera point (yellow cross) + reprojected LiDAR (cyan circle)
    # with a residual line between them so the alignment error is visible
    if _st['pairs']:
        obj_pts = np.array([p[0] for p in _st['pairs']], dtype=np.float32)
        rvec, _ = cv2.Rodrigues(R.astype(np.float64))
        reproj, _ = cv2.projectPoints(obj_pts, rvec, t.reshape(3, 1).astype(np.float64),
                                       K.astype(np.float64), D.astype(np.float64))
        reproj = reproj.reshape(-1, 2)
        for i, ((_, pt2d_cam), rp) in enumerate(zip(_st['pairs'], reproj)):
            cx, cy = int(pt2d_cam[0]), int(pt2d_cam[1])
            rx_, ry_ = int(rp[0]), int(rp[1])
            # Residual line (red = large error, green = small error)
            err = float(np.linalg.norm(pt2d_cam - rp))
            err_color = (0, int(255 * max(0, 1 - err / 20)), int(255 * min(1, err / 20)))
            cv2.line(canvas, (cx, cy), (rx_, ry_), err_color, 1, cv2.LINE_AA)
            # Camera observation: yellow cross
            cv2.drawMarker(canvas, (cx, cy), (0, 255, 255),
                           cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
            # Reprojected LiDAR: cyan circle
            cv2.circle(canvas, (rx_, ry_), 4, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(canvas, str(i + 1), (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

    # Pending camera click waiting for its lidar pair — draw as a red cross
    if _st['pending_cam'] is not None:
        pcx, pcy = int(_st['pending_cam'][0]), int(_st['pending_cam'][1])
        cv2.drawMarker(canvas, (pcx, pcy), (0, 0, 255),
                       cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
        cv2.putText(canvas, "click LiDAR", (pcx + 8, pcy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)

    # Custom crosshair cursor — replaces the OS pointer in PICK mode.
    # Green = waiting for camera click, orange = waiting for LiDAR click.
    if _st['mode'] == 'PICK':
        mx, my = _st['mouse_xy']
        if _st['pick_state'] == 'cam':
            xhair_color = (0, 230, 0)      # green  — click the camera feature
        else:
            xhair_color = (0, 140, 255)    # orange — click the LiDAR dot
        arm = 18
        gap = 4
        cv2.line(canvas, (mx - arm, my), (mx - gap, my), xhair_color, 1, cv2.LINE_AA)
        cv2.line(canvas, (mx + gap, my), (mx + arm, my), xhair_color, 1, cv2.LINE_AA)
        cv2.line(canvas, (mx, my - arm), (mx, my - gap), xhair_color, 1, cv2.LINE_AA)
        cv2.line(canvas, (mx, my + gap), (mx, my + arm), xhair_color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (mx, my), gap, xhair_color, 1, cv2.LINE_AA)

    # HUD overlay
    rx, ry, rz = R_to_euler(R)
    tx, ty, tz  = t[0], t[1], t[2]
    n_pairs  = len(_st['pairs'])
    MIN_PAIRS = 4 if _st['has_loaded_calib'] else 6
    if _st['mode'] == 'PICK':
        pick_hint = ("→ click CAMERA feature" if _st['pick_state'] == 'cam'
                     else "→ click LiDAR dot")
        if n_pairs < MIN_PAIRS:
            pair_str = (f"Pairs: {n_pairs}/{MIN_PAIRS}  "
                        f"({MIN_PAIRS - n_pairs} more to first solve)")
        else:
            pair_str = f"Pairs: {n_pairs}  (auto-solving each new pair)"
        mode_str = f"PICK {pick_hint}   {pair_str}"
    else:
        mode_str = f"ADJUST   Pairs: {n_pairs}"
    err_color = ((0, 220, 0)   if err_px < 5
                 else (0, 180, 255) if err_px < 15
                 else (0, 80, 255))
    err_str   = f'{err_px:.1f} px' if err_px < 1e6 else '---'

    hud = [
        f"Frame {frame_idx+1}/{n_frames}   Mode: {mode_str}",
        f"t = ({tx:+.3f}, {ty:+.3f}, {tz:+.3f}) m",
        f"r = ({rx:+.1f}, {ry:+.1f}, {rz:+.1f}) deg     reproj err: {err_str}     range: {max_range} m",
        "n/b=frame  a=adjust  p=pick  s=solve  u=undo  c=clear  r=reset  w=save  q=quit",
    ]
    for i, line in enumerate(hud):
        y0 = 42 + i * 40
        cv2.putText(canvas, line, (8, y0), cv2.FONT_HERSHEY_SIMPLEX,
                    0.95, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(canvas, line, (8, y0), cv2.FONT_HERSHEY_SIMPLEX,
                    0.95, (0, 0, 0), 2, cv2.LINE_AA)

    # ── Value strip appended below the image ──────────────────────────────────
    strip_h = 128
    strip   = np.full((strip_h, w, 3), 30, dtype=np.uint8)
    cv2.line(strip, (0, 0), (w, 0), (70, 70, 70), 1)

    # Row 1 — extrinsic labels
    dof_labels = [
        (f'tx  {tx:+.3f} m',  (180, 220, 255)),
        (f'ty  {ty:+.3f} m',  (180, 220, 255)),
        (f'tz  {tz:+.3f} m',  (180, 220, 255)),
        (f'rx  {rx:+.1f}°',  (180, 255, 200)),
        (f'ry  {ry:+.1f}°',  (180, 255, 200)),
        (f'rz  {rz:+.1f}°',  (180, 255, 200)),
    ]
    col_w = w // len(dof_labels)
    for ci, (text, color) in enumerate(dof_labels):
        xpos = ci * col_w + 8
        cv2.putText(strip, text, (xpos, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(strip, text, (xpos, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, color,     2, cv2.LINE_AA)

    # Row 2 — extrinsic zero-indicator bars
    for ci, val in enumerate([tx, ty, tz, rx / 180, ry / 180, rz / 180]):
        xpos   = ci * col_w + 8
        bar_w  = col_w - 16
        centre = xpos + bar_w // 2
        cv2.line(strip, (xpos, 56), (xpos + bar_w, 56), (70, 70, 70), 2)
        cv2.line(strip, (centre, 50), (centre, 62), (110, 110, 110), 1)
        filled = int(np.clip(val, -1, 1) * (bar_w // 2))
        bar_color = (100, 200, 100) if abs(val) < 0.5 else (80, 80, 220)
        if filled != 0:
            cv2.line(strip, (centre, 56), (centre + filled, 56), bar_color, 6)
        dot_x = int(np.clip(centre + filled, xpos, xpos + bar_w))
        cv2.circle(strip, (dot_x, 56), 6, (255, 255, 255), -1)

    # Separator
    cv2.line(strip, (0, 74), (w, 74), (60, 60, 60), 1)

    # Row 3 — distortion coefficients (read-only, updated automatically by solver)
    d = D.flatten()
    d5 = np.zeros(5); d5[:min(len(d), 5)] = d[:5]
    dist_labels = [
        (f'k1  {d5[0]:+.4f}', (220, 200, 255)),
        (f'k2  {d5[1]:+.4f}', (220, 200, 255)),
        (f'p1  {d5[2]:+.5f}', (255, 220, 180)),
        (f'p2  {d5[3]:+.5f}', (255, 220, 180)),
        (f'k3  {d5[4]:+.4f}', (220, 200, 255)),
    ]
    dcol_w = w // len(dist_labels)
    for ci, (text, color) in enumerate(dist_labels):
        xpos = ci * dcol_w + 8
        cv2.putText(strip, text, (xpos, 104),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(strip, text, (xpos, 104),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, color,     2, cv2.LINE_AA)
    cv2.putText(strip, '← distortion (auto-optimised)', (w - 320, 122),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 100, 100), 1, cv2.LINE_AA)

    canvas = np.vstack([canvas, strip])
    return canvas

# ─── PnP solver ───────────────────────────────────────────────────────────────

def _mean_reproj_error(pairs, K, D, R, t):
    """Return mean reprojection error in pixels for all pairs."""
    if not pairs:
        return float('inf')
    obj_pts = np.array([p[0] for p in pairs], dtype=np.float32)
    img_pts = np.array([p[1] for p in pairs], dtype=np.float32)
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    proj, _ = cv2.projectPoints(obj_pts, rvec, t.reshape(3, 1).astype(np.float64),
                                 K.astype(np.float64), D.astype(np.float64))
    errs = np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1)
    return float(errs.mean())

def _joint_residuals(params, obj_pts, img_pts, K):
    """Reprojection residual vector for Levenberg-Marquardt over [rvec, tvec, D]."""
    rvec = params[:3].reshape(3, 1)
    tvec = params[3:6].reshape(3, 1)
    D    = params[6:11]
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec,
                                 K.astype(np.float64), D.astype(np.float64))
    return (proj.reshape(-1, 2) - img_pts).flatten()


def solve_pnp(pairs, K, D, R_init=None, t_init=None):
    """
    Solve for R, t, D (LiDAR→camera) from point pairs.

    Pass 1 — RANSAC: robust initial R, t, outlier rejection.
    Pass 2 — solvePnP iterative: refine R, t on inliers.
    Pass 3 — Levenberg-Marquardt: jointly optimise R, t, and D on inliers
              to minimise reprojection error.  D is only updated on the
              inlier set so outlier pairs cannot corrupt it.

    Returns (R, t, D) all as float64, or (None, None, D_unchanged) on failure.
    """
    D = np.asarray(D, dtype=np.float64).flatten()
    # Pad / trim to exactly 5 coefficients [k1, k2, p1, p2, k3]
    D5 = np.zeros(5, dtype=np.float64)
    D5[:min(len(D), 5)] = D[:5]

    obj_pts = np.array([p[0] for p in pairs], dtype=np.float32)
    img_pts = np.array([p[1] for p in pairs], dtype=np.float32)

    # Pass 1 — RANSAC with a generous reprojection threshold
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts, img_pts, K.astype(np.float64), D5,
        iterationsCount=3000, reprojectionError=12.0,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        print("  PnP RANSAC failed — try adding more spread-out pairs")
        return None, None, D5

    inlier_idx = inliers.flatten() if inliers is not None else np.arange(len(pairs))

    # Pass 2 — iterative refinement on inliers, seeded with current estimate
    if len(inlier_idx) >= 4:
        rvec_seed = rvec
        tvec_seed = tvec
        if R_init is not None and t_init is not None:
            rvec_seed, _ = cv2.Rodrigues(R_init.astype(np.float64))
            tvec_seed    = t_init.reshape(3, 1).astype(np.float64)
        ok2, rvec2, tvec2 = cv2.solvePnP(
            obj_pts[inlier_idx], img_pts[inlier_idx],
            K.astype(np.float64), D5,
            rvec=rvec_seed, tvec=tvec_seed,
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok2:
            rvec, tvec = rvec2, tvec2

    # Pass 3 — jointly optimise [rvec(3), tvec(3), D(5)] with LM on inliers
    try:
        from scipy.optimize import least_squares
        x0 = np.concatenate([rvec.flatten(), tvec.flatten(), D5])
        res = least_squares(
            _joint_residuals, x0,
            args=(obj_pts[inlier_idx].astype(np.float64),
                  img_pts[inlier_idx].astype(np.float64),
                  K),
            method='lm', max_nfev=4000)
        cost_before = float(np.sum(np.square(_joint_residuals(
            x0, obj_pts[inlier_idx].astype(np.float64),
            img_pts[inlier_idx].astype(np.float64), K))))
        if res.cost < cost_before:          # only accept if it actually improved
            rvec = res.x[:3].reshape(3, 1)
            tvec = res.x[3:6].reshape(3, 1)
            D5   = res.x[6:11]
    except ImportError:
        print("  (scipy not available — skipping joint R/t/D refinement)")
    except Exception as e:
        print(f"  (joint optimisation skipped: {e})")

    R, _ = cv2.Rodrigues(rvec)
    t    = tvec.flatten()
    n_in = len(inlier_idx)
    err  = _mean_reproj_error(pairs, K, D5, R, t)
    rx, ry, rz = R_to_euler(R)
    print(f"  PnP solved  inliers={n_in}/{len(pairs)}  "
          f"mean_reproj={err:.1f} px")
    print(f"    t  = ({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}) m")
    print(f"    r  = ({rx:+.2f}°, {ry:+.2f}°, {rz:+.2f}°)")
    print(f"    D  = [{D5[0]:+.4f}, {D5[1]:+.4f}, {D5[2]:+.5f}, "
          f"{D5[3]:+.5f}, {D5[4]:+.4f}]")
    return R, t, D5

# ─── Topic picker ─────────────────────────────────────────────────────────────

def _pick(topics: dict, label: str, substr: str) -> str:
    candidates = sorted(t for t, mt in topics.items()
                        if substr.lower() in mt.lower())
    if not candidates:
        candidates = sorted(topics)
    print(f"\nSelect {label}:")
    for i, t in enumerate(candidates):
        print(f"  [{i}]  {t}   ({topics[t]})")
    while True:
        raw = input(f"  Choice [0–{len(candidates)-1}]: ").strip()
        if raw in topics:
            return raw
        try:
            idx = int(raw)
            if 0 <= idx < len(candidates):
                return candidates[idx]
        except ValueError:
            pass

# ─── Load / Save calibration ──────────────────────────────────────────────────

def load_calibration(path: str):
    """
    Load a previously saved calibration.yaml.
    Restores R, t, D and point pairs.
    Returns (R, t, D) as float64 arrays, or (None, None, None) on error.
    """
    try:
        with open(path) as f:
            doc = yaml.safe_load(f)
        R = np.array(doc['R_cam_from_lidar']['data'],
                     dtype=np.float64).reshape(3, 3)
        t = np.array(doc['t_cam_from_lidar']['data'],
                     dtype=np.float64).flatten()
        D = np.array(doc['dist_coeffs']['data'],
                     dtype=np.float64).flatten()

        # Restore saved point pairs if present
        raw_pairs = doc.get('point_pairs', [])
        if raw_pairs:
            _st['pairs'].clear()
            for entry in raw_pairs:
                pt3d = np.array(entry['pt3d'], dtype=np.float32)
                pt2d = np.array(entry['pt2d'], dtype=np.float32)
                _st['pairs'].append((pt3d, pt2d))
            print(f"  Loaded {len(_st['pairs'])} point pairs from {path}")

        print(f"  Loaded calibration from {path}")
        rx, ry, rz = R_to_euler(R)
        print(f"    t = ({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}) m")
        print(f"    r = ({rx:+.2f}°, {ry:+.2f}°, {rz:+.2f}°)")
        print(f"    D = {D.tolist()}")
        return R, t, D
    except Exception as e:
        print(f"  Could not load {path}: {e}")
        return None, None, None

def save(path: str, K: np.ndarray, D: np.ndarray, R: np.ndarray, t: np.ndarray,
         img_w: int, img_h: int):
    rx, ry, rz = R_to_euler(R)
    doc = {
        # Resolution the K matrix was calibrated at — scale K proportionally for
        # other resolutions; R, t, and D are resolution-independent.
        'image_width':      img_w,
        'image_height':     img_h,
        'camera_matrix':    {'rows': 3, 'cols': 3, 'data': K.flatten().tolist()},
        'dist_coeffs':      {'rows': 1, 'cols': int(D.size), 'data': D.tolist()},
        'R_cam_from_lidar': {'rows': 3, 'cols': 3, 'data': R.flatten().tolist()},
        't_cam_from_lidar': {'rows': 3, 'cols': 1, 'data': t.flatten().tolist()},
        'euler_deg_xyz':    {'rx': float(rx), 'ry': float(ry), 'rz': float(rz)},
        'point_pairs': [
            {'pt3d': p[0].tolist(), 'pt2d': p[1].tolist()}
            for p in _st['pairs']
        ],
    }
    with open(path, 'w') as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    print(f"  Saved → {path}  ({img_w}×{img_h}, {len(_st['pairs'])} point pairs)")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Camera-LiDAR extrinsic calibration tool')
    ap.add_argument('camera_bag',      help='Bag containing camera images (ROS1 .bag or ROS2 dir)')
    ap.add_argument('lidar_bag',       nargs='?', default=None,
                    help='Bag containing LiDAR data (omit if same as camera_bag)')
    ap.add_argument('--image',         help='Image topic')
    ap.add_argument('--lidar',         help='LiDAR PointCloud2 topic')
    ap.add_argument('--info',          help='CameraInfo topic')
    ap.add_argument('--load',          default=None,
                    help='Load a previous calibration.yaml to use as starting point')
    ap.add_argument('--out',           default='calibration.yaml')
    ap.add_argument('--frames', '-f',  type=int, default=60,
                    help='Max frames to load (default 60)')
    args = ap.parse_args()

    lidar_bag = args.lidar_bag or args.camera_bag   # same bag if not split

    print(f"\nCamera bag: {args.camera_bag}")
    cam_topics, is_ros1 = list_topics(args.camera_bag)
    print(f"  Format : {'ROS1' if is_ros1 else 'ROS2'}")
    for t in sorted(cam_topics):
        print(f"    {t:<50}  {cam_topics[t]}")

    if lidar_bag != args.camera_bag:
        print(f"\nLiDAR bag:  {lidar_bag}")
        lidar_topics, _ = list_topics(lidar_bag)
        for t in sorted(lidar_topics):
            print(f"    {t:<50}  {lidar_topics[t]}")
    else:
        lidar_topics = cam_topics

    img_topic   = args.image or _pick(cam_topics,   'camera image topic',      'Image')
    lidar_topic = args.lidar or _pick(lidar_topics, 'LiDAR PointCloud2 topic', 'PointCloud2')
    info_topic  = args.info  or _pick(cam_topics,   'CameraInfo topic',        'CameraInfo')

    print(f"\n  image : {img_topic}")
    print(f"  lidar : {lidar_topic}")
    print(f"  info  : {info_topic}")

    frames = load_frames(args.camera_bag, lidar_bag,
                         img_topic, lidar_topic, info_topic, args.frames)
    if not frames:
        print("No synced frames found — check topic names and bag contents.")
        sys.exit(1)

    K, D = frames[0][2], frames[0][3]
    _st['K'], _st['D'] = K, D   # make available to auto-solve

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    img_h, img_w = frames[0][0].shape[:2]
    cv2.resizeWindow(WIN, min(1280, img_w), min(900, img_h + 130))
    make_trackbars()
    cv2.setMouseCallback(WIN, _on_mouse)

    frame_idx = 0
    R_cur = np.eye(3, dtype=np.float64)
    t_cur = np.zeros(3, dtype=np.float64)
    D_cur = frames[0][3].copy()   # start from camera_info D (possibly zeroed if rectified)

    if args.load:
        R_loaded, t_loaded, D_loaded = load_calibration(args.load)
        if R_loaded is not None:
            R_cur, t_cur = R_loaded, t_loaded
            D_cur = D_loaded          # restore previously-optimised D
            rx0, ry0, rz0 = R_to_euler(R_cur)
            set_trackbars(t_cur[0], t_cur[1], t_cur[2], rx0, ry0, rz0)
            _st['has_loaded_calib'] = True
            if len(_st['pairs']) >= 4:
                print(f"  Re-solving with {len(_st['pairs'])} loaded pairs...")
                _st['needs_solve'] = True

    print("\nReady.  Adjust trackbars to align the LiDAR overlay with the image.")
    _xlib_init()   # connect to X11 for cursor control (no-op if unavailable)

    while True:
        bgr, pts3d, K, _ = frames[frame_idx]   # D comes from D_cur, not frame

        # Auto-solve BEFORE render so the updated calibration is visible immediately
        if _st['needs_solve']:
            _st['needs_solve'] = False
            R_sol, t_sol, D_sol = solve_pnp(_st['pairs'], K, D_cur, R_cur, t_cur)
            if R_sol is not None:
                rx_s, ry_s, rz_s = R_to_euler(R_sol)
                set_trackbars(t_sol[0], t_sol[1], t_sol[2], rx_s, ry_s, rz_s)
                R_cur, t_cur, D_cur = R_sol, t_sol, D_sol

        # Read extrinsic trackbars — manual slider adjustments override solver
        tx, ty, tz, rx, ry, rz = read_trackbars()
        R_cur = euler_to_R(rx, ry, rz)
        t_cur = np.array([tx, ty, tz], dtype=np.float64)

        err_px = _mean_reproj_error(_st['pairs'], K, D_cur, R_cur, t_cur)
        canvas = render(bgr, pts3d, K, D_cur, R_cur, t_cur, frame_idx, len(frames), err_px)
        cv2.imshow(WIN, canvas)

        # Re-assert cursor every frame — OpenCV's zoom/pan mode overrides it
        # with a hand cursor, so we push our preferred shape back each tick.
        if _st['mode'] == 'PICK':
            _set_cursor(_XC_CROSSHAIR + 1 if _st['pick_state'] == 'lid'
                        else _XC_CROSSHAIR)
        else:
            _set_cursor(_XC_ARROW)

        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('n'):
            frame_idx = (frame_idx + 1) % len(frames)
        elif key == ord('b'):
            frame_idx = (frame_idx - 1) % len(frames)
        elif key == ord('a'):
            _st['mode'] = 'ADJUST'
            _st['pending_cam'] = None
            _st['pick_state']  = 'cam'
            _set_cursor(_XC_ARROW)
            print("→ ADJUST mode")
        elif key == ord('p'):
            _st['mode'] = 'PICK'
            _st['pick_state'] = 'cam'
            _set_cursor(_XC_CROSSHAIR)
            print("→ PICK mode  (green crosshair = click camera feature,"
                  " orange = click LiDAR dot)")
        elif key == ord('u'):
            if _st['pairs']:
                _st['pairs'].pop()
                _st['pending_cam'] = None
                _st['pick_state']  = 'cam'
                if _st['mode'] == 'PICK':
                    _set_cursor(_XC_CROSSHAIR)
                n = len(_st['pairs'])
                print(f"  Undo — {n} pair(s) remaining")
                min_pairs = 4 if _st['has_loaded_calib'] else 6
                if n >= min_pairs:
                    _st['needs_solve'] = True
            else:
                print("  Nothing to undo")
        elif key == ord('c'):
            _st['pairs'].clear()
            _st['pending_cam'] = None
            _st['pick_state']  = 'cam'
            if _st['mode'] == 'PICK':
                _set_cursor(_XC_CROSSHAIR)   # back to cam-pick crosshair (green)
            print("  Pairs cleared")
        elif key == ord('r'):
            set_trackbars(0, 0, 0, 0, 0, 0)
            print("  Extrinsic reset to zero")
        elif key == ord('s'):
            min_pairs = 4 if _st['has_loaded_calib'] else 6
            if len(_st['pairs']) < min_pairs:
                print(f"  Need ≥ {min_pairs} pairs, have {len(_st['pairs'])}")
            else:
                R_sol, t_sol, D_sol = solve_pnp(_st['pairs'], K, D_cur, R_cur, t_cur)
                if R_sol is not None:
                    rx_s, ry_s, rz_s = R_to_euler(R_sol)
                    set_trackbars(t_sol[0], t_sol[1], t_sol[2], rx_s, ry_s, rz_s)
                    R_cur, t_cur, D_cur = R_sol, t_sol, D_sol
        elif key == ord('w'):
            img_h, img_w = frames[frame_idx][0].shape[:2]
            save(args.out, K, D_cur, R_cur, t_cur, img_w, img_h)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

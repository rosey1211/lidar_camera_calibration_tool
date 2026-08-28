#!/usr/bin/env python3
"""
verify_calibration.py
=====================
Display LiDAR points projected onto camera images using a saved calibration.
Use the resolution trackbar to test how calibration_utils.py scales K for
different image sizes while keeping R, t, and D unchanged.

Usage:
  python verify_calibration.py <camera_bag> <lidar_bag> --calibration calibration.yaml
  python verify_calibration.py <camera_bag> <lidar_bag> --calibration calibration.yaml \
      --image /cam/image_raw --lidar /lidar/points --info /cam/camera_info

Controls:
  Resolution trackbar — scale the image (and K) from 25 % to 200 %
  n / b              — next / previous frame
  q / Esc            — quit
"""

import sys
import argparse
import numpy as np
import cv2
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from calibration_utils import load_calibration, scale_K

# ─── Bag reading (minimal, shared logic with calibrate.py) ────────────────────

def _is_ros1(path):
    return Path(path).is_file() and Path(path).suffix == '.bag'

def _make_reader_and_store(path):
    from rosbags.typesys import get_typestore, Stores
    if _is_ros1(path):
        from rosbags.rosbag1 import Reader
        return Reader(path), get_typestore(Stores.ROS1_NOETIC), True
    else:
        from rosbags.rosbag2 import Reader
        return Reader(path), get_typestore(Stores.ROS2_HUMBLE), False

def iter_messages(path, wanted):
    reader, typestore, is_ros1 = _make_reader_and_store(path)
    with reader:
        conns = [c for c in reader.connections if c.topic in wanted]
        for conn, ts, raw in reader.messages(connections=conns):
            msg = (typestore.deserialize_ros1(raw, conn.msgtype) if is_ros1
                   else typestore.deserialize_cdr(raw, conn.msgtype))
            yield conn.topic, ts, msg

def list_topics(path):
    reader, _, _ = _make_reader_and_store(path)
    topics = {}
    with reader:
        for conn in reader.connections:
            topics[conn.topic] = conn.msgtype
    return topics

# ─── Message decoders ─────────────────────────────────────────────────────────

_PC2_DTYPES = {1:'i1',2:'u1',3:'i2',4:'u2',5:'i4',6:'u4',7:'f4',8:'f8'}

def pc2_to_xyz(msg):
    dt = np.dtype({
        'names':   [f.name for f in msg.fields],
        'formats': [_PC2_DTYPES.get(f.datatype, 'u1') for f in msg.fields],
        'offsets': [f.offset for f in msg.fields],
        'itemsize': msg.point_step,
    })
    raw = np.frombuffer(bytes(msg.data), dtype=dt)
    pts = np.stack([raw['x'].astype(np.float32),
                    raw['y'].astype(np.float32),
                    raw['z'].astype(np.float32)], axis=1)
    return pts[np.isfinite(pts).all(axis=1)]

def decode_image(msg):
    if hasattr(msg, 'format'):
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
    bayer = {'bayer_rggb8': cv2.COLOR_BayerRGGB2BGR,
             'bayer_bggr8': cv2.COLOR_BayerBGGR2BGR,
             'bayer_gbrg8': cv2.COLOR_BayerGBRG2BGR,
             'bayer_grbg8': cv2.COLOR_BayerGRBG2BGR}
    if enc in bayer:
        return cv2.cvtColor(raw.reshape(h, w), bayer[enc])
    raise ValueError(f"Unsupported encoding: {msg.encoding}")

def parse_camera_info(msg):
    return (np.array(msg.K, dtype=np.float64).reshape(3, 3),
            np.array(msg.D, dtype=np.float64).flatten())

# ─── Frame loading ────────────────────────────────────────────────────────────

def _pick(topics, label, substr):
    candidates = sorted(t for t, mt in topics.items()
                        if substr.lower() in mt.lower()) or sorted(topics)
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

def load_frames(cam_path, lidar_path, img_topic, lidar_topic, info_topic,
                max_frames=60):
    K = D = None
    imgs, clouds = {}, {}

    print(f"  Scanning camera bag: {cam_path}")
    for topic, ts, msg in iter_messages(cam_path, {img_topic, info_topic}):
        if topic == info_topic and K is None:
            K, D = parse_camera_info(msg)
        elif topic == img_topic:
            imgs[ts] = decode_image(msg)

    print(f"  Scanning lidar  bag: {lidar_path}")
    for topic, ts, msg in iter_messages(lidar_path, {lidar_topic}):
        clouds[ts] = pc2_to_xyz(msg)

    print(f"  {len(imgs)} images, {len(clouds)} clouds")
    if K is None:
        raise RuntimeError("No camera_info found — check --info topic.")

    img_ts   = sorted(imgs)
    cloud_ts = np.array(sorted(clouds), dtype=np.int64)
    frames   = []
    for it in img_ts:
        if len(frames) >= max_frames:
            break
        diffs = np.abs(cloud_ts - np.int64(it))
        ci    = int(np.argmin(diffs))
        if diffs[ci] > 200_000_000:
            continue
        frames.append((imgs[it], clouds[cloud_ts[ci]]))

    print(f"  {len(frames)} synced frames")
    return frames, K, D

# ─── Projection ───────────────────────────────────────────────────────────────

def project_onto(bgr, pts3d, K, D, R, t, max_range=30.0):
    """Return a copy of bgr with LiDAR dots overlaid, coloured by range."""
    h, w = bgr.shape[:2]

    pts_cam = (R @ pts3d.T + t.reshape(3, 1)).T
    in_front = pts_cam[:, 2] > 0.1
    if not in_front.any():
        return bgr.copy()

    front    = pts3d[in_front].astype(np.float32)
    ranges   = np.linalg.norm(pts_cam[in_front], axis=1).astype(np.float32)
    rvec, _  = cv2.Rodrigues(R.astype(np.float64))
    pts2d, _ = cv2.projectPoints(front, rvec, t.reshape(3,1).astype(np.float64),
                                  K.astype(np.float64), D.astype(np.float64))
    pts2d = pts2d.reshape(-1, 2)

    in_bounds = ((pts2d[:, 0] >= 0) & (pts2d[:, 0] < w) &
                 (pts2d[:, 1] >= 0) & (pts2d[:, 1] < h))
    pts2d  = pts2d[in_bounds].astype(np.int32)
    ranges = ranges[in_bounds]

    canvas = bgr.copy()
    if len(pts2d):
        norm    = np.clip(ranges / max_range, 0.0, 1.0)
        lut_idx = (norm * 255).astype(np.uint8).reshape(-1, 1, 1)
        colors  = cv2.applyColorMap(lut_idx, cv2.COLORMAP_JET).reshape(-1, 3)
        for (px, py), (b, g, r_) in zip(pts2d, colors):
            cv2.circle(canvas, (int(px), int(py)), 2,
                       (int(b), int(g), int(r_)), -1, cv2.LINE_AA)
    return canvas

# ─── Info strip ──────────────────────────────────────────────────────────────

_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_FS    = 0.75   # font scale for bar labels and value readouts
_FT    = 2      # thickness to match larger font
_LBL_W = 240    # pixels reserved for the left-side label
_VAL_W = 100    # pixels reserved for the right-side value readout
_PAD   = 12     # left/right outer margin
_ROW_H = 44     # vertical space per bar row


def _draw_bar_row(strip, y_top, label, val_min, val_max, current_val, unit):
    """One row: label | bar | value.  All sizes are fixed pixels."""
    h, w = strip.shape[:2]
    bar_x0 = _PAD + _LBL_W
    bar_x1 = w - _PAD - _VAL_W
    bar_y  = y_top + _ROW_H // 2

    # Left label
    cv2.putText(strip, label, (_PAD, bar_y + 7),
                _FONT, _FS, (180, 180, 180), _FT, cv2.LINE_AA)

    # Bar line
    cv2.line(strip, (bar_x0, bar_y), (bar_x1, bar_y), (90, 90, 90), 3)

    # Current-value position marker
    frac = (current_val - val_min) / (val_max - val_min)
    cx   = bar_x0 + int(frac * (bar_x1 - bar_x0))
    cx   = int(np.clip(cx, bar_x0, bar_x1))
    cv2.line(strip, (cx, bar_y - 11), (cx, bar_y + 11), (0, 220, 255), 3)

    # Right-side value readout
    cv2.putText(strip, f"{current_val}{unit}", (bar_x1 + 10, bar_y + 7),
                _FONT, _FS, (0, 220, 255), _FT, cv2.LINE_AA)


def draw_info_strip(width, scale_pct, max_range, frame_idx, n_frames,
                    new_w, new_h, K_scaled):
    """Fixed-height strip below the image showing bar readouts and K info."""
    h     = _ROW_H * 2 + 36   # two bar rows + bottom info line
    strip = np.full((h, width, 3), 28, dtype=np.uint8)
    cv2.line(strip, (0, 0), (width, 0), (70, 70, 70), 1)

    _draw_bar_row(strip, y_top=2,
                  label='Image resolution %:',
                  val_min=10, val_max=200, current_val=scale_pct, unit='%')

    _draw_bar_row(strip, y_top=2 + _ROW_H,
                  label='Max LiDAR range:',
                  val_min=5,  val_max=80,  current_val=max_range,  unit=' m')

    # Bottom info line: current resolution + scaled K values
    info = (f"{new_w}x{new_h} px   "
            f"fx={K_scaled[0,0]:.1f}  fy={K_scaled[1,1]:.1f}  "
            f"cx={K_scaled[0,2]:.1f}  cy={K_scaled[1,2]:.1f}   "
            f"n/b = frame    q = quit")
    cv2.putText(strip, info, (_PAD, h - 10),
                _FONT, 0.52, (150, 150, 150), 1, cv2.LINE_AA)

    return strip


# ─── Main ─────────────────────────────────────────────────────────────────────

WIN        = 'Calibration Verification'
TB_SCALE   = 'Resolution %  (100 = original)'
TB_RANGE   = 'Max range (m)'

def main():
    ap = argparse.ArgumentParser(description='Verify camera-LiDAR calibration')
    ap.add_argument('camera_bag')
    ap.add_argument('lidar_bag',  nargs='?', default=None)
    ap.add_argument('--calibration', '-c', required=True,
                    help='calibration.yaml produced by calibrate.py')
    ap.add_argument('--image',  default=None)
    ap.add_argument('--lidar',  default=None)
    ap.add_argument('--info',   default=None)
    ap.add_argument('--frames', type=int, default=60)
    args = ap.parse_args()

    lidar_bag = args.lidar_bag or args.camera_bag

    # Load calibration at original resolution to get R, t, D and calib dims
    with open(args.calibration) as f:
        calib_doc = yaml.safe_load(f)
    calib_w = calib_doc.get('image_width')
    calib_h = calib_doc.get('image_height')
    R = np.array(calib_doc['R_cam_from_lidar']['data'], dtype=np.float64).reshape(3,3)
    t = np.array(calib_doc['t_cam_from_lidar']['data'], dtype=np.float64).flatten()
    D = np.array(calib_doc['dist_coeffs']['data'],      dtype=np.float64).flatten()
    K_full = np.array(calib_doc['camera_matrix']['data'], dtype=np.float64).reshape(3,3)
    print(f"Calibration: {args.calibration}")
    if calib_w:
        print(f"  Reference resolution: {calib_w}×{calib_h}")

    # Topic selection
    cam_topics   = list_topics(args.camera_bag)
    lidar_topics = list_topics(lidar_bag) if lidar_bag != args.camera_bag else cam_topics
    img_topic   = args.image or _pick(cam_topics,   'camera image topic', 'Image')
    lidar_topic = args.lidar or _pick(lidar_topics, 'LiDAR topic',        'PointCloud2')
    info_topic  = args.info  or _pick(cam_topics,   'CameraInfo topic',   'CameraInfo')

    frames, K_bag, D_bag = load_frames(args.camera_bag, lidar_bag,
                                       img_topic, lidar_topic, info_topic,
                                       args.frames)
    if not frames:
        print("No synced frames found.")
        sys.exit(1)

    orig_h, orig_w = frames[0][0].shape[:2]
    print(f"  Image size from bag: {orig_w}×{orig_h}")

    # ── GUI ──────────────────────────────────────────────────────────────────
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(1280, orig_w), min(960, orig_h + 100))
    cv2.createTrackbar(TB_SCALE, WIN, 100, 200, lambda _: None)
    cv2.setTrackbarMin(TB_SCALE, WIN, 10)
    cv2.createTrackbar(TB_RANGE, WIN, 30, 80, lambda _: None)
    cv2.setTrackbarMin(TB_RANGE, WIN, 5)

    frame_idx = 0
    print("\nn/b = next/prev frame   q = quit")

    while True:
        bgr, pts3d = frames[frame_idx]
        scale_pct  = cv2.getTrackbarPos(TB_SCALE, WIN)
        max_range  = cv2.getTrackbarPos(TB_RANGE, WIN)

        scale = scale_pct / 100.0
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))

        # Resize the image to the target resolution
        if scale != 1.0:
            display_bgr = cv2.resize(bgr, (new_w, new_h),
                                     interpolation=(cv2.INTER_AREA   if scale < 1.0
                                                    else cv2.INTER_LINEAR))
        else:
            display_bgr = bgr

        # Scale K proportionally — use scale_K so the same logic used by
        # downstream code is exercised.  Reference dims come from the YAML when
        # available (most accurate); fall back to bag image size otherwise.
        ref_w = calib_w or orig_w
        ref_h = calib_h or orig_h
        K_scaled = scale_K(K_full, ref_w, ref_h, new_w, new_h)

        canvas = project_onto(display_bgr, pts3d, K_scaled, D, R, t, max_range)

        # Always show at original resolution so the window never resizes
        if new_w != orig_w or new_h != orig_h:
            canvas = cv2.resize(canvas, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Fixed HUD in top-left
        hud = f"Frame {frame_idx+1}/{len(frames)}   {scale_pct}%  {new_w}x{new_h}"
        cv2.putText(canvas, hud, (10, 38), _FONT, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, hud, (10, 38), _FONT, 1.0, (230, 230, 230), 2, cv2.LINE_AA)

        strip = draw_info_strip(orig_w, scale_pct, max_range,
                                frame_idx, len(frames), new_w, new_h, K_scaled)
        cv2.imshow(WIN, np.vstack([canvas, strip]))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('n'):
            frame_idx = (frame_idx + 1) % len(frames)
        elif key == ord('b'):
            frame_idx = (frame_idx - 1) % len(frames)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

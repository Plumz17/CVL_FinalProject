"""
SISTEM DETEKSI PELANGGARAN KECEPATAN KENDARAAN
YOLOv8 + ByteTrack + Homography (Bird's Eye View)

Tekan 'q' untuk berhenti, 'p' untuk pause.
"""

import cv2
import numpy as np
import math
import os
import datetime
from collections import defaultdict, deque
from ultralytics import YOLO


# ================================================================
#  ██████╗ ██████╗ ███╗   ██╗███████╗██╗ ██████╗
# ██╔════╝██╔═══██╗████╗  ██║██╔════╝██║██╔════╝
# ██║     ██║   ██║██╔██╗ ██║█████╗  ██║██║  ███╗
# ██║     ██║   ██║██║╚██╗██║██╔══╝  ██║██║   ██║
# ╚██████╗╚██████╔╝██║ ╚████║██║     ██║╚██████╔╝
#  ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝ ╚═════╝
# ================================================================

MODEL_PATH = 'Hasil_Training_CCTV/Model_V13/weights/best.pt'
VIDEO_PATH = 'data/pagisiang.3gp'
OUTPUT_DIR = 'violations_new3'

# Urutan: kiri-jauh, kanan-jauh, kanan-dekat, kiri-dekat
#   1 ──── 2
#   │      │
#   4 ──── 3
STATIC_ROI_PTS = [
    (422,  216),
    (745,  240),
    (1230, 721),
    (222,  718),
]

REAL_WIDTH_METER  = 10.0   # lebar ROI di lapangan (meter)
REAL_HEIGHT_METER = 30.0   # panjang ROI di lapangan (meter)

SPEED_LIMIT_KMH = 80

BEV_WIDTH  = 300
BEV_HEIGHT = 500

CONF_THRESHOLD   = 0.3
SPEED_WINDOW     = 6
MIN_SPEED_FRAMES = 6

VIOLATION_COOLDOWN_SEC = 5
MAX_ACCEL_KMH_PER_SEC  = 4    # lonjakan naik melebihi ini → jitter
MAX_DECEL_KMH_PER_SEC  = 8    # lonjakan turun melebihi ini → jitter

STABILITY_FRAMES    = 6       # harus stabil N frame berturut-turut sebelum bisa violation
STABILITY_DELTA_KMH = 8.0     # toleransi perubahan kecepatan antar frame untuk dianggap stabil

DISPLAY_WIDTH        = 1100
SAVE_VIOLATION_FRAME = True

PIXEL_PER_METER_X = BEV_WIDTH  / REAL_WIDTH_METER
PIXEL_PER_METER_Y = BEV_HEIGHT / REAL_HEIGHT_METER


# ================================================================
# HOMOGRAPHY
# ================================================================

def build_homography(roi_pts, bev_w, bev_h):
    src = np.float32(roi_pts)
    dst = np.float32([
        [0,     0    ],
        [bev_w, 0    ],
        [bev_w, bev_h],
        [0,     bev_h],
    ])
    H, _ = cv2.findHomography(src, dst)
    if H is None:
        raise ValueError("Gagal menghitung homography — 4 titik ROI tidak boleh collinear.")
    return H


def point_to_bev(pt, H):
    src = np.float32([[[pt[0], pt[1]]]])
    dst = cv2.perspectiveTransform(src, H)
    return float(dst[0][0][0]), float(dst[0][0][1])


# ================================================================
# SPEED TRACKER
# ================================================================

class VehicleSpeedTracker:
    def __init__(self, fps, speed_limit_kmh,
                 px_per_m_x, px_per_m_y,
                 speed_window=7, min_frames=3,
                 violation_cooldown_sec=5,
                 max_accel_kmh_per_sec=20.0,
                 max_decel_kmh_per_sec=20.0):
        self.fps             = fps
        self.speed_limit     = speed_limit_kmh
        self.px_per_m_x      = px_per_m_x
        self.px_per_m_y      = px_per_m_y
        self.speed_window    = speed_window
        self.min_frames      = min_frames
        self.cooldown_frames = int(violation_cooldown_sec * fps)
        self.max_accel       = max_accel_kmh_per_sec
        self.max_decel       = max_decel_kmh_per_sec

        self.pos_history: dict[int, deque]      = defaultdict(lambda: deque(maxlen=speed_window + 1))
        self.speed_history_short: dict[int, deque] = defaultdict(lambda: deque(maxlen=STABILITY_FRAMES))
        self.speed_kmh: dict[int, float]        = {}
        self.prev_speed_kmh: dict[int, float]   = {}
        self.violating: dict[int, bool]         = {}
        self.jitter_suppressed: dict[int, bool] = {}
        self.is_stable: dict[int, bool]         = {}
        self.last_viol_frame: dict[int, int]    = {}

    def update(self, track_id: int, bev_pos: tuple, frame_no: int):
        self.pos_history[track_id].append((frame_no, bev_pos[0], bev_pos[1]))
        hist = self.pos_history[track_id]

        if len(hist) < self.min_frames:
            self.speed_kmh[track_id] = 0.0
            return

        oldest, newest = hist[0], hist[-1]
        dt_frames = newest[0] - oldest[0]
        if dt_frames == 0:
            return

        dx_m   = (newest[1] - oldest[1]) / self.px_per_m_x
        dy_m   = (newest[2] - oldest[2]) / self.px_per_m_y
        dist_m = math.hypot(dx_m, dy_m)
        dt_sec = dt_frames / self.fps

        self.prev_speed_kmh[track_id] = self.speed_kmh.get(track_id, 0.0)
        self.speed_kmh[track_id]      = round((dist_m / dt_sec) * 3.6, 1)
        self._update_stability(track_id)

    def _update_stability(self, track_id: int):
        # Kendaraan dianggap stabil jika N frame terakhir perubahannya kecil.
        # Menghindari false violation dari kendaraan yang baru keluar oklusi.
        curr = self.speed_kmh.get(track_id, 0.0)
        self.speed_history_short[track_id].append(curr)
        hist = self.speed_history_short[track_id]

        if len(hist) < STABILITY_FRAMES:
            self.is_stable[track_id] = False
            return

        speeds = list(hist)
        deltas = [abs(speeds[i] - speeds[i-1]) for i in range(1, len(speeds))]
        self.is_stable[track_id] = all(d <= STABILITY_DELTA_KMH for d in deltas)

    def _is_jitter_spike(self, track_id: int) -> bool:
        # Lonjakan kecepatan naik terlalu cepat → bukan akselerasi nyata.
        hist = self.pos_history.get(track_id)
        if not hist or len(hist) < 2:
            return False

        prev    = self.prev_speed_kmh.get(track_id, 0.0)
        curr    = self.speed_kmh.get(track_id, 0.0)
        delta_v = curr - prev
        if delta_v <= 0 or prev == 0.0:
            return False

        dt_sec = (hist[-1][0] - hist[0][0]) / self.fps
        if dt_sec == 0:
            return False
        return (delta_v / dt_sec) > self.max_accel

    def _is_decel_spike(self, track_id: int) -> bool:
        # Penurunan kecepatan terlalu cepat → window masih ingat kecepatan lama.
        hist = self.pos_history.get(track_id)
        if not hist or len(hist) < 2:
            return False

        prev    = self.prev_speed_kmh.get(track_id, 0.0)
        curr    = self.speed_kmh.get(track_id, 0.0)
        delta_v = curr - prev
        if delta_v >= 0:
            return False

        dt_sec = (hist[-1][0] - hist[0][0]) / self.fps
        if dt_sec == 0:
            return False
        return (abs(delta_v) / dt_sec) > self.max_decel

    def check_violation(self, track_id: int, frame_no: int) -> tuple[bool, float]:
        speed = self.speed_kmh.get(track_id, 0.0)

        if speed > self.speed_limit:
            # Kecepatan belum konvergen — kendaraan baru muncul / keluar oklusi
            if not self.is_stable.get(track_id, False):
                self.jitter_suppressed[track_id] = True
                self.violating[track_id] = False
                print(f"  🔄 UNSTABLE  ID:{track_id}  {speed:.0f} km/h  frame:{frame_no}")
                return False, speed

            # Akselerasi tidak wajar
            if self._is_jitter_spike(track_id):
                self.jitter_suppressed[track_id] = True
                self.violating[track_id] = False
                prev = self.prev_speed_kmh.get(track_id, 0.0)
                print(f"  ⚡ JITTER (accel)  ID:{track_id}  {prev:.0f}→{speed:.0f} km/h  frame:{frame_no}")
                return False, speed

            # Deselerasi tidak wajar
            if self._is_decel_spike(track_id):
                self.jitter_suppressed[track_id] = True
                self.violating[track_id] = False
                prev = self.prev_speed_kmh.get(track_id, 0.0)
                print(f"  ⚡ JITTER (decel)  ID:{track_id}  {prev:.0f}→{speed:.0f} km/h  frame:{frame_no}")
                return False, speed

            self.jitter_suppressed[track_id] = False
            self.violating[track_id] = True
            last = self.last_viol_frame.get(track_id, -self.cooldown_frames * 10)
            if frame_no - last >= self.cooldown_frames:
                self.last_viol_frame[track_id] = frame_no
                return True, speed
            return False, speed

        else:
            self.violating[track_id]         = False
            self.jitter_suppressed[track_id] = False
            return False, speed

    def get_speed(self, track_id: int) -> float:
        return self.speed_kmh.get(track_id, 0.0)

    def is_violating(self, track_id: int) -> bool:
        return self.violating.get(track_id, False)

    def is_jitter_suppressed(self, track_id: int) -> bool:
        return self.jitter_suppressed.get(track_id, False)

    def is_speed_stable(self, track_id: int) -> bool:
        return self.is_stable.get(track_id, False)

    def evict_stale(self, active_ids: set, max_idle_frames=90):
        stale = set(self.pos_history) - active_ids
        for tid in stale:
            self.pos_history.pop(tid, None)
            self.speed_history_short.pop(tid, None)
            self.speed_kmh.pop(tid, None)
            self.prev_speed_kmh.pop(tid, None)
            self.violating.pop(tid, None)
            self.jitter_suppressed.pop(tid, None)
            self.is_stable.pop(tid, None)


# ================================================================
# VISUALISASI
# ================================================================

PALETTE = {
    "safe"      : (50,  205,  50),
    "overspeed" : (0,   60,  255),
    "jitter"    : (0,   165, 255),  # oranye — overspeed tapi di-suppress
    "roi_line"  : (0,   220, 220),
    "yellow"    : (0,   210, 255),
}


def draw_vehicle_box(frame, bbox, track_id, speed, violating, jitter_suppressed=False):
    x1, y1, x2, y2 = map(int, bbox)

    if jitter_suppressed:
        color = PALETTE["jitter"]
        thick = 2
        label = f"ID:{track_id}  {speed:.0f} km/h  ⚡ JITTER"
    elif violating:
        color = PALETTE["overspeed"]
        thick = 3
        label = f"ID:{track_id}  {speed:.0f} km/h  ⚠ OVERSPEED"
    elif speed > 0:
        color = PALETTE["safe"]
        thick = 2
        label = f"ID:{track_id}  {speed:.0f} km/h"
    else:
        color = PALETTE["safe"]
        thick = 2
        label = f"ID:{track_id}  --"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    lx, ly = x1, max(y1 - 4, th + 4)
    cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 6, ly + 2), color, -1)
    cv2.putText(frame, label, (lx + 3, ly - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_roi_overlay(frame, roi_pts):
    pts = np.array(roi_pts, np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (0, 220, 220))
    cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)
    cv2.polylines(frame, [pts], True, PALETTE["roi_line"], 2)
    for i, pt in enumerate(roi_pts):
        cv2.circle(frame, pt, 5, PALETTE["roi_line"], -1)
        cv2.putText(frame, str(i + 1), (pt[0] + 6, pt[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, PALETTE["yellow"], 2)


def draw_hud(frame, frame_no, fps, active_cnt, viol_total, speed_limit, total_vehicles):
    lines = [
        f"Frame    : {frame_no}",
        f"FPS      : {fps:.1f}",
        f"Di ROI   : {active_cnt} kendaraan",
        f"Total ID : {total_vehicles} kendaraan",
        f"Limit    : {speed_limit} km/h",
        f"Pelanggaran: {viol_total}",
    ]
    pad, lh, bw = 8, 22, 270
    bh = lh * len(lines) + pad * 2
    cv2.rectangle(frame, (8, 8), (8 + bw, 8 + bh), (20, 20, 20), -1)
    cv2.rectangle(frame, (8, 8), (8 + bw, 8 + bh), (80, 80, 80), 1)
    for i, txt in enumerate(lines):
        if i == 5 and viol_total > 0:
            color = (0, 80, 255)
        elif i == 3:
            color = (0, 210, 255)
        else:
            color = (220, 220, 220)
        cv2.putText(frame, txt, (14, 8 + pad + lh * (i + 1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)


def build_bev_canvas(tracks_bev, speed_tracker, bev_w, bev_h):
    canvas = np.zeros((bev_h, bev_w, 3), np.uint8)
    canvas[:] = (45, 45, 45)
    cv2.rectangle(canvas, (0, 0), (bev_w - 1, bev_h - 1), (90, 90, 90), 2)

    # Marka jalan tengah putus-putus
    for y in range(0, bev_h, 35):
        cv2.line(canvas, (bev_w // 2, y), (bev_w // 2, min(y + 18, bev_h)), (160, 160, 160), 1)

    cv2.putText(canvas, "JAUH",  (bev_w // 2 - 20, 14),       cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 130), 1)
    cv2.putText(canvas, "DEKAT", (bev_w // 2 - 24, bev_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 130), 1)

    for tid, (bx, by) in tracks_bev.items():
        bxi  = int(np.clip(bx, 0, bev_w - 1))
        byi  = int(np.clip(by, 0, bev_h - 1))
        spd  = speed_tracker.get_speed(tid)
        viol = speed_tracker.is_violating(tid)
        col  = PALETTE["overspeed"] if viol else PALETTE["safe"]
        cv2.circle(canvas, (bxi, byi), 9, col, -1)
        cv2.circle(canvas, (bxi, byi), 9, (255, 255, 255), 1)
        cv2.putText(canvas, f"#{tid} {spd:.0f}" if spd > 0 else f"#{tid} --",
                    (bxi + 11, byi + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

    cv2.putText(canvas, "BEV", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return canvas


# ================================================================
# MAIN
# ================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n📦 Memuat model YOLO...")
    model = YOLO(MODEL_PATH)
    print(f"   ✔ {MODEL_PATH}")

    roi_pts     = STATIC_ROI_PTS
    roi_polygon = np.array(roi_pts, np.float32)
    print(f"✔ ROI: {roi_pts}")

    # Preview ROI sebelum mulai
    cap_tmp = cv2.VideoCapture(VIDEO_PATH)
    ok, first_frame = cap_tmp.read()
    cap_tmp.release()
    if not ok:
        raise FileNotFoundError(f"Tidak bisa membaca video: {VIDEO_PATH}")

    preview = first_frame.copy()
    pts_arr = np.array(roi_pts, np.int32)
    overlay = preview.copy()
    cv2.fillPoly(overlay, [pts_arr], (0, 220, 220))
    cv2.addWeighted(overlay, 0.15, preview, 0.85, 0, preview)
    cv2.polylines(preview, [pts_arr], True, (0, 220, 220), 2)
    for i, pt in enumerate(roi_pts):
        lbl = ["1:Kiri-Jauh", "2:Kanan-Jauh", "3:Kanan-Dekat", "4:Kiri-Dekat"][i]
        cv2.circle(preview, pt, 7, (0, 255, 255), -1)
        cv2.putText(preview, lbl, (pt[0] + 8, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(preview, "ROI Preview — Tekan sembarang tombol untuk mulai",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    scale_p = min(1280 / preview.shape[1], 720 / preview.shape[0], 1.0)
    cv2.imshow("ROI Preview", cv2.resize(preview, (int(preview.shape[1] * scale_p), int(preview.shape[0] * scale_p))))
    cv2.waitKey(0)
    cv2.destroyWindow("ROI Preview")

    H = build_homography(roi_pts, BEV_WIDTH, BEV_HEIGHT)
    print(f"✔ Homography OK  |  {REAL_WIDTH_METER}m×{REAL_HEIGHT_METER}m  →  {BEV_WIDTH}×{BEV_HEIGHT}px")
    print(f"  {PIXEL_PER_METER_X:.1f} px/m (X)  |  {PIXEL_PER_METER_Y:.1f} px/m (Y)")

    cap         = cv2.VideoCapture(VIDEO_PATH)
    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n▶ Mulai  |  FPS={fps:.1f}  |  frames≈{total_frame}  |  limit={SPEED_LIMIT_KMH} km/h")
    print("  q=keluar  p=pause\n")

    speed_tracker = VehicleSpeedTracker(
        fps=fps,
        speed_limit_kmh=SPEED_LIMIT_KMH,
        px_per_m_x=PIXEL_PER_METER_X,
        px_per_m_y=PIXEL_PER_METER_Y,
        speed_window=SPEED_WINDOW,
        min_frames=MIN_SPEED_FRAMES,
        violation_cooldown_sec=VIOLATION_COOLDOWN_SEC,
        max_accel_kmh_per_sec=MAX_ACCEL_KMH_PER_SEC,
        max_decel_kmh_per_sec=MAX_DECEL_KMH_PER_SEC,
    )

    frame_no         = 0
    violation_count  = 0
    paused           = False
    total_unique_ids = set()

    video_writer      = None
    output_video_path = os.path.join(OUTPUT_DIR, "demo_pelanggaran.mp4")

    while cap.isOpened():
        if paused:
            key = cv2.waitKey(30) & 0xFF
            if key == ord('p'):
                paused = False
            elif key == ord('q'):
                break
            continue

        ret, frame = cap.read()
        if not ret:
            print("✅ Video selesai.")
            break
        frame_no += 1

        results = model.track(
            frame,
            conf    = CONF_THRESHOLD,
            tracker = "bytetrack.yaml",
            persist = True,
            verbose = False,
        )

        tracks_bev         = {}
        active_ids         = set()
        pending_violations = []  # screenshot ditunda sampai semua draw selesai

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                if box.id is None:
                    continue

                track_id = int(box.id.item())
                bbox     = box.xyxy[0].cpu().numpy()
                foot     = ((bbox[0] + bbox[2]) / 2, bbox[3])

                if cv2.pointPolygonTest(roi_polygon, foot, False) < 0:
                    continue

                bx, by = point_to_bev(foot, H)
                if not (0 <= bx <= BEV_WIDTH and 0 <= by <= BEV_HEIGHT):
                    continue

                active_ids.add(track_id)
                total_unique_ids.add(track_id)
                tracks_bev[track_id] = (bx, by)

                speed_tracker.update(track_id, (bx, by), frame_no)

                is_new_viol, speed = speed_tracker.check_violation(track_id, frame_no)
                if is_new_viol:
                    violation_count += 1
                    ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    fname = os.path.join(OUTPUT_DIR, f"viol_ID{track_id}_{speed:.0f}kmh_{ts}.jpg")
                    pending_violations.append((track_id, speed, fname))
                    print(f"  ⚠  PELANGGARAN #{violation_count}  ID:{track_id}  {speed:.0f} km/h  frame:{frame_no}  → {fname}")

                draw_vehicle_box(frame, bbox, track_id,
                                 speed_tracker.get_speed(track_id),
                                 speed_tracker.is_violating(track_id),
                                 speed_tracker.is_jitter_suppressed(track_id))

        speed_tracker.evict_stale(active_ids, max_idle_frames=int(fps * 3))

        draw_roi_overlay(frame, roi_pts)
        draw_hud(frame, frame_no, fps, len(active_ids),
                 violation_count, SPEED_LIMIT_KMH, len(total_unique_ids))

        bev_canvas   = build_bev_canvas(tracks_bev, speed_tracker, BEV_WIDTH, BEV_HEIGHT)
        fh, _        = frame.shape[:2]
        bev_w_scaled = int(BEV_WIDTH * fh / BEV_HEIGHT)
        bev_disp     = cv2.resize(bev_canvas, (bev_w_scaled, fh))
        cv2.rectangle(bev_disp, (0, 0), (bev_w_scaled - 1, fh - 1), (80, 80, 80), 1)
        cv2.putText(bev_disp, "BIRD'S EYE VIEW", (4, fh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

        combined  = np.hstack([frame, bev_disp])
        scale_out = DISPLAY_WIDTH / combined.shape[1]
        out       = cv2.resize(combined, (DISPLAY_WIDTH, int(combined.shape[0] * scale_out)))

        if SAVE_VIOLATION_FRAME:
            for _, _, fname in pending_violations:
                cv2.imwrite(fname, out)

        if video_writer is None:
            out_h, out_w = out.shape[:2]
            fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))
            print(f" ⏺ Merekam → {output_video_path}")
        video_writer.write(out)

        cv2.imshow("Speed Violation Detection | q=keluar | p=pause", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n⏹  Dihentikan.")
            break
        elif key == ord('p'):
            paused = True
            print("⏸  Pause — tekan 'p' lagi untuk lanjut")

    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 50)
    print(f" ✅  Pelanggaran : {violation_count}")
    print(f"     Kendaraan  : {len(total_unique_ids)}")
    if violation_count:
        print(f"     Screenshot  : '{OUTPUT_DIR}/'")
    print("=" * 50)


if __name__ == "__main__":
    main()
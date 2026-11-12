# .\.venv\Scripts\Activate.ps1
from __future__ import annotations
import copy, json, os, sys, time
from collections import deque
from typing import Optional, Tuple
import cv2
import numpy as np
import pyautogui
try:
    import mediapipe as mp
except ImportError:
    print("ERROR: pip install mediapipe==0.10.21")
    sys.exit(1)


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "airmouse_config.json")

DEFAULT_CONFIG = {
    "camera":   {"index": 1, "backend": "DSHOW", "width": 640, "height": 480,
                 "fps": 30, "fourcc": "", "mirror": False},
    "processing": {"width": 320, "height": 240},
    "workspace":  {"corners_norm": None, "desk_width_cm": 40, "desk_height_cm": 30},
    "mouse": {
        "smoothing": "oneeuro",
        "ema_alpha": 0.5,
        "oneeuro_mincutoff": 0.9,
        "oneeuro_beta": 0.04,
        "oneeuro_dcutoff": 1.0,
        "edge_margin_px": 10,
        "sensitivity": 1.0,
    },
    "pinch": {
        "enabled": True,
        "enter_ratio": 0.30,
        "exit_ratio": 0.50,
        "click_cooldown_ms": 300,
        "scroll": {
            "dead_zone": 0.010,
            "step": 0.015,
            "max_notches": 5,
            "invert": False,
        },
    },
    "display": {"debug": False},
}


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if (k in out and isinstance(out[k], dict) and isinstance(v, dict)) else v
    return out


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return deep_merge(copy.deepcopy(DEFAULT_CONFIG), json.load(f))
        except Exception as e:
            print(f"[config] load failed: {e}; using defaults")
    return copy.deepcopy(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    try:
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, CONFIG_PATH)
    except Exception as e:
        print(f"[config] save failed: {e}")


class OneEuroFilter:
    def __init__(self, mincutoff=1.0, beta=0.02, dcutoff=1.0):
        self.mincutoff, self.beta, self.dcutoff = float(mincutoff), float(beta), float(dcutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_prev, self.dx_prev, self.t_prev = None, 0.0, None

    def __call__(self, x, t):
        if self.x_prev is None or self.t_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = t - self.t_prev
        if dt <= 1e-5:
            return self.x_prev
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.dcutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class EMAFilter:
    def __init__(self, alpha=0.5):
        self.alpha = float(alpha)
        self.x_prev = None

    def reset(self):
        self.x_prev = None

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x
        self.x_prev = self.alpha * x + (1.0 - self.alpha) * self.x_prev
        return self.x_prev


def discover_cameras(max_index=4):
    out = []
    for idx in range(max_index):
        for name, be in (("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)):
            try:
                cap = cv2.VideoCapture(idx, be)
                if cap.isOpened():
                    ok, f = cap.read()
                    if ok and f is not None:
                        out.append((idx, name, f.shape[1], f.shape[0]))
                cap.release()
            except Exception:
                pass
    return out


def open_camera(cfg):
    cam = cfg["camera"]
    idx = int(cam["index"])
    backend_map = {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF, "ANY": cv2.CAP_ANY}
    requested = str(cam.get("backend", "DSHOW")).upper()
    chain = [(requested, backend_map[requested])] if requested in backend_map else []
    for n in ("DSHOW", "MSMF", "ANY"):
        if n != requested:
            chain.append((n, backend_map[n]))

    for name, be in chain:
        try:
            cap = cv2.VideoCapture(idx, be)
            if not cap.isOpened():
                cap.release(); continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam["width"]))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam["height"]))
            cap.set(cv2.CAP_PROP_FPS, int(cam["fps"]))
            try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception: pass
            if cam.get("fourcc"):
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam["fourcc"]))
                except Exception: pass
            ok, f = cap.read()
            if not ok or f is None:
                cap.release(); continue
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"[camera] idx={idx} backend={name} -> {w}x{h} @ {fps:.1f}")
            return cap
        except Exception as e:
            print(f"[camera] idx={idx} {name}: {e}")
    return None
PALM_IDX = (0, 5, 9, 13, 17)
def palm_center(lm):
    return (float(sum(lm[i].x for i in PALM_IDX) / 5),
            float(sum(lm[i].y for i in PALM_IDX) / 5))
def _d2(a, b):
    return float(((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5)
def pinch_ratios(lm):
    palm_w = _d2(lm[5], lm[17])
    if palm_w < 1e-4:
        return 1.0, 1.0, 1.0
    return (_d2(lm[4], lm[8]) / palm_w,
            _d2(lm[8], lm[12]) / palm_w,
            _d2(lm[4], lm[12]) / palm_w)


class GestureDetector:
    LEFT, RIGHT, SCROLL = "left", "right", "scroll"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.enter = float(cfg["pinch"]["enter_ratio"])
        self.exit = float(cfg["pinch"]["exit_ratio"])
        self.cooldown = float(cfg["pinch"]["click_cooldown_ms"]) / 1000.0
        sc = cfg["pinch"]["scroll"]
        self.sc_dead = float(sc["dead_zone"])
        self.sc_step = float(sc["step"])
        self.sc_max = int(sc["max_notches"])
        self.sc_invert = bool(sc["invert"])

        self.active = None
        self.smooth = [1.0, 1.0, 1.0]
        self.anchor_y = None
        self.cooldown_until = 0.0
        self.last_event = ""
        self.scroll_total = 0

    def reset(self):
        self.active = None
        self.smooth = [1.0, 1.0, 1.0]
        self.anchor_y = None
        self.cooldown_until = 0.0

    def update(self, d_oi, d_im, d_om, palm_y, now):

        a = 0.4
        r = (d_oi, d_im, d_om)
        for i in range(3):
            self.smooth[i] = (1 - a) * self.smooth[i] + a * r[i]
        s_oi, s_im, s_om = self.smooth

        events = []
        notches = 0


        if self.active == self.LEFT   and s_oi > self.exit: self.active = None
        if self.active == self.RIGHT  and s_im > self.exit: self.active = None
        if self.active == self.SCROLL and s_om > self.exit:
            self.active = None; self.anchor_y = None


        if self.active is None and now >= self.cooldown_until:
            pairs = sorted(
                [(self.LEFT, s_oi), (self.RIGHT, s_im), (self.SCROLL, s_om)],
                key=lambda p: p[1])
            best, best_r = pairs[0]
            second_r = pairs[1][1]
            if best_r < self.enter and (best_r < 1e-6 or second_r > best_r * 1.25):
                self.active = best
                if best == self.LEFT:
                    events.append("left"); self.cooldown_until = now + self.cooldown
                elif best == self.RIGHT:
                    events.append("right"); self.cooldown_until = now + self.cooldown
                elif best == self.SCROLL:
                    self.anchor_y = palm_y


        if self.active == self.SCROLL and self.anchor_y is not None and palm_y is not None:
            delta = palm_y - self.anchor_y
            if abs(delta) > self.sc_dead:
                eff = abs(delta) - self.sc_dead
                n = min(int(eff / self.sc_step), self.sc_max)
                if n > 0:
                    sign = 1 if delta > 0 else -1
                    notches = -sign * n if not self.sc_invert else sign * n
                    self.anchor_y += sign * n * self.sc_step
                    self.scroll_total += notches

        if events:
            self.last_event = events[-1]
        return events, notches


class WorkspaceCalibrator:
    CORNERS = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
    DURATION = 1.0

    def __init__(self):
        self.active = False
        self.step = 0
        self.corners = []
        self.collecting = False
        self.t0 = 0.0
        self.samples = []

    def start(self, now):
        self.active = True
        self.step = 0
        self.corners = []
        self.collecting = False
        self.samples = []
        self.t0 = now

    def cancel(self):
        self.active = False
        self.collecting = False

    def press_space(self, now):
        if self.active and not self.collecting and self.step < 4:
            self.collecting = True
            self.t0 = now
            self.samples = []

    def feed(self, pxy, now):
        if not (self.active and self.collecting and pxy is not None):
            return
        self.samples.append(pxy)
        if now - self.t0 >= self.DURATION:
            if len(self.samples) >= 5:
                arr = np.array(self.samples, dtype=np.float32)
                m = arr.mean(axis=0)
                self.corners.append([float(m[0]), float(m[1])])
                self.step += 1
            self.collecting = False
            if self.step >= 4:
                self.active = False

    def done(self):
        return not self.active and len(self.corners) == 4

    def progress(self, now):
        return min(1.0, (now - self.t0) / self.DURATION) if self.collecting else 0.0

    def corner_name(self):
        return self.CORNERS[self.step] if self.step < 4 else "-"

    @staticmethod
    def build_transform(corners_norm, sw, sh, margin=0):
        src = np.array(corners_norm, dtype=np.float32)
        m = float(margin)
        dst = np.array([[m, m], [sw - 1 - m, m],
                        [sw - 1 - m, sh - 1 - m], [m, sh - 1 - m]], dtype=np.float32)
        return cv2.getPerspectiveTransform(src, dst)


def _fallback_transform(sw, sh):
    fx1, fx2, fy1, fy2 = 0.25, 0.75, 0.30, 0.70
    sx = (sw - 1) / (fx2 - fx1)
    sy = (sh - 1) / (fy2 - fy1)
    return np.array([[sx, 0, -fx1 * sx],
                     [0, sy, -fy1 * sy],
                     [0, 0, 1]], dtype=np.float64)


def draw_overlay(frame, cfg, palm_xy, hand, mouse_on, estop, debug, ws_cal,
                 gest, has_cal, fps, proc_ms):
    h, w = frame.shape[:2]
    F = cv2.FONT_HERSHEY_SIMPLEX

    line = [f"FPS {fps:4.1f}", f"proc {proc_ms:4.1f}ms",
            "HAND" if hand else "NO HAND",
            "MOUSE ON" if (mouse_on and not estop) else "MOUSE OFF"]
    if estop: line.append("E-STOP")
    line.append("CALIB OK" if has_cal else "FALLBACK MAP (C)")
    if debug: line.append("DEBUG")
    txt = " | ".join(line)
    cv2.putText(frame, txt, (8, 18), F, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, txt, (8, 18), F, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    if cfg["workspace"]["corners_norm"]:
        pts = np.array([[int(c[0] * w), int(c[1] * h)] for c in cfg["workspace"]["corners_norm"]],
                       dtype=np.int32)
        cv2.polylines(frame, [pts], True, (0, 200, 255), 1, cv2.LINE_AA)

    if palm_xy is not None:
        px, py = int(palm_xy[0] * w), int(palm_xy[1] * h)
        color = (0, 255, 0) if gest.active is None else (0, 200, 255)
        cv2.circle(frame, (px, py), 8, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 2, color, -1, cv2.LINE_AA)


    if gest.active:
        cv2.putText(frame, f"GESTURE: {gest.active.upper()}", (8, 38),
                    F, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"GESTURE: {gest.active.upper()}", (8, 38),
                    F, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    if gest.scroll_total != 0:
        cv2.putText(frame, f"scroll {gest.scroll_total:+d}", (8, 58),
                    F, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

    if debug:
        y = 78
        cv2.putText(frame, f"ratios oi={gest.smooth[0]:.2f} im={gest.smooth[1]:.2f} om={gest.smooth[2]:.2f}",
                    (8, y), F, 0.4, (255, 255, 0), 1, cv2.LINE_AA); y += 16
        cv2.putText(frame, f"enter={gest.enter:.2f} exit={gest.exit:.2f} active={gest.active}",
                    (8, y), F, 0.4, (255, 255, 0), 1, cv2.LINE_AA); y += 16

    if ws_cal.active:
        msg = f"WORKSPACE CALIB | palm at {ws_cal.corner_name()} | SPACE to capture"
        cv2.putText(frame, msg, (8, h - 30), F, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, msg, (8, h - 30), F, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        if ws_cal.collecting:
            p = ws_cal.progress(time.perf_counter())
            cv2.rectangle(frame, (20, h - 16), (20 + int((w - 40) * p), h - 6), (0, 255, 0), -1)


def run() -> int:
    cfg = load_config()
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = True
    sw, sh = pyautogui.size()
    print(f"[screen] {sw}x{sh}")

    cap = open_camera(cfg)
    if cap is None:
        print("[camera] scanning indices 0..3")
        found = discover_cameras()
        if not found:
            print("[camera] no devices"); return 1
        pick = next((i for i, *_ in found if i != 0), found[0][0])
        cfg["camera"]["index"] = int(pick); save_config(cfg)
        cap = open_camera(cfg)
        if cap is None:
            print("[camera] failed"); return 1

    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, model_complexity=0,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    ws_cal = WorkspaceCalibrator()
    gest = GestureDetector(cfg)


    transform = None
    is_cal = False
    if cfg["workspace"]["corners_norm"]:
        try:
            transform = WorkspaceCalibrator.build_transform(
                cfg["workspace"]["corners_norm"], sw, sh, cfg["mouse"]["edge_margin_px"])
            is_cal = True
            print("[calib] loaded saved calibration")
        except Exception as e:
            print(f"[calib] saved corners invalid: {e}")
    if transform is None:
        transform = _fallback_transform(sw, sh)
        print("[calib] using FALLBACK map — press C for precision")


    use_oe = cfg["mouse"]["smoothing"] == "oneeuro"
    if use_oe:
        fx = OneEuroFilter(cfg["mouse"]["oneeuro_mincutoff"], cfg["mouse"]["oneeuro_beta"], cfg["mouse"]["oneeuro_dcutoff"])
        fy = OneEuroFilter(cfg["mouse"]["oneeuro_mincutoff"], cfg["mouse"]["oneeuro_beta"], cfg["mouse"]["oneeuro_dcutoff"])
        ex = ey = None
    else:
        fx = fy = None
        ex = EMAFilter(cfg["mouse"]["ema_alpha"]); ey = EMAFilter(cfg["mouse"]["ema_alpha"])

    mouse_on = True
    estop = False
    debug = bool(cfg["display"].get("debug", False))
    last_int: Optional[Tuple[int, int]] = None
    last_hand_t = 0.0
    frame_ts: deque = deque(maxlen=30)
    proc_ts: deque = deque(maxlen=30)
    no_frame = 0
    SAFE = 10.0

    WIN = "AirMouse 2.0"
    print("[app] Q quit | C calib | F flip | M mouse | D debug | R reset | ESC stop")

    try:
        while True:
            now = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                no_frame += 1
                if no_frame > 30:
                    print("[camera] reconnecting")
                    cap.release(); time.sleep(0.3)
                    cap = open_camera(cfg); no_frame = 0
                    if cap is None:
                        print("[camera] failed"); break
                time.sleep(0.02); continue
            no_frame = 0

            if cfg["camera"].get("mirror", False):
                frame = cv2.flip(frame, 1)

            pw, ph = int(cfg["processing"]["width"]), int(cfg["processing"]["height"])
            small = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            t0 = time.perf_counter()
            res = hands.process(rgb)
            proc_ts.append(time.perf_counter() - t0)

            hand_lm = res.multi_hand_landmarks[0].landmark if res.multi_hand_landmarks else None
            palm_xy = None

            if hand_lm is not None:
                palm_xy = palm_center(hand_lm)
                last_hand_t = now
            else:
                if now - last_hand_t > 0.5:
                    if use_oe: fx.reset(); fy.reset()
                    else: ex.reset(); ey.reset()
                    last_int = None
                    gest.reset()


            if ws_cal.active:
                ws_cal.feed(palm_xy, now)
                if ws_cal.done():
                    cfg["workspace"]["corners_norm"] = ws_cal.corners
                    transform = WorkspaceCalibrator.build_transform(
                        ws_cal.corners, sw, sh, cfg["mouse"]["edge_margin_px"])
                    is_cal = True
                    save_config(cfg)
                    print("[calib] saved")


            if hand_lm is not None and not ws_cal.active:
                oi, im, om = pinch_ratios(hand_lm)
                events, notches = gest.update(oi, im, om, palm_xy[1], now)
                for ev in events:
                    if ev == "left" and mouse_on and not estop:
                        pyautogui.click(button="left")
                        print(f"[click] LEFT")
                    elif ev == "right" and mouse_on and not estop:
                        pyautogui.click(button="right")
                        print(f"[click] RIGHT")
                if notches != 0 and mouse_on and not estop:
                    pyautogui.scroll(int(notches))
                    print(f"[scroll] {notches:+d}")


            can_move = (mouse_on and not estop and not ws_cal.active
                        and palm_xy is not None and gest.active is None)
            if can_move:
                pt = np.array([[[palm_xy[0], palm_xy[1]]]], dtype=np.float32)
                sp = cv2.perspectiveTransform(pt, transform)[0][0]
                sx, sy = float(sp[0]), float(sp[1])

                sens = float(cfg["mouse"]["sensitivity"])
                if sens != 1.0:
                    cx, cy = sw / 2.0, sh / 2.0
                    sx, sy = cx + (sx - cx) * sens, cy + (sy - cy) * sens
                sx = max(SAFE, min(sx, sw - 1 - SAFE))
                sy = max(SAFE, min(sy, sh - 1 - SAFE))

                if use_oe:
                    csx, csy = fx(sx, now), fy(sy, now)
                else:
                    csx, csy = ex(sx), ey(sy)

                ix, iy = int(round(csx)), int(round(csy))
                if last_int != (ix, iy):
                    pyautogui.moveTo(ix, iy, duration=0)
                    last_int = (ix, iy)


            frame_ts.append(now)
            fps = (len(frame_ts) - 1) / (frame_ts[-1] - frame_ts[0]) if len(frame_ts) > 1 else 0.0
            proc_ms = float(np.mean(proc_ts)) * 1000.0 if proc_ts else 0.0

            draw_overlay(frame, cfg, palm_xy, hand_lm is not None,
                         mouse_on, estop, debug, ws_cal, gest, is_cal, fps, proc_ms)
            cv2.imshow(WIN, frame)

            try:
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')): break
            elif key in (ord('c'), ord('C')):
                print("[calib] start workspace calibration")
                ws_cal.start(now); gest.reset()
            elif key in (ord('f'), ord('F')):
                cfg["camera"]["mirror"] = not cfg["camera"].get("mirror", False)
                cfg["workspace"]["corners_norm"] = None
                transform = _fallback_transform(sw, sh)
                is_cal = False
                gest.reset(); last_int = None
                if use_oe: fx.reset(); fy.reset()
                else: ex.reset(); ey.reset()
                save_config(cfg)
                print(f"[ui] mirror={cfg['camera']['mirror']} — RECALIBRATE (C)")
            elif key in (ord('d'), ord('D')):
                debug = not debug; cfg["display"]["debug"] = debug
            elif key in (ord('m'), ord('M')):
                mouse_on = not mouse_on; estop = False
                print(f"[ui] mouse_on={mouse_on}")
            elif key in (ord('r'), ord('R')):
                ws_cal.cancel(); gest.reset()
                cfg["workspace"]["corners_norm"] = None
                transform = _fallback_transform(sw, sh); is_cal = False
                last_int = None
                if use_oe: fx.reset(); fy.reset()
                else: ex.reset(); ey.reset()
                save_config(cfg)
                print("[calib] reset -> fallback active")
            elif key == 27:
                estop = True; mouse_on = False
                print("[ui] E-STOP — press M to re-enable")
            elif key == 32 and ws_cal.active:
                ws_cal.press_space(now)

    finally:
        try: cap.release()
        except Exception: pass
        try: hands.close()
        except Exception: pass
        cv2.destroyAllWindows()
        save_config(cfg)
        print("[app] exited")

    return 0


if __name__ == "__main__":
    sys.exit(run())

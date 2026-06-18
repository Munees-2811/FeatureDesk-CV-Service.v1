"""
FeatureDesk CV Service — Local Desktop Tester (Tkinter)
=======================================================

A standalone desktop app to verify the computer-vision pipeline on YOUR webcam
BEFORE the client connects it to their dashboard. It runs the *exact same*
pipeline the API uses (`AnalysisService.run_pipeline`), draws the detected
face/person + phone boxes on the live video, and shows the flat JSON the
dashboard will receive:

    {"student_id": "S001", "status": "Focused", "attention": 94,
     "phone": false, "faces": 1}

Run (from the repo root, with your venv active):

    pip install -r requirements.txt        # core CV deps (cv2, mediapipe, ultralytics)
    pip install pillow                      # only needed by this tester's UI
    python download_models.py               # one-time model download
    python tools/desktop_tester.py

Tkinter ships with standard Python on Windows/macOS. No server needed — this
talks to the pipeline directly, in-process.
"""
import os
import sys
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk

import cv2

# Allow running as `python tools/desktop_tester.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit("This tester needs Pillow for the video preview:  pip install pillow")

from services.analysis_service import AnalysisService
from api.v1.schemas.responses import LiveAnalysisResponse

STUDENT_ID = "S001"
SESSION_ID = "local-test"
TARGET_FPS = 12  # keep CPU sane on a laptop

# BGR colours.
GREEN = (0, 200, 0)
RED = (0, 0, 230)
YELLOW = (0, 200, 230)


class CameraWorker(threading.Thread):
    """Grabs frames, runs the pipeline off the UI thread, and pushes results
    (annotated frame + live JSON) to a queue the UI polls."""

    def __init__(self, cam_index: int, out_queue: "queue.Queue") -> None:
        super().__init__(daemon=True)
        self._cam_index = cam_index
        self._queue = out_queue
        self._stop = threading.Event()
        self._service = AnalysisService()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        # CAP_DSHOW opens much faster on Windows; harmless elsewhere.
        backend = cv2.CAP_DSHOW if os.name == "nt" else 0
        cap = cv2.VideoCapture(self._cam_index, backend)
        if not cap.isOpened():
            self._queue.put(("error", f"Could not open camera index {self._cam_index}"))
            return

        min_dt = 1.0 / TARGET_FPS
        while not self._stop.is_set():
            t0 = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                self._queue.put(("error", "Camera read failed"))
                break

            frame = cv2.flip(frame, 1)  # mirror so it feels like a selfie cam
            try:
                response, detections, tracks = self._service.run_pipeline(
                    frame, SESSION_ID, int(time.time() * 1000)
                )
                live = LiveAnalysisResponse.from_analysis(response, STUDENT_ID)
                annotated = self._annotate(frame, detections, tracks, response, live)
                self._queue.put(("frame", (annotated, live, response)))
            except Exception as exc:  # keep the UI alive on any per-frame error
                self._queue.put(("error", f"{type(exc).__name__}: {exc}"))

            elapsed = time.monotonic() - t0
            if elapsed < min_dt:
                time.sleep(min_dt - elapsed)

        cap.release()

    @staticmethod
    def _annotate(frame, detections, tracks, response, live) -> "cv2.Mat":
        # Person boxes (one per tracked student).
        for i, track in enumerate(tracks):
            x1, y1, x2, y2 = track.bbox
            student = response.students[i] if i < len(response.students) else None
            label = live.status if student else "?"
            colour = GREEN if (student and student.eye_metrics) else YELLOW
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, f"FACE / {label}", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

        # Phone boxes in red.
        for d in detections:
            if d.class_name.lower() == "cell phone":
                x1, y1, x2, y2 = d.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)
                cv2.putText(frame, "PHONE", (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)

        # Top banner.
        banner = ("FACE DETECTED" if live.faces > 0 else "NO FACE")
        bcol = GREEN if live.faces > 0 else RED
        cv2.putText(frame, f"{banner}  |  {live.status}  |  attention {live.attention}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bcol, 2)
        return frame


# UI palette
BG = "#0f1117"
CARD = "#1a1f2e"
MUTED = "#9ca3af"
PREVIEW_W, PREVIEW_H = 380, 285  # small left preview (4:3)

STATE_COLORS = {
    "Focused": "#22c55e",
    "Distracted": "#f59e0b",
    "Sleeping": "#ef4444",
    "Absent": "#6b7280",
}


class TesterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("FeatureDesk CV — Local Tester")
        self.root.configure(bg=BG)
        self.root.geometry("960x540")
        self.queue: "queue.Queue" = queue.Queue(maxsize=2)
        self.worker: CameraWorker | None = None
        self._imgtk = None  # keep ref so Tk doesn't GC the image

        # Top control bar
        bar = tk.Frame(root, bg=BG)
        bar.pack(fill="x", padx=14, pady=10)
        tk.Label(bar, text="Camera index:", fg="#cbd5e1", bg=BG,
                 font=("Segoe UI", 10)).pack(side="left")
        self.cam_var = tk.StringVar(value="0")
        ttk.Entry(bar, textvariable=self.cam_var, width=4).pack(side="left", padx=(4, 12))
        self.start_btn = ttk.Button(bar, text="▶ Start camera", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="■ Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.live_dot = tk.Label(bar, text="● idle", fg=MUTED, bg=BG, font=("Segoe UI", 10, "bold"))
        self.live_dot.pack(side="right")

        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        # ---- LEFT: small live preview with bounding boxes ----
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", anchor="n")
        tk.Label(left, text="LIVE  (bounding-box analysis)", fg="#93c5fd", bg=BG,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 6))
        self.video = tk.Label(left, bg="black", width=PREVIEW_W, height=PREVIEW_H)
        self.video.pack()
        self.fps = tk.Label(left, text="", fg=MUTED, bg=BG, font=("Consolas", 9))
        self.fps.pack(anchor="w", pady=(6, 0))

        # ---- RIGHT: dashboard ----
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(16, 0))
        tk.Label(right, text="STUDENT BEHAVIOUR DASHBOARD", fg="#e5e7eb", bg=BG,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w")

        # Big status banner
        self.status_card = tk.Frame(right, bg=CARD)
        self.status_card.pack(fill="x", pady=(10, 8))
        self.status_lbl = tk.Label(self.status_card, text="—", fg="#e5e7eb", bg=CARD,
                                    font=("Segoe UI", 22, "bold"), pady=14)
        self.status_lbl.pack()

        # Metric cards row
        row = tk.Frame(right, bg=BG)
        row.pack(fill="x")
        self.attention_val = self._metric(row, "ATTENTION", "0", 0)
        self.faces_val = self._metric(row, "FACES", "0", 1)
        self.phone_val = self._metric(row, "PHONE", "—", 2)

        # Attention bar
        tk.Label(right, text="Attention level", fg=MUTED, bg=BG,
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(10, 2))
        self.att_canvas = tk.Canvas(right, height=16, bg="#111827", highlightthickness=0)
        self.att_canvas.pack(fill="x")

        # Secondary metrics (richer telemetry)
        self.extra = tk.Label(right, text="head —   eyes —", fg=MUTED, bg=BG,
                              font=("Consolas", 10), justify="left")
        self.extra.pack(anchor="w", pady=(10, 4))

        # Raw JSON (the exact contract sent to the client dashboard)
        tk.Label(right, text="JSON sent to client dashboard", fg="#93c5fd", bg=BG,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(6, 2))
        self.json_box = tk.Text(right, height=7, bg="#111827", fg="#e5e7eb",
                                insertbackground="#e5e7eb", font=("Consolas", 11), bd=0)
        self.json_box.pack(fill="both", expand=True)

        self.status_hint = tk.Label(right, text="Idle — click Start. First frame loads "
                                    "models (a few seconds).", fg=MUTED, bg=BG,
                                    font=("Segoe UI", 9))
        self.status_hint.pack(anchor="w", pady=(6, 0))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _metric(self, parent, title, value, col) -> tk.Label:
        card = tk.Frame(parent, bg=CARD)
        card.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 8, 0))
        parent.grid_columnconfigure(col, weight=1)
        tk.Label(card, text=title, fg=MUTED, bg=CARD, font=("Segoe UI", 9)).pack(pady=(10, 0))
        val = tk.Label(card, text=value, fg="#e5e7eb", bg=CARD, font=("Segoe UI", 20, "bold"))
        val.pack(pady=(0, 10))
        return val

    def start(self) -> None:
        if self.worker:
            return
        try:
            cam = int(self.cam_var.get())
        except ValueError:
            cam = 0
        self.status_hint.config(text="Starting camera and loading models…")
        self.live_dot.config(text="● connecting", fg="#f59e0b")
        self.worker = CameraWorker(cam, self.queue)
        self.worker.start()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.root.after(30, self._poll)

    def stop(self) -> None:
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.live_dot.config(text="● idle", fg=MUTED)
        self.status_hint.config(text="Stopped.")

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "error":
                    self.status_hint.config(text=f"⚠ {payload}")
                    self.live_dot.config(text="● error", fg="#ef4444")
                    continue
                self._render(*payload)
        except queue.Empty:
            pass
        if self.worker:
            self.root.after(15, self._poll)

    def _render(self, frame, live, response) -> None:
        # Left preview (scaled small)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((PREVIEW_W, PREVIEW_H))
        self._imgtk = ImageTk.PhotoImage(img)
        self.video.config(image=self._imgtk)

        # Dashboard
        colour = STATE_COLORS.get(live.status, "#e5e7eb")
        self.status_card.config(bg=colour)
        self.status_lbl.config(text=live.status, bg=colour, fg="#0f1117")
        self.attention_val.config(text=str(live.attention))
        self.faces_val.config(text=str(live.faces))
        self.phone_val.config(text="⚠ YES" if live.phone else "✓ no",
                              fg="#ef4444" if live.phone else "#22c55e")

        # Attention bar
        self.att_canvas.delete("all")
        w = self.att_canvas.winfo_width() or 300
        self.att_canvas.create_rectangle(0, 0, w * live.attention / 100, 16,
                                         fill=colour, width=0)

        # Secondary telemetry from the detailed response
        head = eyes = "—"
        if response.students:
            s = response.students[0]
            if s.head_pose:
                head = f"yaw {s.head_pose.yaw:+.0f}  pitch {s.head_pose.pitch:+.0f}"
            if s.eye_metrics:
                avg = (s.eye_metrics.ear_left + s.eye_metrics.ear_right) / 2
                eyes = f"EAR {avg:.2f}"
        self.extra.config(text=f"head: {head}    eyes: {eyes}")

        self.json_box.delete("1.0", "end")
        self.json_box.insert("1.0", live.model_dump_json(indent=2))

        face = "✓ face detected" if live.faces > 0 else "✗ no face in frame"
        self.status_hint.config(text=face)
        self.fps.config(text=f"pipeline {response.processing_ms:.0f} ms/frame")
        self.live_dot.config(text="● LIVE", fg="#22c55e")

    def on_close(self) -> None:
        self.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    TesterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

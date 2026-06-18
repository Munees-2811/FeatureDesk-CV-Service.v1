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
                self._queue.put(("frame", (annotated, live, response.processing_ms)))
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


class TesterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("FeatureDesk CV — Local Tester")
        self.root.configure(bg="#0f1117")
        self.queue: "queue.Queue" = queue.Queue(maxsize=2)
        self.worker: CameraWorker | None = None

        # Controls
        bar = tk.Frame(root, bg="#0f1117")
        bar.pack(fill="x", padx=10, pady=8)
        tk.Label(bar, text="Camera index:", fg="#cbd5e1", bg="#0f1117").pack(side="left")
        self.cam_var = tk.StringVar(value="0")
        ttk.Entry(bar, textvariable=self.cam_var, width=4).pack(side="left", padx=(4, 12))
        self.start_btn = ttk.Button(bar, text="Start camera", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)

        # Video + side panel
        body = tk.Frame(root, bg="#0f1117")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.video = tk.Label(body, bg="black", width=640, height=480)
        self.video.pack(side="left")

        side = tk.Frame(body, bg="#0f1117")
        side.pack(side="left", fill="both", expand=True, padx=(12, 0))
        tk.Label(side, text="Live JSON (sent to dashboard)", fg="#93c5fd",
                 bg="#0f1117", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.json_box = tk.Text(side, height=9, width=40, bg="#111827", fg="#e5e7eb",
                                insertbackground="#e5e7eb", font=("Consolas", 12), bd=0)
        self.json_box.pack(anchor="w", pady=6, fill="x")
        self.status = tk.Label(side, text="Idle. Loads models on first frame "
                               "(first frame can take a few seconds).",
                               fg="#9ca3af", bg="#0f1117", wraplength=320, justify="left")
        self.status.pack(anchor="w", pady=6)
        self.fps = tk.Label(side, text="", fg="#9ca3af", bg="#0f1117")
        self.fps.pack(anchor="w")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._imgtk = None  # keep a ref so Tk doesn't garbage-collect the image

    def start(self) -> None:
        if self.worker:
            return
        try:
            cam = int(self.cam_var.get())
        except ValueError:
            cam = 0
        self.status.config(text="Starting camera and loading models…")
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
        self.status.config(text="Stopped.")

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "error":
                    self.status.config(text=f"⚠ {payload}")
                    continue
                frame, live, proc_ms = payload
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.video.config(image=self._imgtk)

                self.json_box.delete("1.0", "end")
                self.json_box.insert("1.0", live.model_dump_json(indent=2))
                face = "✓ face detected" if live.faces > 0 else "✗ no face"
                self.status.config(text=face)
                self.fps.config(text=f"pipeline {proc_ms:.0f} ms/frame")
        except queue.Empty:
            pass
        if self.worker:
            self.root.after(15, self._poll)

    def on_close(self) -> None:
        self.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    TesterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

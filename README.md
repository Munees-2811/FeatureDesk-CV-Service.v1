# FeatureDesk CV Service

An **independent** computer-vision microservice for the Feature Desk platform. It
analyses a student's webcam frames during exams/lessons and returns **JSON** for
the Student / Teacher / Admin dashboards:

- **Attention** — composite 0–100 score from gaze, head pose and eye state
- **Eye blink** — EAR-based blink detection and per-session blink rate
- **Head motion** — yaw / pitch / roll and head-motion variability
- **Objects** — phone / multiple-face / left-desk flags

It does **not** touch the React/Vite frontend or its Supabase database. The frontend
talks to it over HTTP/WebSocket and stores whatever JSON it wants. Raw frames are
analysed in memory and **never persisted**.

> This service complements the frontend's existing `proctoringService.ts` (which
> only covers browser events: tab-switch, copy/paste, screenshots). This adds the
> camera/vision dimension it lacks.

## Architecture

```
Frontend (React/Vite, Netlify)
   │  HTTPS frame  /  WSS stream   (X-API-Key)
   ▼
Edge      middleware/ + security/   CORS · API-key · rate-limit · timing · errors
API       api/v1/                   REST routes · WebSocket · Pydantic schemas
Service   services/                 analysis · session (aggregation) · report
Domain    vision/ tracking/ behavior/   perception → identity → scoring/flags/state
Infra     loaders/ core/ utils/ models/  singleton model loading · config · logging
```

Layered Clean Architecture; dependencies point inward. Stateless per frame —
session state is in-memory with TTL, so REST scales horizontally (WebSocket needs
sticky sessions).

## API (`/api/v1`, all but `/health` require `X-API-Key`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + `models_loaded` readiness |
| POST | `/sessions` | Start a session → `session_id` |
| GET | `/sessions/{id}` | Live session status |
| POST | `/sessions/{id}/close` | Finalise (marks `completed`/`flagged`) |
| GET | `/sessions/{id}/report` | Aggregated dashboard report |
| POST | `/analysis/frame` | Stateless single-frame analysis (polling) |
| WS | `/ws/stream/{session_id}` | Live frame stream (binary in → JSON out) |

### Per-frame response (`POST /analysis/frame`)
```jsonc
{
  "session_id": "…", "timestamp_ms": 0, "frame_id": 0, "processing_ms": 38.0,
  "students": [{
    "student_id": "0", "track_id": 0,
    "attention_score": 82, "focus_score": 80, "state": "focused",
    "head_pose": { "yaw": -4.2, "pitch": 3.1, "roll": 0.5 },
    "eye_metrics": { "ear_left": 0.31, "ear_right": 0.30, "gaze_x": 0.0, "gaze_y": 0.0 },
    "flags": { "phone_detected": false, "left_desk": false, "eyes_closed": false, "looking_away": false }
  }]
}
```

### Session report (`GET /sessions/{id}/report`)
`avg_attention`, `min_attention`, `blink_count`, `blink_rate_per_min`,
`head_motion_yaw_std`, `dominant_state`, `state_breakdown`, `flag_counts`,
and a `verdict` of `attentive | needs_attention | flagged`.

## Run locally (no cloud, $0)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_models.py            # fetches free YOLO11n + MediaPipe weights
cp .env.example .env                 # set API_KEY
uvicorn app:app --host 0.0.0.0 --port 8000
# docs at http://localhost:8000/docs when DEBUG=true
```

Or with Docker:
```bash
docker compose -f docker/docker-compose.yml up --build
```

## Frontend integration (no frontend code changed)

Point the frontend at the service with one env var, then POST frames:

```ts
// VITE_CV_SERVICE_URL + VITE_CV_API_KEY in the frontend .env
const res = await fetch(`${import.meta.env.VITE_CV_SERVICE_URL}/api/v1/analysis/frame`, {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-API-Key": import.meta.env.VITE_CV_API_KEY },
  body: JSON.stringify({ session_id, frame_b64 }) // frame_b64 = canvas.toDataURL().split(",")[1]
});
const analysis = await res.json(); // render attention_score / flags on the dashboard
```

## Configuration

All via env (see `.env.example`): `API_KEY`, `CORS_ORIGINS`, model paths,
`PRELOAD_MODELS`, `BLINK_EAR_THRESHOLD`, `LOOKING_AWAY_*_DEG`, `SESSION_TTL_SECONDS`,
rate-limit window.

## Tests

```bash
pytest tests/            # vision + behavior unit tests, REST + WS integration tests
```

## Models (free, CPU-only)

| File | Role |
|---|---|
| `yolo11n.pt` | Phone / person / object detection (Ultralytics) |
| `face_landmarker.task` | Face mesh → eyes, blink, gaze, head pose (MediaPipe) |
| `pose_landmarker_lite.task` | Body / head motion (MediaPipe) |

Weights are gitignored and fetched by `download_models.py` at build time.

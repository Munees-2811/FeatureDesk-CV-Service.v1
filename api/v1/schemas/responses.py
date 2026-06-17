from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class AttentionState(str, Enum):
    FOCUSED = "focused"
    DISTRACTED = "distracted"
    SLEEPING = "sleeping"
    ABSENT = "absent"


class HeadPose(BaseModel):
    yaw: float
    pitch: float
    roll: float


class EyeMetrics(BaseModel):
    ear_left: float = Field(..., description="Eye Aspect Ratio — left eye")
    ear_right: float = Field(..., description="Eye Aspect Ratio — right eye")
    gaze_x: Optional[float] = None
    gaze_y: Optional[float] = None


class BehaviorFlags(BaseModel):
    phone_detected: bool = False
    left_desk: bool = False
    eyes_closed: bool = False
    looking_away: bool = False


class StudentAnalysis(BaseModel):
    student_id: str
    track_id: Optional[int] = None
    attention_score: float = Field(..., ge=0, le=100)
    focus_score: float = Field(..., ge=0, le=100)
    state: AttentionState
    head_pose: Optional[HeadPose] = None
    eye_metrics: Optional[EyeMetrics] = None
    flags: BehaviorFlags = BehaviorFlags()


class AnalysisResponse(BaseModel):
    session_id: str
    timestamp_ms: int
    frame_id: int
    students: List[StudentAnalysis]
    processing_ms: float


class SessionResponse(BaseModel):
    session_id: str
    student_id: str
    created_at: int
    frame_count: int


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    models_loaded: bool

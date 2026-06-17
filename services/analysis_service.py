import time
from api.v1.schemas.requests import FrameAnalysisRequest
from api.v1.schemas.responses import AnalysisResponse
from utils.image_utils import decode_base64_frame
from vision.detector import YOLODetector
from vision.face_analyzer import FaceAnalyzer
from vision.pose_analyzer import PoseAnalyzer
from vision.head_pose import HeadPoseEstimator
from vision.eye_tracker import EyeTracker
from vision.object_classifier import ObjectClassifier
from behavior.attention_scorer import AttentionScorer
from behavior.focus_scorer import FocusScorer
from behavior.state_machine import StateMachine
from behavior.flag_detector import FlagDetector
from tracking.tracker import ByteTrackWrapper
from core.logger import get_logger

logger = get_logger(__name__)


class AnalysisService:
    """Orchestrates the full CV → behavior pipeline for a single frame."""

    def __init__(self) -> None:
        self._detector = YOLODetector()
        self._face = FaceAnalyzer()
        self._pose = PoseAnalyzer()
        self._head_pose = HeadPoseEstimator()
        self._eye = EyeTracker()
        self._obj_clf = ObjectClassifier()
        self._tracker = ByteTrackWrapper()
        self._attention = AttentionScorer()
        self._focus = FocusScorer()
        self._state = StateMachine()
        self._flags = FlagDetector()
        self._frame_counter: dict[str, int] = {}

    async def process_frame(self, req: FrameAnalysisRequest) -> AnalysisResponse:
        t0 = time.monotonic()
        frame = decode_base64_frame(req.frame_b64)

        detections = self._detector.detect(frame)
        tracks = self._tracker.update(detections, frame)
        students = []

        for track in tracks:
            face_result = self._face.analyse(frame, track.bbox)
            pose_result = self._pose.analyse(frame, track.bbox)
            head = self._head_pose.estimate(face_result)
            eye = self._eye.compute(face_result)
            objects = self._obj_clf.classify(detections)

            attention = self._attention.score(head, eye)
            focus = self._focus.score(head, eye, pose_result)
            state = self._state.transition(track.id, attention, eye)
            flags = self._flags.detect(eye, objects, pose_result)

            students.append(self._build_student(track, attention, focus, state, head, eye, flags))

        frame_id = self._frame_counter.setdefault(req.session_id, 0)
        self._frame_counter[req.session_id] = frame_id + 1

        return AnalysisResponse(
            session_id=req.session_id,
            timestamp_ms=req.timestamp_ms or int(time.time() * 1000),
            frame_id=frame_id,
            students=students,
            processing_ms=round((time.monotonic() - t0) * 1000, 2),
        )

    @staticmethod
    def _build_student(track, attention, focus, state, head, eye, flags):
        from api.v1.schemas.responses import StudentAnalysis, HeadPose, EyeMetrics, BehaviorFlags
        return StudentAnalysis(
            student_id=str(track.id),
            track_id=track.id,
            attention_score=attention,
            focus_score=focus,
            state=state,
            head_pose=HeadPose(**head) if head else None,
            eye_metrics=EyeMetrics(**eye) if eye else None,
            flags=BehaviorFlags(**flags),
        )

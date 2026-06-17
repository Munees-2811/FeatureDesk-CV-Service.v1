from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_file_override=False,   # don't crash if .env is absent
    )

    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    API_KEY: str = Field(default="dev-secret-key", description="Secret API key for auth")
    CORS_ORIGINS: List[str] = ["http://localhost:5173"]
    LOG_LEVEL: str = "INFO"

    # Frame constraints
    MAX_FRAME_WIDTH: int = 1280
    MAX_FRAME_HEIGHT: int = 720

    # Model paths
    YOLO_MODEL_PATH: str = "models/yolo11n.pt"
    BEST_MODEL_PATH: str = "models/best.pt"
    FACE_LANDMARKER_PATH: str = "models/face_landmarker.task"
    POSE_LANDMARKER_PATH: str = "models/pose_landmarker_lite.task"


settings = Settings()
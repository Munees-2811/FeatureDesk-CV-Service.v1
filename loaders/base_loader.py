from abc import ABC, abstractmethod


class BaseLoader(ABC):
    """Abstract base for singleton model loaders."""

    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls._load_model()
        return cls._instance

    @classmethod
    @abstractmethod
    def _load_model(cls):
        ...

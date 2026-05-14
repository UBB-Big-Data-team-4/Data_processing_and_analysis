import threading
from pathlib import Path
from typing import Any, cast
import shutil

from ultralytics import YOLO


class ModelSingleton:
    _instances = {}
    _instances_lock = threading.Lock()

    def __new__(cls, model_name: str, models_dir: str = "models"):
        key = f"{model_name}::{models_dir}"
        with cls._instances_lock:
            if key in cls._instances:
                return cls._instances[key]
            inst = super().__new__(cls)
            cls._instances[key] = inst
            return inst

    def __init__(self, model_name: str, models_dir: str = "models"):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self.model_name = model_name
        self.models_dir = Path(models_dir)
        self.model_path = ""
        self.model = None
        self._load_model()

    def _ensure_supported_extension(self, path: Path) -> Path:
        if path.suffix:
            return path

        pt_path = path.with_suffix(".pt")
        if not pt_path.exists():
            shutil.copy2(path, pt_path)
        return pt_path

    def _find_model_path(self) -> str:
        candidate = self.models_dir / self.model_name
        if candidate.exists() and candidate.is_file():
            return str(self._ensure_supported_extension(candidate))

        candidate_pt = self.models_dir / f"{self.model_name}.pt"
        if candidate_pt.exists() and candidate_pt.is_file():
            return str(candidate_pt)

        candidate_dir = self.models_dir / self.model_name
        if candidate_dir.exists() and candidate_dir.is_dir():
            for file_path in candidate_dir.glob("*.pt"):
                return str(file_path)

        return ""

    def _load_model(self):
        model_path = self._find_model_path()
        if not model_path:
            print(f"Model '{self.model_name}' not found in '{self.models_dir}'", flush=True)
            return

        self.model_path = model_path
        try:
            print(f"Loading model from: {self.model_path}", flush=True)
            self.model = YOLO(str(self.model_path))
        except Exception as e:
            print(f"Failed to load model '{self.model_path}': {e}", flush=True)
            self.model = None

    def predict(self, frame) -> int:
        model = self.model
        if model is None:
            return -1

        model = cast(Any, model)

        try:
            results = model.predict(frame, classes=[0], verbose=False)

            if len(results) > 0 and hasattr(results[0], "boxes"):
                return len(results[0].boxes)

            return 0
        except Exception as e:
            print(f"Model prediction failed: {e}", flush=True)
            return -1
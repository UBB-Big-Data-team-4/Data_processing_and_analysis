import threading
from pathlib import Path
from typing import Any, cast
import shutil

import numpy as np
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
            return 0
        model = cast(Any, model)

        try:
            results = model(frame)
            probs = None

            if hasattr(results, "probs"):
                probs = results.probs
            elif len(results) > 0 and hasattr(results[0], "probs"):
                probs = results[0].probs

            if probs is not None:
                if hasattr(probs, 'top5') and hasattr(probs, 'top5conf'):
                    top_classes = [model.names[i] for i in probs.top5[:3]]
                    top_confs = [round(float(c), 2) for c in probs.top5conf[:3]]
                    print(f"Top 3 guesses: {list(zip(top_classes, top_confs))}", flush=True)
                class_index = self._extract_class_index(probs)
                return self._resolve_label(class_index)

            return self._resolve_label(0)
        except Exception as e:
            print(f"Model prediction failed: {e}", flush=True)
            return 0

    def _extract_class_index(self, probs) -> int:
        if hasattr(probs, 'top1') and probs.top1 is not None:
            return int(probs.top1)

        try:
            if hasattr(probs, 'data'):
                arr = probs.data.cpu().numpy()
            else:
                arr = np.array(probs)

            if arr.ndim == 1:
                return int(arr.argmax())
            if arr.ndim == 2:
                return int(arr[0].argmax())
        except Exception as e:
            print(f"Error parsing probabilities: {e}", flush=True)

        return 0

    def _resolve_label(self, class_index: int) -> int:
        class_index = max(0, int(class_index))
        names = getattr(self.model, "names", None)

        if isinstance(names, dict):
            label = names.get(class_index, class_index)
        elif isinstance(names, (list, tuple)) and class_index < len(names):
            label = names[class_index]
        else:
            label = class_index

        try:
            return max(0, min(99, int(str(label))))
        except (TypeError, ValueError):
            return max(0, min(99, class_index))

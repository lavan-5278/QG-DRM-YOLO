# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
DSR-Det predictor (Phase 0/1 skeleton).

Optional subclass of the official ``DetectionPredictor``; candidate/gate visualization
hooks are added in Phase 5. Predictions continue to run through ``model.predict()``.
"""

from ultralytics.models.yolo.detect.predict import DetectionPredictor


class DSRDetectionPredictor(DetectionPredictor):
    """Optional predictor subclass for DSR candidate/gate visualization hooks."""

    pass

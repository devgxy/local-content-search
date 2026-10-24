"""STEP-02 重排服务测试（不加载真实模型）。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from app.config import Settings
from app.reranker import RerankerService, RerankerUnavailable


class FakeCrossEncoder:
    def predict(self, pairs, **kwargs):  # noqa: ANN001
        assert len(pairs) == 2
        return np.asarray([0.1, 1.2], dtype=np.float32)


def test_missing_model_raises(tmp_settings: Settings) -> None:
    settings = replace(tmp_settings, models_dir=tmp_settings.models_dir / "missing")
    service = RerankerService(settings)
    with pytest.raises(RerankerUnavailable):
        service.score("q", ["a"])


def test_score_clipped(tmp_settings: Settings) -> None:
    service = RerankerService(tmp_settings)
    service._model = FakeCrossEncoder()  # noqa: SLF001
    scores = service.score("q", ["a", "b"])
    assert np.allclose(scores, np.asarray([0.1, 1.0], dtype=np.float32))


def test_empty_passages(tmp_settings: Settings) -> None:
    service = RerankerService(tmp_settings)
    scores = service.score("q", [])
    assert scores.size == 0


def test_health(tmp_settings: Settings) -> None:
    service = RerankerService(tmp_settings)
    health = service.health()
    assert health["ready"] is False
    assert "available" in health

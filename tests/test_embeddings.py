"""STEP-02 模型层测试（不加载真实模型）。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from app.config import Settings
from app.embeddings import (
    BGE_QUERY_INSTRUCTION,
    GEMMA_QUERY_PREFIX,
    EmbeddingService,
    ModelNotReady,
)


class FakeModel:
    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts, **kwargs):  # noqa: ANN001
        self.calls.append(list(texts))
        return np.ones((len(texts), self.dim), dtype=np.float32) / np.sqrt(self.dim)


def test_missing_model_raises(tmp_settings: Settings) -> None:
    settings = replace(tmp_settings, models_dir=tmp_settings.models_dir / "missing")
    service = EmbeddingService(settings)
    with pytest.raises(ModelNotReady):
        service.encode_query("测试")


def test_query_gets_bge_instruction(tmp_settings: Settings) -> None:
    service = EmbeddingService(tmp_settings)
    fake = FakeModel()
    service._model = fake  # noqa: SLF001
    vector = service.encode_query("混合检索")
    assert vector.shape == (1024,)
    assert fake.calls == [[BGE_QUERY_INSTRUCTION + "混合检索"]]


def test_gemma_prompts_and_title_template(tmp_settings: Settings) -> None:
    settings = replace(tmp_settings, embed_provider="gemma", embed_dim=768)
    service = EmbeddingService(settings)
    fake = FakeModel(dim=768)
    service._model = fake  # noqa: SLF001
    service.encode_query("混合检索")
    assert fake.calls[-1] == [GEMMA_QUERY_PREFIX + "混合检索"]
    service.encode_documents(["正文"], titles=["标题"])
    assert fake.calls[-1] == ["title: 标题 | text: 正文"]


def test_batch_shapes(tmp_settings: Settings) -> None:
    service = EmbeddingService(tmp_settings)
    service._model = FakeModel()  # noqa: SLF001
    vectors = service.encode_documents(["a", "b", "c"])
    assert len(vectors) == 3
    assert len(vectors[0]) == 1024
    assert service.encode_documents([]) == []


def test_health_not_loaded(tmp_settings: Settings) -> None:
    service = EmbeddingService(tmp_settings)
    health = service.health()
    assert health["ready"] is False
    assert health["dim"] == 1024
    assert health["provider"] == "bge"


def test_model_load_once_concurrent(tmp_settings: Settings, monkeypatch) -> None:
    import threading
    import time

    service = EmbeddingService(tmp_settings)
    calls = 0

    def slow_load(settings):  # noqa: ANN001
        nonlocal calls
        time.sleep(0.05)
        calls += 1
        return FakeModel()

    monkeypatch.setattr("app.embeddings._load_model", slow_load)
    barrier = threading.Barrier(2)

    def load():
        barrier.wait()
        _ = service.model

    threads = [threading.Thread(target=load) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert calls == 1

"""pytest 公共 fixtures。

Step: STEP-00
Doc: docs/02-代码结构与追溯矩阵.md §7
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from lancedb.embeddings import TextEmbeddingFunction, get_registry, register

from app.config import PROJECT_ROOT, Settings

_FAKE_REGISTERED = False


def _register_fake() -> None:
    global _FAKE_REGISTERED
    if _FAKE_REGISTERED:
        return

    @register("test-fake-1024")
    class FakeEmbedding(TextEmbeddingFunction):
        """确定性 1024 维假向量，仅用于测试，不加载真实模型。"""

        def ndims(self) -> int:
            return 1024

        def generate_embeddings(self, texts, *args, **kwargs):  # noqa: ANN001
            vectors = []
            for text in texts:
                seed = hashlib.sha256(str(text).encode("utf-8")).digest()
                raw = (seed * 32)[:1024]
                vec = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                norm = float(np.linalg.norm(vec)) or 1.0
                vectors.append((vec / norm).tolist())
            return vectors

    _FAKE_REGISTERED = True


def make_fake_func():
    _register_fake()
    return get_registry().get("test-fake-1024").create()


@pytest.fixture()
def tmp_settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    web_dir = tmp_path / "web"
    settings = Settings(
        data_dir=data_dir,
        models_dir=models_dir,
        web_dir=web_dir,
    )
    settings.ensure_dirs()
    return settings


@pytest.fixture()
def fake_func():
    return make_fake_func()


@pytest.fixture()
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    target = PROJECT_ROOT / "tests" / "fixtures"
    if not (target / "sample.pdf").exists():
        subprocess.run(
            [sys.executable, "scripts/make_fixtures.py", "--out", str(target)],
            check=True,
            cwd=PROJECT_ROOT,
        )
    return target

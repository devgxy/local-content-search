"""bge-reranker-large 两段式重排服务。

检索先用 RRF/向量/BM25 取 top_k，再用 CrossEncoder 对 query-passage
打分；不把 CrossEncoderReranker 直接塞进 hybrid，避免替换 RRF。

Step: STEP-02
Doc: docs/steps/STEP-02-模型层.md §4.4
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from app.config import Settings

logger = logging.getLogger(__name__)


class RerankerUnavailable(RuntimeError):
    """重排模型不可用。"""
    code = "RERANK_UNAVAILABLE"


def _check_model_dir(path: Path) -> None:
    """校验重排模型目录。

    Args:
        path: 模型目录。

    Raises:
        RerankerUnavailable: 配置/权重缺失。
    """
    if not path.is_dir():
        raise RerankerUnavailable(f"重排模型目录不存在：{path}")
    if not (path / "config.json").exists():
        raise RerankerUnavailable(f"重排模型缺少 config.json：{path}")
    if not (path / "pytorch_model.bin").exists() and not (path / "model.safetensors").exists():
        raise RerankerUnavailable(f"重排模型缺少权重文件：{path}")


class RerankerService:
    """bge-reranker CrossEncoder 服务。"""
    def __init__(self, settings: Settings):
        """保存配置，模型延迟加载。

        Args:
            settings: 运行期配置。
        """
        self.settings = settings
        self._model = None
        self._loaded = False
        self._lock = threading.Lock()

    @property
    def model(self):
        """懒加载 CrossEncoder。"""
        with self._lock:
            if self._model is None:
                _check_model_dir(self.settings.rerank_model_path)
                try:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(
                        str(self.settings.rerank_model_path),
                        device=self.settings.device,
                        max_length=self.settings.rerank_max_length,
                        local_files_only=True,
                        trust_remote_code=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    raise RerankerUnavailable(f"加载重排模型失败：{exc}") from exc
            return self._model

    def warmup(self) -> None:
        """预热重排模型。"""
        try:
            self.score("预热查询", ["预热文本"])
        except RerankerUnavailable:
            logger.warning("重排模型预热失败", exc_info=True)
            raise

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """对候选 passage 打 0..1 相关性分。

        Args:
            query: 原始查询。
            passages: 候选文本列表。

        Returns:
            float32 分数数组，长度与 passages 一致。

        Raises:
            RerankerUnavailable: 模型不可用/预测失败。
        """
        if not passages:
            return np.asarray([], dtype=np.float32)
        try:
            logger.debug("重排 count=%d", len(passages))
            raw = self.model.predict(
                [[query, passage] for passage in passages],
                batch_size=self.settings.rerank_batch_size,
                show_progress_bar=False,
            )
            self._loaded = True
        except RerankerUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RerankerUnavailable(f"重排打分失败：{exc}") from exc
        array = np.asarray(raw, dtype=np.float32).reshape(-1)
        return np.clip(array, 0.0, 1.0)

    def health(self) -> dict:
        """返回重排模型 ready/available/device。"""
        return {
            "ready": self._loaded,
            "available": self.settings.rerank_model_path.is_dir(),
            "path": str(self.settings.rerank_model_path),
            "device": self.settings.device,
        }

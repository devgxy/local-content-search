"""BGE / EmbeddingGemma 2 文本 embedding 服务。

加载策略：本地路径 + 离线；Gemma 只加载文本塔，支持 128/256/512/768
MRL 截断；首次加载在 RLock 内完成，避免并发重复加载。

Step: STEP-02 / P0
Doc: docs/09-嵌入模型P0实施记录.md
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from app.config import Settings

logger = logging.getLogger(__name__)

BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
GEMMA_QUERY_PREFIX = "task: search result | query: "
GEMMA_DOC_TEMPLATE = "title: {title} | text: {text}"


class ModelNotReady(RuntimeError):
    """模型缺失或加载失败。"""

    code = "MODEL_NOT_READY"


def _check_model_dir(path: Path) -> None:
    """校验模型目录含 config 与权重。

    Args:
        path: 本地模型目录。

    Raises:
        ModelNotReady: 目录/配置/权重缺失。
    """
    if not path.is_dir():
        raise ModelNotReady(f"模型目录不存在：{path}")
    if not (path / "config.json").exists():
        raise ModelNotReady(f"模型目录缺少 config.json：{path}")
    if not (path / "pytorch_model.bin").exists() and not (path / "model.safetensors").exists():
        raise ModelNotReady(f"模型目录缺少权重文件：{path}")


def query_prompt(settings: Settings) -> str:
    """返回 provider 的查询前缀（自定义优先）。"""
    if settings.embed_query_prompt:
        return settings.embed_query_prompt
    if settings.embed_provider == "gemma":
        return GEMMA_QUERY_PREFIX
    return BGE_QUERY_INSTRUCTION


def document_text(settings: Settings, text: str, title: str = "") -> str:
    """按 provider 模板包装文档标题与正文。

    Args:
        settings: 配置。
        text: chunk 正文。
        title: 文档标题；空标题用 none 占位。

    Returns:
        Gemma 为 ``title: ... | text: ...``；BGE 原样返回。
    """
    template = settings.embed_doc_template
    if not template and settings.embed_provider == "gemma":
        template = GEMMA_DOC_TEMPLATE
    if not template:
        return text
    return template.format(title=title or "none", text=text)


def create_embedding_function(settings: Settings):
    """创建 LanceDB schema 用 embedding function（不加载权重）。"""
    _check_model_dir(settings.embed_model_path)
    try:
        from lancedb.embeddings import get_registry

        return get_registry().get("sentence-transformers").create(
            name=str(settings.embed_model_path),
            device=settings.device,
            normalize=True,
            trust_remote_code=False,
            max_retries=1,
        )
    except ModelNotReady:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ModelNotReady(f"创建 embedding function 失败：{exc}") from exc


def _load_model(settings: Settings):
    """加载原始 SentenceTransformer；Gemma 只加载文本塔。

    Args:
        settings: 配置（模型路径/provider/dtype/dim）。

    Returns:
        SentenceTransformer。

    Raises:
        ModelNotReady: 权重不支持/加载失败/dtype 不支持。
    """
    _check_model_dir(settings.embed_model_path)
    logger.info(
        "加载 embedding provider=%s model=%s dim=%d dtype=%s device=%s",
        settings.embed_provider,
        settings.embed_model,
        settings.embed_dim,
        settings.embed_dtype,
        settings.device,
    )
    try:
        from sentence_transformers import SentenceTransformer

        kwargs = {
            "model_name_or_path": str(settings.embed_model_path),
            "device": settings.device,
            "local_files_only": True,
            "trust_remote_code": False,
        }
        if settings.embed_provider == "gemma":
            # 文本检索不需要视觉/音频塔，避免 740M 全量加载；
            # MRL 通过 truncate_dim 截断输出，不减少前向计算。
            kwargs["config_kwargs"] = {"vision_config": None, "audio_config": None}
            if settings.embed_dim < 768:
                # EmbeddingGemma 2 的 Matryoshka 截断（128/256/512/768）。
                kwargs["truncate_dim"] = settings.embed_dim
        if settings.embed_dtype == "bfloat16":
            import torch

            if not hasattr(torch, "bfloat16"):
                raise ModelNotReady("当前 PyTorch 不支持 bfloat16")
            kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
        model = SentenceTransformer(**kwargs)
        logger.info("embedding 加载完成 provider=%s", settings.embed_provider)
        return model
    except ModelNotReady:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ModelNotReady(
            f"加载嵌入模型失败：{exc}；若为 Gemma，请确认 transformers 支持 "
            "embedding_gemma2，或先使用 EMBED_DTYPE=float32"
        ) from exc


class EmbeddingService:
    """文本 embedding 服务：加载、编码、健康检查。"""
    def __init__(self, settings: Settings):
        """保存配置；模型延迟到首次 encode/health 时加载。

        Args:
            settings: 运行期配置。
        """
        self.settings = settings
        self._func = None
        self._model = None
        self._loaded = False
        self._lock = threading.RLock()

    @property
    def func(self):
        """LanceDB schema 用 embedding function（不加载权重）。"""
        with self._lock:
            if self._func is None:
                self._func = create_embedding_function(self.settings)
            return self._func

    @property
    def model(self):
        # 权重加载也放在锁内，避免预热与首查并发加载两份模型。
        """懒加载 SentenceTransformer。"""
        with self._lock:
            if self._model is None:
                self._model = _load_model(self.settings)
            return self._model

    def warmup(self) -> None:
        """预热模型，失败抛 ModelNotReady。"""
        try:
            self.encode_documents(["预热文本"])
            self._loaded = True
        except ModelNotReady:
            logger.warning("向量模型预热失败", exc_info=True)
            raise

    def encode_query(self, query: str) -> np.ndarray:
        """编码查询并归一化。

        Args:
            query: 原始查询。

        Returns:
            float32、单位范数向量。
        """
        text = query_prompt(self.settings) + query
        try:
            logger.debug("编码查询 query_len=%d", len(query))
            vector = self.model.encode(
                [text],
                batch_size=1,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            self._loaded = True
        except Exception as exc:  # noqa: BLE001
            raise ModelNotReady(f"查询编码失败：{exc}") from exc
        array = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(array))
        return array / norm if norm else array

    def encode_documents(
        self,
        texts: list[str],
        titles: list[str] | None = None,
    ) -> list[list[float]]:
        """批量编码文档，可选标题模板。"""
        if not texts:
            return []
        if titles is None:
            titles = [""] * len(texts)
        if len(titles) != len(texts):
            raise ValueError("titles 与 texts 数量不一致")
        prepared = [
            document_text(self.settings, text, title)
            for text, title in zip(texts, titles, strict=True)
        ]
        try:
            logger.debug("编码文档 count=%d provider=%s", len(prepared), self.settings.embed_provider)
            array = self.model.encode(
                prepared,
                batch_size=self.settings.embed_batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            self._loaded = True
        except Exception as exc:  # noqa: BLE001
            raise ModelNotReady(f"文档编码失败：{exc}") from exc
        array = np.asarray(array, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        array = array / np.where(norms == 0, 1.0, norms)
        return array.tolist()

    def health(self) -> dict:
        """返回模型可用/维度/provider/设备。"""
        return {
            "ready": self._loaded,
            "available": self.settings.embed_model_path.is_dir(),
            "path": str(self.settings.embed_model_path),
            "device": self.settings.device,
            "dim": self.settings.embed_dim,
            "provider": self.settings.embed_provider,
            "dtype": self.settings.embed_dtype,
        }

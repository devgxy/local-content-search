"""图片语义向量：独立 image_assets 表 + Gemma 视觉塔。

只有 provider=gemma、IMAGE_EMBED_ENABLED=1 且维度=768 时启用；
文本检索仍走 chunks，图片命中追加到结果尾部。

Step: P1
Doc: docs/10-P1实施记录.md
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
from lancedb.pydantic import LanceModel, Vector

from app.config import Settings
from app.embeddings import ModelNotReady, _check_model_dir

logger = logging.getLogger(__name__)
ASSET_TABLE = "image_assets"
IMAGE_DIM = 768


def build_asset_model(dim: int = IMAGE_DIM) -> type[LanceModel]:
    """构造 image_assets 的 LanceModel。

    Args:
        dim: 图片向量维度，默认 768。

    Returns:
        带 vector 列的 LanceModel 子类。
    """
    class ImageAsset(LanceModel):
        """图片向量行。

        Attributes:
            id: 稳定 asset ID。
            doc_id: 文档 ID。
            root_id: 索引根。
            file_path/file_name: 原图路径/文件名。
            title: 文档标题。
            text: OCR 文本或标题，供结果摘要。
            file_type: 固定 image。
            page/slide: 预留页码。
            vector: 768 维图片向量。
            run_id: 入库运行 ID。
            indexed_at: 写入时间戳。
        """

        id: str
        doc_id: str
        root_id: int
        file_path: str
        file_name: str
        title: str = ""
        text: str = ""
        file_type: str = "image"
        page: int = 0
        slide: int = 0
        vector: Vector(dim)
        run_id: str = ""
        indexed_at: float = 0.0

    return ImageAsset


def open_asset_table(db, settings: Settings):  # noqa: ANN001
    """按配置打开/创建 image_assets 表。

    Args:
        db: LanceDB 连接。
        settings: 配置；非 Gemma/768/未启用时返回 None。

    Returns:
        LanceTable 或 None。
    """
    if (
        settings.embed_provider != "gemma"
        or not settings.image_embed_enabled
        or settings.embed_dim != IMAGE_DIM
    ):
        return None
    model = build_asset_model(IMAGE_DIM)
    names = set(db.table_names())
    if ASSET_TABLE in names:
        return db.open_table(ASSET_TABLE)
    return db.create_table(ASSET_TABLE, schema=model)


class ImageEmbeddingService:
    """Gemma 视觉塔懒加载与图片编码。"""
    def __init__(self, settings: Settings):
        """保存配置；视觉模型延迟到首次编码时加载。

        Args:
            settings: 运行期配置。
        """
        self.settings = settings
        self._model = None
        self._lock = threading.RLock()

    @property
    def model(self):
        """懒加载 Gemma 全量模型（含视觉塔）。"""
        with self._lock:
            if self._model is None:
                _check_model_dir(self.settings.embed_model_path)
                logger.info("加载 Gemma 视觉塔 path=%s", self.settings.embed_model_path)
                try:
                    import torch
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(
                        str(self.settings.embed_model_path),
                        device=self.settings.device,
                        local_files_only=True,
                        trust_remote_code=False,
                        model_kwargs={"torch_dtype": torch.bfloat16},
                    )
                except Exception as exc:  # noqa: BLE001
                    raise ModelNotReady(f"加载 Gemma 视觉塔失败：{exc}") from exc
            return self._model

    def encode_image(self, path: Path) -> list[float]:
        """编码单张图片并归一化。

        Args:
            path: 图片绝对路径。

        Returns:
            768 维单位向量。

        Raises:
            ModelNotReady: 图片不可读或视觉模型失败。
        """
        from PIL import Image

        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
            vector = self.model.encode(
                {"image": rgb},
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelNotReady(f"图片编码失败：{exc}") from exc
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(array))
        logger.debug("图片编码完成 path=%s dim=%d", path.name, len(array))
        return (array / norm if norm else array).tolist()

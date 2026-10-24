"""P1 图片语义索引测试（fake embedding，不加载 Gemma）。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from app.config import Settings
from app.db import connect_db
from app.ids import make_doc_id
from app.image_index import build_asset_model, open_asset_table
from app.ingest import IngestService
from app.manifest import Manifest
from app.schema import build_chunk_model
from app.search import SearchService


class FakeTextEmbedder:
    def __init__(self, func, dim: int = 1024) -> None:
        self.func = func
        self.dim = dim

    def encode_query(self, query: str):
        return np.ones(self.dim, dtype=np.float32) / np.sqrt(self.dim)

    def encode_documents(self, texts, titles=None):  # noqa: ANN001
        return [self.encode_query(text).tolist() for text in texts]


class FakeImageEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def encode_image(self, path: Path) -> list[float]:
        self.calls += 1
        return [0.1] * 768


class FakeReranker:
    def score(self, query, texts):  # noqa: ANN001
        return [0.0] * len(texts)


def test_asset_table_disabled_for_bge(tmp_settings: Settings, fake_func) -> None:
    db = connect_db(tmp_settings)
    assert open_asset_table(db, tmp_settings) is None


def test_image_ingest_and_delete(tmp_settings: Settings, fake_func, fixtures_dir: Path) -> None:
    db = connect_db(tmp_settings)
    table = db.create_table("chunks", schema=build_chunk_model(fake_func))
    assets = db.create_table("image_assets", schema=build_asset_model(768))
    manifest = Manifest(tmp_settings.manifest_path)
    image_embedder = FakeImageEmbedder()
    service = IngestService(
        tmp_settings,
        table,
        manifest,
        FakeTextEmbedder(fake_func),
        asset_table=assets,
        image_embedder=image_embedder,
    )
    path = fixtures_dir / "sample.png"
    result = service.ingest_paths([path])
    assert result.images == 1
    assert assets.count_rows() == 1
    assert image_embedder.calls == 1
    row = assets.search().limit(1).to_list()[0]
    assert row["file_name"] == "sample.png"
    assert len(row["vector"]) == 768
    assert service.delete_paths([path])["deleted"] == 1
    assert assets.count_rows() == 0
    manifest.close()


def test_image_search_merges_asset(tmp_settings: Settings, fake_func) -> None:
    settings = replace(tmp_settings, embed_dim=768)
    shot_path = Path("/tmp/shot.png").resolve()
    db = connect_db(settings)
    table = db.create_table("chunks", schema=build_chunk_model(fake_func, 768))
    assets = db.create_table("image_assets", schema=build_asset_model(768))
    assets.add(
        [
            {
                "id": "img1",
                "doc_id": make_doc_id(shot_path),
                "root_id": 0,
                "file_path": "/tmp/shot.png",
                "file_name": "shot.png",
                "title": "shot",
                "text": "搜索结果截图",
                "file_type": "image",
                "page": 0,
                "slide": 0,
                "vector": [0.1] * 768,
                "run_id": "r1",
                "indexed_at": 0.0,
            }
        ]
    )
    manifest = Manifest(settings.manifest_path)
    manifest.upsert_pending(shot_path, 0, "image", 0, 0)
    manifest.mark_indexed(
        str(shot_path),
        chunk_count=0,
        doc_hash="x",
        size=0,
        mtime=0,
        run_id="r1",
    )
    service = SearchService(
        settings,
        table,
        FakeTextEmbedder(fake_func, 768),
        FakeReranker(),
        manifest,
        assets,
        FakeImageEmbedder(),
    )
    result = service.search("截图", top_k=10, final_k=5, mode="vector")
    assert [hit.score_kind for hit in result.results] == ["image"]
    assert result.results[0].file_name == "shot.png"
    manifest.close()

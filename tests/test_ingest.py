"""STEP-06 入库流水线测试。"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from app.config import Settings
from app.db import connect_db, open_table
from app.ingest import IngestService
from app.jobs import JobCancelled
from app.manifest import Manifest


class FakeEmbedder:
    def __init__(self, func):
        self.func = func

    def encode_documents(self, texts: list[str], titles: list[str] | None = None) -> list[list[float]]:
        return self.func.generate_embeddings(texts)


@pytest.fixture()
def stack(tmp_settings: Settings, fake_func):
    manifest = Manifest(tmp_settings.manifest_path)
    table = open_table(connect_db(tmp_settings), tmp_settings, fake_func)
    service = IngestService(tmp_settings, table, manifest, FakeEmbedder(fake_func))
    yield service, table, manifest
    manifest.close()


def test_ingest_and_idempotent(tmp_path: Path, stack) -> None:
    service, table, manifest = stack
    a = tmp_path / "a.md"
    b = tmp_path / "b.txt"
    a.write_text("# 标题\n\n混合检索系统。", encoding="utf-8")
    b.write_text("普通文本内容。", encoding="utf-8")

    result = service.ingest_paths([a, b])
    assert result.indexed == 2
    assert result.failed == 0
    assert result.chunks == table.count_rows()
    assert manifest.get_file(str(a.resolve())).status == "indexed"
    assert manifest.stats()["last_sync_at"] is not None

    first_count = table.count_rows()
    again = service.ingest_paths([a, b])
    assert again.indexed == 0
    assert again.skipped == 2
    assert table.count_rows() == first_count


def test_update_removes_stale_chunks(tmp_path: Path, stack) -> None:
    service, table, manifest = stack
    path = tmp_path / "a.md"
    path.write_text("# 标题\n\n第一版内容。", encoding="utf-8")
    service.ingest_paths([path])
    first = table.count_rows()

    path.write_text("# 标题\n\n第二版内容，更长一些。" * 30, encoding="utf-8")
    result = service.ingest_paths([path])
    assert result.indexed == 1
    record = manifest.get_file(str(path.resolve()))
    assert record.chunk_count == table.count_rows()
    assert table.count_rows() >= first


def test_delete(tmp_path: Path, stack) -> None:
    service, table, manifest = stack
    path = tmp_path / "a.md"
    path.write_text("# 标题\n\n内容。", encoding="utf-8")
    service.ingest_paths([path])
    assert table.count_rows() > 0

    result = service.delete_paths([path])
    assert result["deleted"] == 1
    assert table.count_rows() == 0
    assert manifest.get_file(str(path.resolve())).status == "deleted"


def test_bad_file_isolated(tmp_path: Path, stack) -> None:
    service, _, manifest = stack
    good = tmp_path / "good.md"
    bad = tmp_path / "bad.docx"
    good.write_text("正常内容。", encoding="utf-8")
    bad.write_bytes(b"not a docx")

    result = service.ingest_paths([bad, good])
    assert result.indexed == 1
    assert result.failed == 1
    assert result.errors[0].code in {"BROKEN_FILE", "PARSE_FAILED"}
    assert manifest.get_file(str(good.resolve())).status == "indexed"
    assert manifest.get_file(str(bad.resolve())).status == "error"


def test_empty_document(tmp_path: Path, stack) -> None:
    service, _, manifest = stack
    path = tmp_path / "empty.md"
    path.write_text("", encoding="utf-8")
    result = service.ingest_paths([path])
    assert result.failed == 1
    assert result.errors[0].code == "EMPTY_DOCUMENT"
    assert manifest.get_file(str(path.resolve())).status == "error"


def test_progress_and_cancel(tmp_path: Path, stack) -> None:
    service, _, _ = stack
    path = tmp_path / "a.md"
    path.write_text("内容。", encoding="utf-8")
    seen: list[tuple[int, int]] = []
    service.ingest_paths([path], progress=lambda d, t, m: seen.append((d, t)))
    assert seen[-1] == (1, 1)

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(JobCancelled):
        service.ingest_paths([path], cancel=cancel)


def test_require_roots_rejects_outside(tmp_path: Path, stack) -> None:
    service, _, _ = stack
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    service.require_roots = True
    result = service.ingest_paths([outside])
    assert result.failed == 1
    assert result.errors[0].code == "INVALID_PATH"


def test_delete_only_missing(tmp_path: Path, stack) -> None:
    service, table, _ = stack
    path = tmp_path / "a.md"
    path.write_text("内容。", encoding="utf-8")
    service.ingest_paths([path])
    before = table.count_rows()
    result = service.delete_paths([path], only_missing=True)
    assert result["deleted"] == 0
    assert result["skipped"] == 1
    assert table.count_rows() == before
    result = service.delete_paths([path])
    assert result["deleted"] == 1
    assert table.count_rows() == 0


def test_symlink_delete_does_not_hit_target(tmp_path: Path, stack) -> None:
    service, table, manifest = stack
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("A 文档内容。", encoding="utf-8")
    b.write_text("B 文档内容。", encoding="utf-8")
    service.ingest_paths([a, b])
    a.unlink()
    a.symlink_to(b)
    result = service.delete_paths([a])
    assert result["deleted"] == 1
    assert manifest.get_file(str(a)).status == "deleted"
    assert manifest.get_file(str(b)).status == "indexed"
    assert table.count_rows() == 1


def test_touch_does_not_reembed(tmp_path: Path, stack) -> None:
    service, table, _ = stack
    path = tmp_path / "a.md"
    path.write_text("内容。", encoding="utf-8")
    service.ingest_paths([path])
    count = table.count_rows()
    future = path.stat().st_mtime + 10
    os.utime(path, (future, future))
    result = service.ingest_paths([path])
    assert result.indexed == 0
    assert result.skipped == 1
    assert table.count_rows() == count


def test_force_reindex_ignores_hash(tmp_path: Path, stack) -> None:
    service, table, _ = stack
    path = tmp_path / "a.md"
    path.write_text("内容。", encoding="utf-8")
    service.ingest_paths([path])
    count = table.count_rows()
    result = service.ingest_paths([path], force=True)
    assert result.indexed == 1
    assert result.skipped == 0
    assert table.count_rows() == count


def test_ingest_rejects_symlink(tmp_path: Path, stack) -> None:
    service, _, _ = stack
    service.manifest.add_root(tmp_path)
    real = tmp_path / "real.md"
    real.write_text("真实内容。", encoding="utf-8")
    link = tmp_path / "link.md"
    link.symlink_to(real)
    result = service.ingest_paths([link])
    assert result.failed == 1
    assert result.errors[0].code == "INVALID_PATH"


def test_detects_file_change_during_parse(tmp_path: Path, stack, monkeypatch) -> None:
    from app.parsing.base import ParsedBlock, ParsedDocument, finalize_document

    service, _, manifest = stack
    service.manifest.add_root(tmp_path)
    path = tmp_path / "a.md"
    path.write_text("初始内容。", encoding="utf-8")

    class MutatingParser:
        def parse(self, target, settings):  # noqa: ANN001
            target.write_text("解析期间被修改。", encoding="utf-8")
            return finalize_document(
                ParsedDocument(
                    path=target,
                    file_type="md",
                    title="a",
                    blocks=[ParsedBlock(text="初始内容。", kind="paragraph")],
                )
            )

    monkeypatch.setattr("app.ingest.get_parser", lambda _path: MutatingParser())
    result = service.ingest_paths([path])
    assert result.failed == 1
    assert result.errors[0].code == "FILE_CHANGED"
    assert manifest.get_file(str(path)).status == "error"


def test_encode_failure_keeps_old_run(tmp_path: Path, stack) -> None:
    service, table, manifest = stack
    path = tmp_path / "a.md"
    path.write_text("# 标题\n\n旧内容。", encoding="utf-8")
    service.ingest_paths([path])
    old = manifest.get_file(str(path.resolve()))
    assert old is not None
    old_run = old.run_id
    old_count = table.count_rows()

    def boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("encode failed")

    service.embedder.encode_documents = boom  # type: ignore[method-assign]
    path.write_text("# 标题\n\n新内容，但编码失败。" * 20, encoding="utf-8")
    result = service.ingest_paths([path])

    assert result.failed == 1
    record = manifest.get_file(str(path.resolve()))
    assert record is not None
    assert record.status == "error"
    assert record.run_id == old_run
    assert table.count_rows() == old_count


def test_structured_file_types_persisted(tmp_path: Path, stack) -> None:
    service, _, manifest = stack
    json_path = tmp_path / "a.json"
    csv_path = tmp_path / "b.csv"
    json_path.write_text('{"name":"检索"}', encoding="utf-8")
    csv_path.write_text("module,desc\nfts,keyword\n", encoding="utf-8")
    result = service.ingest_paths([json_path, csv_path])
    assert result.indexed == 2
    assert manifest.get_file(str(json_path.resolve())).file_type == "json"
    assert manifest.get_file(str(csv_path.resolve())).file_type == "csv"


def test_mark_error_uses_resolved_path(tmp_path: Path, stack) -> None:
    service, _, manifest = stack
    real = tmp_path / "real.md"
    real.write_text("内容。", encoding="utf-8")
    link = tmp_path / "link.md"
    link.symlink_to(real)
    service._mark_error(link, "TEST_ERROR", "path test")
    record = manifest.get_file(str(real.resolve()))
    assert record is not None
    assert record.status == "error"
    assert "TEST_ERROR" in (record.last_error or "")


def test_merge_failure_keeps_old_version(tmp_path: Path, stack, monkeypatch) -> None:
    service, table, manifest = stack
    path = tmp_path / "a.md"
    path.write_text("# 标题\n\n旧内容。", encoding="utf-8")
    service.ingest_paths([path])
    old = manifest.get_file(str(path.resolve()))
    old_count = table.count_rows()
    assert old is not None

    def boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("merge failed")

    monkeypatch.setattr(table, "merge_insert", boom)
    path.write_text("# 标题\n\n新内容。" * 20, encoding="utf-8")
    result = service.ingest_paths([path])

    assert result.failed == 1
    record = manifest.get_file(str(path.resolve()))
    assert record is not None
    assert record.run_id == old.run_id
    assert table.count_rows() == old_count

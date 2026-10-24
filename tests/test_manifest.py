"""STEP-05 清单与扫描测试。"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest

from app.config import Settings
from app.ids import sha256_file
from app.manifest import Manifest


@pytest.fixture()
def manifest(tmp_settings: Settings):
    store = Manifest(tmp_settings.manifest_path)
    yield store
    store.close()


def _make_tree(root: Path) -> tuple[Path, Path]:
    (root / "sub").mkdir(parents=True)
    (root / ".git").mkdir()
    a = root / "a.md"
    b = root / "sub" / "b.txt"
    a.write_text("# A\n\n混合检索", encoding="utf-8")
    b.write_text("普通文本", encoding="utf-8")
    (root / ".hidden.md").write_text("hidden", encoding="utf-8")
    (root / ".git" / "c.md").write_text("git", encoding="utf-8")
    (root / "build").mkdir()
    (root / "build" / "artifact.md").write_text("build artifact", encoding="utf-8")
    (root / ".searchignore").write_text("secret.pdf\n", encoding="utf-8")
    (root / "secret.pdf").write_bytes(b"%PDF-1.4")
    (root / "d.exe").write_bytes(b"x")
    return a, b


def test_add_remove_root(tmp_path: Path, manifest: Manifest) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    record = manifest.add_root(root)
    assert record.enabled is True
    assert manifest.list_roots()[0].path == str(root.resolve())

    paths = manifest.remove_root(record.id, delete_files=False)
    assert paths == []
    assert manifest.list_roots()[0].enabled is False


def test_scan_added_modified_deleted(
    tmp_path: Path, manifest: Manifest, tmp_settings: Settings
) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    a, b = _make_tree(root)
    manifest.add_root(root)

    diff = manifest.scan(tmp_settings)
    added = {item.path for item in diff.added}
    assert added == {str(a.resolve()), str(b.resolve())}
    assert diff.deleted == []
    assert diff.unchanged == 0

    for path in added:
        manifest.upsert_pending(
            path,
            manifest.list_roots()[0].id,
            "md" if path.endswith(".md") else "txt",
            Path(path).stat().st_size,
            Path(path).stat().st_mtime,
        )
        manifest.mark_indexed(
            path,
            chunk_count=1,
            doc_hash=sha256_file(path),
            size=Path(path).stat().st_size,
            mtime=Path(path).stat().st_mtime,
        )

    time.sleep(0.01)
    b.write_text("普通文本，已修改", encoding="utf-8")
    diff2 = manifest.scan(tmp_settings)
    assert {item.path for item in diff2.modified} == {str(b.resolve())}
    assert diff2.added == []
    assert diff2.unchanged == 1

    a.unlink()
    diff3 = manifest.scan(tmp_settings)
    assert diff3.deleted == [str(a.resolve())]


def test_recover_indexing(tmp_path: Path, manifest: Manifest) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    a = root / "a.md"
    a.write_text("x", encoding="utf-8")
    manifest.add_root(root)
    manifest.upsert_pending(a, 1, "md", a.stat().st_size, a.stat().st_mtime)
    manifest.mark_indexing(str(a.resolve()), "run1")
    stale = manifest.recover_indexing()
    assert stale == [str(a.resolve())]
    assert manifest.get_file(str(a.resolve())).status == "error"


def test_needs_reindex(tmp_path: Path, manifest: Manifest) -> None:
    path = tmp_path / "a.md"
    path.write_text("abc", encoding="utf-8")
    manifest.upsert_pending(path, 1, "md", path.stat().st_size, path.stat().st_mtime)
    assert manifest.needs_reindex(path) is True
    manifest.mark_indexed(
        str(path.resolve()),
        chunk_count=1,
        doc_hash=sha256_file(path),
        size=path.stat().st_size,
        mtime=path.stat().st_mtime,
    )
    assert manifest.needs_reindex(path) is False
    path.write_text("abcd", encoding="utf-8")
    assert manifest.needs_reindex(path) is True


def test_stats_and_remove_files(tmp_path: Path, manifest: Manifest) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    a = root / "a.md"
    a.write_text("x", encoding="utf-8")
    record = manifest.add_root(root)
    manifest.upsert_pending(a, record.id, "md", a.stat().st_size, a.stat().st_mtime)
    manifest.mark_indexed(
        str(a.resolve()),
        chunk_count=2,
        doc_hash=sha256_file(a),
        size=a.stat().st_size,
        mtime=a.stat().st_mtime,
    )
    stats = manifest.stats()
    assert stats["files"] == 1
    assert stats["by_status"]["indexed"] == 1
    assert stats["roots"] == 1

    paths = manifest.remove_root(record.id, delete_files=True)
    assert paths == [str(a.resolve())]
    assert manifest.list_roots()[0].enabled is False


def test_oversize_known_file_not_deleted(
    tmp_path: Path, manifest: Manifest, tmp_settings: Settings
) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    path = root / "a.md"
    path.write_text("x", encoding="utf-8")
    record = manifest.add_root(root)
    manifest.upsert_pending(path, record.id, "md", path.stat().st_size, path.stat().st_mtime)
    manifest.mark_indexed(
        str(path.resolve()),
        chunk_count=1,
        doc_hash=sha256_file(path),
        size=path.stat().st_size,
        mtime=path.stat().st_mtime,
    )
    path.write_bytes(b"x" * (2 * 1024 * 1024))
    small = replace(tmp_settings, max_file_mb=1)
    diff = manifest.scan(small)
    assert str(path.resolve()) not in diff.deleted
    assert any(error.path == str(path.resolve()) for error in diff.errors)


def test_missing_root_does_not_delete(tmp_path: Path, manifest: Manifest) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    path = root / "a.md"
    path.write_text("x", encoding="utf-8")
    record = manifest.add_root(root)
    manifest.upsert_pending(path, record.id, "md", path.stat().st_size, path.stat().st_mtime)
    manifest.mark_indexed(
        str(path.resolve()),
        chunk_count=1,
        doc_hash=sha256_file(path),
        size=path.stat().st_size,
        mtime=path.stat().st_mtime,
    )
    root.rename(tmp_path / "moved")
    diff = manifest.scan(Settings(data_dir=tmp_path / "data"))
    assert diff.deleted == []
    assert any("索引根不存在" in error.message for error in diff.errors)


def test_error_status_is_retried(
    tmp_path: Path, manifest: Manifest, tmp_settings: Settings
) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    path = root / "a.md"
    path.write_text("x", encoding="utf-8")
    record = manifest.add_root(root)
    manifest.upsert_pending(path, record.id, "md", path.stat().st_size, path.stat().st_mtime)
    manifest.mark_error(str(path.resolve()), "PARSE_FAILED", "boom")
    diff = manifest.scan(tmp_settings)
    assert str(path.resolve()) in {item.path for item in diff.modified}


def test_nested_roots_rejected(tmp_path: Path, manifest: Manifest) -> None:
    parent = tmp_path / "docs"
    child = parent / "sub"
    child.mkdir(parents=True)
    manifest.add_root(parent)
    try:
        manifest.add_root(child)
    except ValueError as exc:
        assert "嵌套" in str(exc)
    else:
        raise AssertionError("嵌套索引根应被拒绝")


def test_empty_root_does_not_delete_all(
    tmp_path: Path, manifest: Manifest, tmp_settings: Settings
) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    path = root / "a.md"
    path.write_text("x", encoding="utf-8")
    record = manifest.add_root(root)
    manifest.upsert_pending(path, record.id, "md", path.stat().st_size, path.stat().st_mtime)
    manifest.mark_indexed(
        str(path.resolve()),
        chunk_count=1,
        doc_hash=sha256_file(path),
        size=path.stat().st_size,
        mtime=path.stat().st_mtime,
    )
    path.unlink()
    diff = manifest.scan(tmp_settings)
    assert diff.deleted == []
    assert any("为空" in error.message for error in diff.errors)


def test_repair_nested_roots(tmp_path: Path) -> None:
    from app.manifest import Manifest

    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    path = tmp_path / "manifest.sqlite"
    m = Manifest(path)
    m.add_root(parent)
    # 模拟旧版本已经写入的嵌套根
    with m._lock, m._conn:
        m._conn.execute(
            "INSERT INTO roots(path, enabled, created_at) VALUES(?, 1, datetime('now'))", (str(child),)
        )
    m.close()

    try:
        Manifest(path)
    except RuntimeError as exc:
        assert "嵌套" in str(exc)
    else:
        raise AssertionError("嵌套根应阻断启动")

    repair = Manifest(path, validate_roots=False)
    repaired = repair.repair_nested_roots()
    assert repaired == [{"path": str(child), "root_id": 2, "parent_root_id": 1}]
    repair.close()
    assert Manifest(path).list_roots()

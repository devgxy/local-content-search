"""Spotlight 候选脚本测试（不依赖 macOS Spotlight）。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from app.config import Settings
from app.manifest import Manifest


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "spotlight_candidates.py"
    spec = importlib.util.spec_from_file_location("spotlight_candidates", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_mdfind_output_filters_and_dedupes(tmp_path: Path) -> None:
    module = _load_script()
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "a.md"
    inside.write_text("x", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("x", encoding="utf-8")
    output = f"{inside}\n{inside}\n{outside}\n"
    assert module.parse_mdfind_output(output, root) == [inside.resolve()]


def test_classify_indexed_unsupported_and_missing(
    tmp_path: Path, tmp_settings: Settings
) -> None:
    module = _load_script()
    manifest = Manifest(tmp_settings.manifest_path)
    try:
        indexed = tmp_path / "a.md"
        indexed.write_text("x", encoding="utf-8")
        root = manifest.add_root(tmp_path)
        manifest.upsert_pending(indexed, root.id, "md", 1, 0.0)
        manifest.mark_indexed(
            str(indexed),
            chunk_count=1,
            doc_hash="h",
            size=1,
            mtime=0.0,
            run_id="r",
        )
        rows = {
            item["state"]: item
            for item in (
                module.classify(indexed, manifest),
                module.classify(tmp_path / "b.zip", manifest),
                module.classify(tmp_path / "missing.md", manifest),
            )
        }
        assert rows["indexed"]["indexed"] is True
        assert rows["unsupported"]["indexed"] is False
        assert rows["not_indexed"]["indexed"] is False
    finally:
        manifest.close()

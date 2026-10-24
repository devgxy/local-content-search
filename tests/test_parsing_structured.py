"""STEP-03 JSON/CSV/LOG 测试。"""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.parsing.registry import get_parser


def test_json_flattened(tmp_path: Path, tmp_settings: Settings) -> None:
    path = tmp_path / "data.json"
    path.write_text('{"name":"检索","items":[{"id":1,"ok":true}]}', encoding="utf-8")
    doc = get_parser(path).parse(path, tmp_settings)
    assert doc.file_type == "json"
    assert "name: 检索" in doc.text
    assert "items[0].id: 1" in doc.text


def test_csv_as_table(tmp_path: Path, tmp_settings: Settings) -> None:
    path = tmp_path / "data.csv"
    path.write_text("module,desc\nfts,keyword\n", encoding="utf-8")
    doc = get_parser(path).parse(path, tmp_settings)
    assert doc.file_type == "csv"
    assert doc.blocks[0].kind == "table"
    assert "module | desc" in doc.text


def test_log_as_text(tmp_path: Path, tmp_settings: Settings) -> None:
    path = tmp_path / "app.log"
    path.write_text("2026-10-23 INFO started\n2026-10-23 ERROR timeout\n", encoding="utf-8")
    doc = get_parser(path).parse(path, tmp_settings)
    assert doc.file_type == "txt"
    assert "ERROR timeout" in doc.text

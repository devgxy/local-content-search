"""STEP-00 配置/自检/ID 测试。"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from app import preflight
from app.config import Settings, load_settings
from app.ids import (
    escape_sql_literal,
    make_chunk_id,
    make_content_hash,
    make_doc_id,
    sha256_file,
    sha256_text,
)
from app.logging_setup import setup_logging


def test_paths_are_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = load_settings({"DATA_DIR": "relative-data"})
    assert settings.data_dir.is_absolute()
    assert settings.data_dir == (tmp_path / "relative-data").resolve()


def test_env_override() -> None:
    settings = load_settings(
        {
            "APP_PORT": "9001",
            "CHUNK_SIZE": "320",
            "CHUNK_OVERLAP": "40",
            "OCR_ENABLED": "false",
        }
    )
    assert settings.port == 9001
    assert settings.chunk_size == 320
    assert settings.chunk_overlap == 40
    assert settings.ocr_enabled is False


def test_invalid_port() -> None:
    with pytest.raises(ValueError):
        load_settings({"APP_PORT": "0"})


def test_invalid_chunk_overlap() -> None:
    with pytest.raises(ValueError):
        load_settings({"CHUNK_SIZE": "100", "CHUNK_OVERLAP": "100"})


def test_ensure_dirs(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "d", models_dir=tmp_path / "m")
    settings.ensure_dirs()
    assert settings.data_dir.is_dir()
    assert settings.log_dir.is_dir()


def test_ids_stable(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text("内容", encoding="utf-8")
    assert make_doc_id(path) == make_doc_id(str(path))
    assert make_content_hash("abc") == make_content_hash("abc")
    assert make_content_hash("abc") != make_content_hash("abd")
    chunk_id = make_chunk_id("doc", 0, "hash")
    assert len(chunk_id) == 32
    assert chunk_id == make_chunk_id("doc", 0, "hash")
    assert chunk_id != make_chunk_id("doc", 1, "hash")


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"hello")
    assert sha256_file(path) == sha256_text("hello")


def test_escape_sql_literal() -> None:
    assert escape_sql_literal("a'b") == "'a''b'"


def test_log_file_created(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", models_dir=tmp_path / "models")
    setup_logging(settings, force=True)
    logging.getLogger("test").info("hello")
    assert (settings.log_dir / "app.log").exists()


def test_missing_model_fails(tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight, "MIN_EMBED_BYTES", 1)
    monkeypatch.setattr(preflight, "MIN_RERANK_BYTES", 1)
    report = preflight.run_preflight(tmp_settings, check_mps=False, check_ocr=False)
    assert report.ok is False
    assert any(item.name == "ready_marker" and not item.ok for item in report.items)


def test_preflight_ok_with_fake_models(
    tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preflight, "MIN_EMBED_BYTES", 1)
    monkeypatch.setattr(preflight, "MIN_RERANK_BYTES", 1)
    tmp_settings.models_dir.mkdir(parents=True, exist_ok=True)
    (tmp_settings.models_dir / "READY.json").write_text("{}", encoding="utf-8")
    for name in ("bge-large-zh-v1.5", "bge-reranker-large"):
        model_dir = tmp_settings.models_dir / name
        model_dir.mkdir(parents=True)
        (model_dir / "pytorch_model.bin").write_bytes(b"x")
    report = preflight.run_preflight(tmp_settings, check_mps=False, check_ocr=False)
    assert report.ok is True, [item.message for item in report.errors]


def test_embedding_provider_config() -> None:
    settings = load_settings(
        {
            "EMBED_PROVIDER": "gemma",
            "EMBED_DIM": "768",
            "EMBED_DTYPE": "bfloat16",
            "EMBED_QUERY_PROMPT": "task: search result | query: ",
        }
    )
    assert settings.embed_provider == "gemma"
    assert settings.embed_dim == 768
    assert settings.embed_dtype == "bfloat16"
    assert settings.embed_query_prompt.startswith("task: search")


def test_fp16_rejected() -> None:
    with pytest.raises(ValueError):
        load_settings({"EMBED_DTYPE": "float16"})


def test_search_profile_config() -> None:
    assert load_settings({"SEARCH_PROFILE": "fast"}).search_profile == "fast"
    with pytest.raises(ValueError):
        load_settings({"SEARCH_PROFILE": "turbo"})


def test_mrl_dimensions() -> None:
    for dim in (128, 256, 512, 768):
        assert load_settings({"EMBED_PROVIDER": "gemma", "EMBED_DIM": str(dim)}).embed_dim == dim
    with pytest.raises(ValueError):
        load_settings({"EMBED_PROVIDER": "bge", "EMBED_DIM": "256"})


def test_cleaning_flags() -> None:
    settings = load_settings(
        {"CHUNK_ADAPTIVE": "false", "PDF_STRIP_HEADER_FOOTER": "false"}
    )
    assert settings.chunk_adaptive is False
    assert settings.pdf_strip_header_footer is False

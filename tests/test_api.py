"""STEP-08 API 测试。"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.config import Settings
from app.db import connect_db, open_table
from app.ids import make_doc_id
from app.jobs import JobManager
from app.manifest import Manifest


class FakeEmbedder:
    def __init__(self, func):
        self.func = func

    def encode_query(self, query: str) -> np.ndarray:
        return np.asarray(self.func.generate_embeddings([query])[0], dtype=np.float32)

    def encode_documents(self, texts: list[str], titles: list[str] | None = None) -> list[list[float]]:
        return self.func.generate_embeddings(texts)

    def health(self) -> dict:
        return {"ready": True, "available": True, "device": "cpu", "dim": 1024}


class FakeReranker:
    def score(self, query: str, passages: list[str]) -> np.ndarray:
        return np.asarray([0.9 if "E1024" in text else 0.1 for text in passages], dtype=np.float32)

    def health(self) -> dict:
        return {"ready": True, "available": True, "device": "cpu"}


def _add_document(manifest: Manifest, table, func, path: Path) -> None:
    path.write_text("# 手册\n\n错误码 E1024：连接超时。", encoding="utf-8")
    manifest.add_root(path.parent)
    root_id = manifest.list_roots()[0].id
    manifest.upsert_pending(path, root_id, "md", path.stat().st_size, path.stat().st_mtime)
    manifest.mark_indexed(
        str(path),
        chunk_count=1,
        doc_hash="doc",
        size=path.stat().st_size,
        mtime=path.stat().st_mtime,
        run_id="r1",
    )
    doc_id = make_doc_id(path)
    table.add(
        [
            {
                "id": "c1",
                "doc_id": doc_id,
                "root_id": root_id,
                "chunk_index": 0,
                "text": "错误码 E1024：连接超时。",
                "vector": func.generate_embeddings(["错误码 E1024：连接超时。"])[0],
                "display_text": "错误码 E1024：连接超时。",
                "file_name": path.name,
                "file_path": str(path.resolve()),
                "file_type": "md",
                "title": "手册",
                "heading": "",
                "page": 0,
                "slide": 0,
                "char_start": 0,
                "char_end": 12,
                "content_hash": "h1",
                "doc_hash": "doc",
                "run_id": "r1",
                "indexed_at": 0.0,
                "source_kind": "text",
                "meta_json": "{}",
            }
        ]
    )


@pytest.fixture()
def api(tmp_path: Path, tmp_settings: Settings, fake_func):
    table = open_table(connect_db(tmp_settings), tmp_settings, fake_func)
    manifest = Manifest(tmp_settings.manifest_path)
    doc = tmp_path / "docs" / "manual.md"
    doc.parent.mkdir()
    _add_document(manifest, table, fake_func, doc)
    app = create_app(
        tmp_settings,
        preflight=False,
        embedder=FakeEmbedder(fake_func),
        reranker=FakeReranker(),
        table=table,
        manifest=manifest,
        jobs=JobManager(),
    )
    with TestClient(app) as client:
        yield client, tmp_settings, manifest, table, doc


def _wait_job(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in {"done", "failed", "cancelled"}:
            return payload
        time.sleep(0.05)
    raise AssertionError("job timeout")


def test_health(api) -> None:
    client, *_ = api
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["models"]["embedder"]["ready"] is True


def test_search(api) -> None:
    client, *_ = api
    response = client.post("/api/search", json={"query": "E1024", "top_k": 10, "final_k": 3})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["results"]
    assert payload["results"][0]["file_name"] == "manual.md"
    assert "e1024" in payload["highlight_terms"]


def test_search_validation(api) -> None:
    client, *_ = api
    response = client.post("/api/search", json={"query": "x", "top_k": 5, "final_k": 10})
    assert response.status_code == 422


def test_library_and_document(api) -> None:
    client, settings, manifest, table, doc = api
    library = client.get("/api/library")
    assert library.status_code == 200
    assert library.json()["total"] == 1
    detail = client.get(f"/api/documents/{make_doc_id(doc)}")
    assert detail.status_code == 200
    assert detail.json()["chunks"][0]["chunk_id"] == "c1"


def test_roots_and_scan_sync(api, tmp_path: Path) -> None:
    client, settings, manifest, table, doc = api
    new_file = doc.parent / "new.md"
    new_file.write_text("新增内容：量子纠缠。", encoding="utf-8")

    scan = client.post("/api/scan", json={})
    assert scan.status_code == 202
    job = _wait_job(client, scan.json()["job_id"])
    assert job["status"] == "done"
    added = [item["path"] for item in job["result"]["added"]]
    assert str(new_file.resolve()) in added

    sync = client.post(
        "/api/sync",
        json={"paths": [str(new_file.resolve())], "deleted_paths": [], "delete_missing": False},
    )
    assert sync.status_code == 202, sync.text
    job = _wait_job(client, sync.json()["job_id"])
    assert job["status"] == "done"
    assert job["result"]["indexed"] == 1
    assert table.count_rows() >= 2


def test_open_outside_roots(api, tmp_path: Path) -> None:
    client, *_ = api
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    response = client.post("/api/open", json={"path": str(outside)})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_PATH"


def test_job_not_found(api) -> None:
    client, *_ = api
    assert client.get("/api/jobs/nope").status_code == 404


def test_delete_missing(api) -> None:
    client, settings, manifest, table, doc = api
    other = doc.parent / "other.md"
    other.write_text("保留文件。", encoding="utf-8")
    added = client.post("/api/sync", json={"paths": [str(other.resolve())]})
    _wait_job(client, added.json()["job_id"])

    path = str(doc.resolve())
    doc.unlink()

    scan = client.post("/api/scan", json={})
    job = _wait_job(client, scan.json()["job_id"])
    assert path in job["result"]["deleted"]

    sync = client.post(
        "/api/sync",
        json={"paths": [], "deleted_paths": [path], "delete_missing": True},
    )
    assert sync.status_code == 202, sync.text
    job = _wait_job(client, sync.json()["job_id"])
    assert job["status"] == "done"
    assert job["result"]["deleted"]["deleted"] == 1
    assert manifest.get_file(path).status == "deleted"
    assert table.count_rows() == 1


def test_open_symlink_escape(api, tmp_path: Path) -> None:
    client, settings, manifest, table, doc = api
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = doc.parent / "link.txt"
    link.symlink_to(outside)
    response = client.post("/api/open", json={"path": str(link)})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_PATH"


def test_library_filter(api) -> None:
    client, *_ = api
    hit = client.get("/api/library", params={"q": "manual"})
    assert hit.status_code == 200
    assert hit.json()["total"] == 1
    miss = client.get("/api/library", params={"q": "no-such-file"})
    assert miss.status_code == 200
    assert miss.json()["total"] == 0


def test_sync_rejects_overlap(api) -> None:
    client, settings, manifest, table, doc = api
    path = str(doc.resolve())
    response = client.post(
        "/api/sync",
        json={"paths": [path], "deleted_paths": [path], "delete_missing": True},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_REQUEST"


def test_search_fast_profile(api) -> None:
    client, *_ = api
    response = client.post(
        "/api/search",
        json={"query": "E1024", "top_k": 20, "final_k": 5, "profile": "fast"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["results"]
    assert all(item["rerank_score"] is None for item in response.json()["results"])


def test_search_keyword_mode(api) -> None:
    client, *_ = api
    response = client.post(
        "/api/search",
        json={"query": "E1024", "top_k": 10, "final_k": 3, "mode": "fts"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["results"]
    assert all(item["score_kind"] == "bm25" for item in response.json()["results"])


def test_pick_folder_endpoint(api, monkeypatch, tmp_path: Path) -> None:
    from types import SimpleNamespace

    client, *_ = api
    monkeypatch.setattr(
        "app.api.routes_sync.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=str(tmp_path), stderr=""),
    )
    response = client.post("/api/pick-folder")
    assert response.status_code == 200
    assert response.json()["path"] == str(tmp_path.resolve())


def test_preview_original_file(api) -> None:
    client, _, manifest, _, doc = api
    record = manifest.get_file(str(doc.resolve()))
    response = client.get(f"/api/documents/{record.doc_id}/file")
    assert response.status_code == 200
    assert "E1024" in response.text


def test_remove_root_deletes_index(api) -> None:
    client, _, manifest, table, _ = api
    root = manifest.list_roots()[0]
    response = client.delete(f"/api/roots/{root.id}?delete_files=true")
    assert response.status_code == 200, response.text
    payload = response.json()
    if payload.get("job_id"):
        job = _wait_job(client, payload["job_id"], timeout=10.0)
        assert job["status"] == "done"
    assert manifest.list_roots() == []
    assert table.count_rows() == 0


def test_preview_pdf_page(api, tmp_path: Path) -> None:
    import pymupdf

    client, _, manifest, _, _ = api
    path = tmp_path / "preview.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "PDF preview")
    document.save(path)
    document.close()
    root = manifest.list_roots()[0]
    manifest.upsert_pending(path, root.id, "pdf", path.stat().st_size, path.stat().st_mtime)
    doc_id = make_doc_id(path)
    manifest.mark_indexed(
        str(path),
        chunk_count=1,
        doc_hash="pdf",
        size=path.stat().st_size,
        mtime=path.stat().st_mtime,
    )
    response = client.get(f"/api/documents/{doc_id}/preview?page=1")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


def test_open_uses_neutral_working_directory(api, monkeypatch) -> None:
    from types import SimpleNamespace

    client, _, _, _, doc = api
    calls = {}

    def fake_run(command, **kwargs):
        calls["command"] = command
        calls["cwd"] = kwargs.get("cwd")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("app.api.routes_library.subprocess.run", fake_run)
    response = client.post("/api/open", json={"path": str(doc.resolve())})
    assert response.status_code == 200
    assert calls["cwd"] == str(Path.home())
    assert calls["command"][0] == "open"


def test_shutdown_requires_local_header(api) -> None:
    client, *_ = api
    response = client.post("/api/shutdown")
    assert response.status_code == 403


def test_shutdown_invokes_callback(api) -> None:
    client, *_ = api
    calls = []
    client.app.state.shutdown_callback = lambda: calls.append(True)
    response = client.post("/api/shutdown", headers={"X-Local-Action": "1"})
    assert response.status_code == 200
    assert calls == [True]


def test_search_auto_global_keeps_rerank(api) -> None:
    client, *_ = api

    class RecordingReranker:
        def __init__(self) -> None:
            self.called = False

        def score(self, query: str, passages: list[str]) -> np.ndarray:
            self.called = True
            return np.asarray([0.9] * len(passages), dtype=np.float32)

    recorder = RecordingReranker()
    client.app.state.reranker = recorder
    response = client.post("/api/search", json={"query": "E1024", "top_k": 10, "final_k": 3})
    assert response.status_code == 200, response.text
    assert recorder.called


def test_search_scoped_auto_skips_rerank(api) -> None:
    client, _, _, _, doc = api

    from app.reranker import RerankerUnavailable

    def fail_rerank(query: str, passages: list[str]):
        raise RerankerUnavailable("should not be called")

    client.app.state.reranker.score = fail_rerank
    response = client.post(
        "/api/search",
        json={
            "query": "E1024",
            "top_k": 10,
            "final_k": 3,
            "doc_ids": [make_doc_id(doc)],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["warnings"] == []

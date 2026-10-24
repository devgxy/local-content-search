"""单文件/批量入库流水线。

主流程：
1. 跨进程 write_lock 串行化写入；
2. 逐文件做 symlink/根目录/大小校验，hash 前后快照；
3. 解析 → 切片 → 批量编码；
4. merge_insert 新 run_id，再删除同 doc 旧 run_id；
5. 独立图片文件额外写 image_assets，最后更新 Manifest。

Step: STEP-06
Doc: docs/steps/STEP-06-入库流水线.md §4
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lancedb.table import LanceTable

from app.chunking import chunk_document
from app.config import Settings
from app.ids import escape_sql_literal, make_chunk_id, make_doc_id, sha256_file
from app.jobs import JobCancelled
from app.locks import write_lock
from app.manifest import Manifest
from app.parsing.base import ParseError
from app.parsing.isolated import parse_pdf_isolated
from app.parsing.registry import file_type_for, get_parser

logger = logging.getLogger(__name__)

ProgressCB = Callable[[int, int, str], None]


@dataclass
class IngestError:
    """单个文件失败记录。

    Attributes:
        path: 失败文件绝对路径。
        code: 错误码，如 FILE_CHANGED/FILE_TOO_LARGE/EMPTY_DOCUMENT。
        message: 给用户看的中文原因。
    """

    path: str
    code: str
    message: str


@dataclass
class IngestResult:
    """一次批量入库的汇总。

    Attributes:
        total: 输入文件数（去重后）。
        indexed: 成功入库文件数。
        failed: 失败文件数。
        skipped: 因 hash 相同而跳过的文件数。
        chunks: 写入/更新的文本 chunk 数。
        images: 写入的图片 asset 数。
        duration_ms: 总耗时，毫秒。
        errors: 失败明细。
    """

    total: int = 0
    indexed: int = 0
    failed: int = 0
    skipped: int = 0
    chunks: int = 0
    images: int = 0
    duration_ms: int = 0
    errors: list[IngestError] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为 API/CLI 可序列化字典。

        Returns:
            包含全部计数字段和 errors 明细的 dict。
        """
        return {
            "total": self.total,
            "indexed": self.indexed,
            "failed": self.failed,
            "skipped": self.skipped,
            "chunks": self.chunks,
            "images": self.images,
            "duration_ms": self.duration_ms,
            "errors": [asdict(item) for item in self.errors],
        }


def _root_id_for(path: Path, manifest: Manifest) -> int:
    """返回路径归属的最长前缀索引根 ID。

    Args:
        path: 已解析的文件绝对路径。
        manifest: 清单。

    Returns:
        根 ID；不属于任何启用根时返回 0。
    """
    text = str(path)
    matches = [
        root
        for root in manifest.list_roots()
        if root.enabled and (text == root.path or text.startswith(root.path + os.sep))
    ]
    return max(matches, key=lambda root: len(root.path)).id if matches else 0


class IngestService:
    """入库服务：负责解码/切片/编码/写库，不负责目录扫描。

    Attributes:
        settings: 运行期配置。
        table: chunks LanceDB 表。
        manifest: SQLite 清单。
        embedder: 文本编码器，需实现 encode_documents(texts, titles)。
        require_roots: 是否要求文件属于启用根（API 路径使用 True）。
        image_embedder: 可选 Gemma 视觉编码器。
        asset_table: 可选 image_assets 表。
    """

    def __init__(
        self,
        settings: Settings,
        table: LanceTable,
        manifest: Manifest,
        embedder,
        *,
        require_roots: bool = False,
        image_embedder=None,
        asset_table=None,
    ):
        """保存依赖并设置根目录校验策略。

        Args:
            settings: 运行期配置。
            table: chunks 表。
            manifest: 清单。
            embedder: 文本编码器。
            require_roots: 为 True 时拒绝根外路径（API/重索引）。
            image_embedder: 可选图片编码器；只有 Gemma 768 才启用。
            asset_table: 可选图片 asset 表。
        """
        self.settings = settings
        self.table = table
        self.manifest = manifest
        self.embedder = embedder
        self.require_roots = require_roots
        self.image_embedder = image_embedder
        self.asset_table = asset_table

    def ingest_paths(
        self,
        paths: list[Path | str],
        *,
        progress: ProgressCB | None = None,
        cancel=None,  # noqa: ANN001
        force: bool = False,
    ) -> IngestResult:
        """入库一批文件（带跨进程写锁）。

        Args:
            paths: 文件路径列表；内部会去重并保留原始路径身份。
            progress: 可选进度回调 ``(done, total, message)``。
            cancel: 可选取消事件，触发后抛 JobCancelled。
            force: True 时忽略同 hash 跳过优化。

        Returns:
            IngestResult 汇总。

        Raises:
            JobCancelled: 用户取消。
            OSError: 写锁/文件系统失败。
        """
        with write_lock(self.settings.data_dir / ".write.lock"):
            return self._ingest_paths(
                paths,
                progress=progress,
                cancel=cancel,
                force=force,
            )

    def _ingest_paths(
        self,
        paths: list[Path | str],
        *,
        progress: ProgressCB | None = None,
        cancel=None,  # noqa: ANN001
        force: bool = False,
    ) -> IngestResult:
        """批量入库的内部实现（调用方已持有写锁）。

        Args:
            paths: 原始路径。
            progress: 进度回调。
            cancel: 取消事件。
            force: 是否强制重嵌入。

        Returns:
            IngestResult。
        """
        started = time.perf_counter()
        logger.info("开始入库 files=%d force=%s", len(paths), force)
        unique: list[Path] = []
        seen: set[str] = set()
        for item in paths:
            # 不在批处理入口 resolve，保留路径身份；每个文件读取前再校验。
            resolved = Path(os.path.abspath(Path(item).expanduser()))
            if str(resolved) in seen:
                continue
            seen.add(str(resolved))
            unique.append(resolved)

        result = IngestResult(total=len(unique))
        for index, path in enumerate(unique, start=1):
            if cancel is not None and cancel.is_set():
                raise JobCancelled("任务已取消")
            if progress:
                progress(index - 1, len(unique), f"处理 {path.name}")
            existing = self.manifest.get_file(str(path))
            if (
                existing is not None
                and existing.status == "indexed"
                and existing.doc_hash
                and path.exists()
                and not path.is_symlink()
                and not force
            ):
                try:
                    if not self.manifest.needs_reindex(path):
                        self.manifest.mark_indexed(
                            existing.path,
                            chunk_count=existing.chunk_count,
                            doc_hash=existing.doc_hash,
                            size=path.stat().st_size,
                            mtime=path.stat().st_mtime,
                            run_id=existing.run_id,
                        )
                        result.skipped += 1
                        if progress:
                            progress(index, len(unique), f"未变化 {path.name}")
                        continue
                except OSError:
                    pass
            try:
                chunks, images = self._ingest_one(path, cancel=cancel)
                result.indexed += 1
                result.chunks += chunks
                result.images += images
            except JobCancelled:
                self._mark_error(path, "INTERRUPTED", "任务取消")
                raise
            except ParseError as exc:
                result.failed += 1
                result.errors.append(IngestError(str(path), exc.code, exc.message))
                self._mark_error(path, exc.code, exc.message)
            except Exception as exc:  # noqa: BLE001
                logger.exception("入库失败：%s", path)
                result.failed += 1
                result.errors.append(IngestError(str(path), "INGEST_FAILED", str(exc)))
                self._mark_error(path, "INGEST_FAILED", str(exc))
            if progress:
                progress(index, len(unique), path.name)

        result.duration_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "入库汇总 total=%d indexed=%d failed=%d skipped=%d chunks=%d images=%d ms=%d",
            result.total,
            result.indexed,
            result.failed,
            result.skipped,
            result.chunks,
            result.images,
            result.duration_ms,
        )
        self.manifest.record_sync_run(
            uuid.uuid4().hex,
            "ingest",
            status="done",
            added=result.indexed,
            failed=result.failed,
        )
        return result

    def _ingest_one(self, path: Path, *, cancel=None) -> tuple[int, int]:  # noqa: ANN001
        """解析/编码单个文件并写入 chunks（可选图片 asset）。

        Args:
            path: 原始路径；必须是普通文件，直接 symlink 会被拒绝。
            cancel: 取消事件，按编码 batch 检查。

        Returns:
            ``(文本 chunk 数, 图片 asset 数)``。

        Raises:
            ParseError: 文件不存在/过大/解析失败/解析期间变化。
            JobCancelled: 用户取消。
            ModelNotReady: 编码模型不可用。
        """
        raw = Path(path).expanduser()
        # 直接 symlink 可能指向根外；统一拒绝，目录内的 symlink 目录由
        # resolve 后再做 root 校验。
        if raw.is_symlink():
            raise ParseError("INVALID_PATH", f"不允许直接入库符号链接：{raw}")
        real = raw.resolve()
        if not real.exists() or not real.is_file():
            raise ParseError("FILE_NOT_FOUND", f"文件不存在：{real}")
        size = real.stat().st_size
        mtime = real.stat().st_mtime
        if size > self.settings.max_file_mb * 1024 * 1024:
            raise ParseError("FILE_TOO_LARGE", f"超过 {self.settings.max_file_mb}MB")

        if self.require_roots and _root_id_for(real, self.manifest) == 0:
            raise ParseError("INVALID_PATH", f"路径不在启用的索引根内：{real}")
        file_type = file_type_for(real)
        record = self.manifest.get_file(str(real))
        root_id = record.root_id if record else _root_id_for(real, self.manifest)
        self.manifest.upsert_pending(real, root_id, file_type, size, mtime)

        # 解析前/后各算一次 hash，防止“解析 A、落库 B”的竞态。
        hash_before = sha256_file(real)
        if real.suffix.lower() == ".pdf":
            document = parse_pdf_isolated(real, self.settings)
        else:
            document = get_parser(real).parse(real, self.settings)
        drafts = chunk_document(document, self.settings)
        if not drafts:
            raise ParseError("EMPTY_DOCUMENT", "解析/OCR 后没有可用文本")
        hash_after = sha256_file(real)
        if hash_before != hash_after:
            raise ParseError("FILE_CHANGED", "解析期间文件发生变化，已放弃本次入库")

        doc_hash = hash_before
        doc_id = make_doc_id(real)
        run_id = uuid.uuid4().hex

        # 先编码，成功后才把 manifest 切到新 run_id；
        # 编码/取消失败时旧 run_id 保持不变，旧版本仍可检索。
        # 批量编码限制 MPS 峰值；每个 batch 前检查取消，避免长文档无法中断。
        vectors: list[list[float]] = []
        batch_size = self.settings.embed_batch_size
        for start in range(0, len(drafts), batch_size):
            if cancel is not None and cancel.is_set():
                raise JobCancelled("任务已取消")
            batch = drafts[start : start + batch_size]
            vectors.extend(
                self.embedder.encode_documents(
                    [item.text for item in batch],
                    titles=[document.title] * len(batch),
                )
            )

        rows = []
        for draft, vector in zip(drafts, vectors, strict=True):
            rows.append(
                {
                    "id": make_chunk_id(doc_id, draft.chunk_index, draft.content_hash),
                    "doc_id": doc_id,
                    "root_id": root_id,
                    "chunk_index": draft.chunk_index,
                    "text": draft.text,
                    "vector": vector,
                    "display_text": draft.display_text,
                    "file_name": real.name,
                    "file_path": str(real),
                    "file_type": file_type,
                    "title": document.title,
                    "heading": draft.heading,
                    "page": draft.page,
                    "slide": draft.slide,
                    "char_start": draft.char_start,
                    "char_end": draft.char_end,
                    "content_hash": draft.content_hash,
                    "doc_hash": doc_hash,
                    "run_id": run_id,
                    "indexed_at": time.time(),
                    "source_kind": draft.source_kind,
                    "meta_json": json.dumps(
                        {
                            "warnings": document.warnings[:20],
                            "chunk_meta": draft.meta,
                        },
                        ensure_ascii=False,
                    ),
                }
            )

        # 文本行 ID 带 run_id，新 run 只插入新行，不覆盖旧版本。
        for row in rows:
            row["id"] = f"{row['id']}-{run_id}"

        image_vector = None
        if (
            file_type == "image"
            and self.asset_table is not None
            and self.image_embedder is not None
        ):
            # 图片语义编码失败必须显式失败；不能吞掉后报“入库成功”。
            image_vector = self.image_embedder.encode_image(real)

        # 1) 写新文本行；旧行仍在，manifest 仍指向旧 run，失败不会隐藏旧版本。
        builder = self.table.merge_insert("id")
        builder = builder.when_matched_update_all().when_not_matched_insert_all()
        builder.execute(rows)

        # 2) 写新图片 asset；ID 同样带 run_id。
        if image_vector is not None:
            asset_id = f"{make_chunk_id(doc_id, -1, 'image')}-{run_id}"
            self.asset_table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(
                [
                    {
                        "id": asset_id,
                        "doc_id": doc_id,
                        "root_id": root_id,
                        "file_path": str(real),
                        "file_name": real.name,
                        "title": document.title,
                        "text": document.text[:1000],
                        "file_type": "image",
                        "page": 0,
                        "slide": 0,
                        "vector": image_vector,
                        "run_id": run_id,
                        "indexed_at": time.time(),
                    }
                ]
            )

        # 3) 新文本/图片都写成功后才切 manifest 当前 run。
        self.manifest.mark_indexed(
            str(real),
            chunk_count=len(rows),
            doc_hash=doc_hash,
            size=size,
            mtime=mtime,
            run_id=run_id,
        )

        # 4) 最后清理旧版本；失败只影响垃圾行，不影响新版本可检索性。
        try:
            self.table.delete(
                f"doc_id = {escape_sql_literal(doc_id)} AND run_id != {escape_sql_literal(run_id)}"
            )
            if image_vector is not None:
                self.asset_table.delete(
                    f"doc_id = {escape_sql_literal(doc_id)} AND run_id != {escape_sql_literal(run_id)}"
                )
        except Exception:  # noqa: BLE001
            logger.warning("清理旧版本失败（新版本已提交）：%s", real, exc_info=True)

        image_count = 1 if image_vector is not None else 0
        logger.info("已入库 %s（%d chunks, %d images）", real.name, len(rows), image_count)
        return len(rows), image_count

    def delete_paths(
        self,
        paths: list[Path | str],
        *,
        progress: ProgressCB | None = None,
        cancel=None,  # noqa: ANN001
        only_missing: bool = False,
    ) -> dict:
        """删除一批文件对应的索引数据。

        Args:
            paths: 要删除的文件路径。
            progress: 进度回调。
            cancel: 取消事件。
            only_missing: True 时只删除确实缺失的文件。

        Returns:
            ``{total, deleted, failed, skipped}``。

        Raises:
            JobCancelled: 用户取消。
        """
        with write_lock(self.settings.data_dir / ".write.lock"):
            return self._delete_paths(
                paths,
                progress=progress,
                cancel=cancel,
                only_missing=only_missing,
            )

    def _delete_paths(
        self,
        paths: list[Path | str],
        *,
        progress: ProgressCB | None = None,
        cancel=None,  # noqa: ANN001
        only_missing: bool = False,
    ) -> dict:
        """删除内部实现（调用方已持有写锁）。

        Args:
            paths: 原始路径。
            progress: 进度回调。
            cancel: 取消事件。
            only_missing: 是否只处理缺失文件。

        Returns:
            删除统计 dict。
        """
        total = len(paths)
        deleted = 0
        failed = 0
        skipped = 0
        for index, item in enumerate(paths, start=1):
            if cancel is not None and cancel.is_set():
                raise JobCancelled("任务已取消")
            # 删除按清单中保存的路径身份，不解析符号链接目标。
            path = Path(os.path.abspath(Path(item).expanduser()))
            record = self.manifest.get_file(str(path))
            if record is None:
                skipped += 1
                continue
            if only_missing and path.exists() and not path.is_symlink():
                skipped += 1
                continue
            doc_id = record.doc_id
            try:
                self.table.delete(f"doc_id = {escape_sql_literal(doc_id)}")
                if self.asset_table is not None:
                    # 图片资产删除失败必须计入 failed，不能仍标记 deleted。
                    self.asset_table.delete(f"doc_id = {escape_sql_literal(doc_id)}")
                self.manifest.mark_deleted(record.path)
                deleted += 1
            except Exception as exc:  # noqa: BLE001
                logger.exception("删除失败：%s", path)
                self.manifest.mark_error(str(path), "DELETE_FAILED", str(exc))
                failed += 1
            if progress:
                progress(index, total, path.name)
        self.manifest.record_sync_run(
            uuid.uuid4().hex,
            "delete",
            status="done",
            deleted=deleted,
            failed=failed,
        )
        return {"total": total, "deleted": deleted, "failed": failed, "skipped": skipped}

    def _mark_error(self, path: Path, code: str, message: str) -> None:
        """把文件状态写为 error；统一使用 resolve 后的真实路径。

        旧逻辑用未 resolve 的 path 查询/更新，macOS /var→/private/var 或
        symlink 根下会匹配不到，反而 upsert 出 root_id=0 的 pending 行。
        """
        try:
            real = Path(path).expanduser().resolve()
            if self.manifest.get_file(str(real)) is None:
                root_id = _root_id_for(real, self.manifest)
                try:
                    fallback_type = file_type_for(real)
                except ParseError:
                    fallback_type = real.suffix.lstrip(".") or "unknown"
                self.manifest.upsert_pending(
                    real,
                    root_id,
                    fallback_type,
                    real.stat().st_size if real.exists() else 0,
                    real.stat().st_mtime if real.exists() else 0.0,
                )
            self.manifest.mark_error(str(real), code, message)
        except Exception:  # noqa: BLE001
            logger.exception("记录错误状态失败：%s", path)

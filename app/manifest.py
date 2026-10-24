"""SQLite 文件清单、扫描 diff 与状态机。

Step: STEP-05
Doc: docs/steps/STEP-05-清单与增量扫描.md
"""

from __future__ import annotations

import fnmatch
import logging
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.config import Settings
from app.ids import make_doc_id, sha256_file
from app.jobs import JobCancelled
from app.parsing.registry import SUPPORTED_EXTENSIONS, file_type_for

logger = logging.getLogger(__name__)

IGNORED_DIRS = {
    ".git",
    "node_modules",
    "Library",
    "Applications",
    "__pycache__",
    ".venv",
    ".Trash",
    # 常见构建产物：Android/Java/前端 build 输出通常不是知识文档。
    # 如需索引名为 build 的业务目录，请改这里或收窄索引根。
    "build",
}
IGNORED_SUFFIXES = {".tmp", ".part", ".crdownload"}
IGNORED_NAMES = {".DS_Store"}

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS roots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT UNIQUE NOT NULL,
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    path            TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL,
    root_id         INTEGER NOT NULL,
    file_type       TEXT NOT NULL,
    size            INTEGER NOT NULL,
    mtime           REAL NOT NULL,
    doc_hash        TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    run_id          TEXT,
    last_indexed_at TEXT,
    last_error      TEXT
);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
CREATE INDEX IF NOT EXISTS idx_files_root ON files(root_id);

CREATE TABLE IF NOT EXISTS sync_runs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    added       INTEGER DEFAULT 0,
    updated     INTEGER DEFAULT 0,
    deleted     INTEGER DEFAULT 0,
    failed      INTEGER DEFAULT 0,
    status      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass
class RootRecord:
    """索引根。

    Attributes:
        id: SQLite 主键。
        path: 绝对路径。
        enabled: 是否参与扫描；停用根仍保留历史清单。
        created_at: 创建时间（SQLite datetime 字符串）。
        doc_count: 该根下 status=indexed 的文件数。
    """

    id: int
    path: str
    enabled: bool
    created_at: str
    doc_count: int = 0


@dataclass
class FileRecord:
    """清单中的单个文件。

    Attributes:
        path: 绝对路径，作为文件身份。
        doc_id: 稳定文档 ID。
        root_id: 所属根。
        file_type: md/txt/pdf/docx/pptx/html/image。
        size: 字节数。
        mtime: Unix 时间戳。
        doc_hash: 内容 SHA-256；未入库时为 None。
        status: pending/indexing/indexed/error/deleted。
        chunk_count: 已写入 chunk 数。
        run_id: 最近一次成功入库的运行 ID。
        last_indexed_at: 最近索引时间。
        last_error: 最近错误码/原因。
    """

    path: str
    doc_id: str
    root_id: int
    file_type: str
    size: int
    mtime: float
    doc_hash: str | None
    status: str
    chunk_count: int
    run_id: str | None
    last_indexed_at: str | None
    last_error: str | None


@dataclass
class FileCandidate:
    """扫描阶段发现的文件元数据（尚未入库）。"""

    path: str
    root_id: int
    size: int
    mtime: float
    file_type: str


@dataclass
class ScanError:
    """扫描错误：文件/目录被跳过，但不影响其他路径。"""

    path: str
    message: str


@dataclass
class ScanDiff:
    """一次扫描的差异。

    Attributes:
        added: 新增文件。
        modified: 修改/需要重试的文件。
        deleted: 确认缺失的文件路径。
        unchanged: 未变化文件数。
        errors: 跳过路径及原因。
        duration_ms: 扫描耗时。
    """

    added: list[FileCandidate] = field(default_factory=list)
    modified: list[FileCandidate] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: int = 0
    errors: list[ScanError] = field(default_factory=list)
    duration_ms: int = 0

    def to_dict(self) -> dict:
        """转换为 API 可序列化字典。

        Returns:
            包含 added/modified/deleted/unchanged/errors/duration_ms 的 dict。
        """
        return {
            "added": [asdict(item) for item in self.added],
            "modified": [asdict(item) for item in self.modified],
            "deleted": list(self.deleted),
            "unchanged": self.unchanged,
            "errors": [asdict(item) for item in self.errors],
            "duration_ms": self.duration_ms,
        }


class Manifest:
    """SQLite 清单：索引根、文件状态、扫描差异、同步历史。

    线程内用 RLock 串行；跨进程由 ingest 的 flock 保护。SQLite 连接
    ``check_same_thread=False``，由锁保证单连接安全。

    Attributes:
        path: manifest.sqlite 路径。
    """

    def __init__(self, path: Path, *, validate_roots: bool = True):
        """打开/初始化 SQLite 并建表。

        Args:
            path: manifest.sqlite 路径；父目录会自动创建。
            validate_roots: 是否拒绝已存在的嵌套根；修复入口可关闭。
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            if validate_roots:
                self._validate_roots_locked()

    def _validate_roots_locked(self) -> None:
        """拒绝已存在的嵌套根，避免父/子根重复扫描同一文件。"""
        rows = self._conn.execute("SELECT id, path FROM roots WHERE enabled=1 ORDER BY id").fetchall()
        for index, left in enumerate(rows):
            for right in rows[index + 1 :]:
                a = str(left["path"])
                b = str(right["path"])
                if a.startswith(b + os.sep) or b.startswith(a + os.sep):
                    raise RuntimeError(
                        f"检测到嵌套索引根：{a} 与 {b}；请先移除其中一个根"
                    )

    def repair_nested_roots(self) -> list[dict]:
        """停用启用根中的嵌套子根，并把文件归属迁移到父根。

        Returns:
            ``[{"path", "root_id", "parent_root_id"}]``；只迁移 manifest，
            LanceDB 的 chunks/assets root_id 由 CLI 同步迁移。
        """
        with self._lock, self._conn:
            rows = self._conn.execute(
                "SELECT id, path FROM roots WHERE enabled=1 ORDER BY id"
            ).fetchall()
            repaired: list[dict] = []
            for index, left in enumerate(rows):
                for right in rows[index + 1 :]:
                    a, b = str(left["path"]), str(right["path"])
                    if not (a.startswith(b + os.sep) or b.startswith(a + os.sep)):
                        continue
                    child = right if len(b) > len(a) else left
                    parent = right if child is left else left
                    self._conn.execute(
                        "UPDATE roots SET enabled=0 WHERE id=?", (child["id"],)
                    )
                    self._conn.execute(
                        "UPDATE files SET root_id=? WHERE root_id=?",
                        (parent["id"], child["id"]),
                    )
                    repaired.append(
                        {
                            "path": str(child["path"]),
                            "root_id": child["id"],
                            "parent_root_id": parent["id"],
                        }
                    )
            return repaired

    def close(self) -> None:
        """关闭 SQLite 连接；重复关闭会抛 ProgrammingError，由调用方兜底。"""
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ roots
    def add_root(self, path: Path | str) -> RootRecord:
        """添加/重新启用索引根。

        Args:
            path: 目录路径。

        Returns:
            根记录。

        Raises:
            ValueError: 目录不存在或与现有根嵌套。
        """
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"目录不存在：{resolved}")
        text = str(resolved)
        for existing in self.list_roots():
            # 停用根不参与嵌套冲突；否则修复后仍无法重新添加。
            if not existing.enabled:
                continue
            if existing.path == text:
                break
            if text.startswith(existing.path + os.sep) or existing.path.startswith(text + os.sep):
                raise ValueError(
                    f"索引根不能嵌套：{resolved} 与 {existing.path} 存在包含关系"
                )
        logger.info("添加索引根 path=%s", resolved)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO roots(path, enabled, created_at) VALUES(?, 1, datetime('now')) "
                "ON CONFLICT(path) DO UPDATE SET enabled=1",
                (str(resolved),),
            )
            row = self._conn.execute("SELECT * FROM roots WHERE path=?", (str(resolved),)).fetchone()
        return self._root_from_row(row)

    def remove_root(self, root_id: int, delete_files: bool = False) -> list[str]:
        """停用索引根；delete_files=True 时返回待异步清理的路径。

        先停用、后异步删除，避免任务失败时丢失清单元数据。

        Args:
            root_id: 根 ID。
            delete_files: 是否返回该根下文件路径，供删除任务使用。

        Returns:
            需要删除的路径列表；delete_files=False 时为空。
        """
        logger.info("移除索引根 root_id=%s delete_files=%s", root_id, delete_files)
        with self._lock, self._conn:
            rows = self._conn.execute(
                "SELECT path FROM files WHERE root_id=?", (root_id,)
            ).fetchall()
            paths = [row["path"] for row in rows]
            self._conn.execute("UPDATE roots SET enabled=0 WHERE id=?", (root_id,))
        return paths if delete_files else []

    def drop_root(self, root_id: int) -> None:
        """删除索引根记录及已清理的文件记录（原文件不动）。

        Args:
            root_id: 根 ID；删除任务成功后才调用。
        """
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM roots WHERE id=?", (root_id,))
            self._conn.execute(
                "DELETE FROM files WHERE root_id=? AND status='deleted'", (root_id,)
            )

    def list_roots(self) -> list[RootRecord]:
        """列出所有根（含停用），并统计 indexed 文档数。

        Returns:
            RootRecord 列表，按 id 升序。
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT r.*, (SELECT COUNT(*) FROM files f WHERE f.root_id=r.id "
                "AND f.status='indexed') AS doc_count FROM roots r ORDER BY r.id"
            ).fetchall()
        return [self._root_from_row(row) for row in rows]

    @staticmethod
    def _root_from_row(row: sqlite3.Row) -> RootRecord:
        """把 SQLite 行转换为 RootRecord。

        Args:
            row: 含 roots 字段和 doc_count 的行。

        Returns:
            RootRecord。
        """
        return RootRecord(
            id=row["id"],
            path=row["path"],
            enabled=bool(row["enabled"]),
            created_at=row["created_at"],
            doc_count=row["doc_count"] if "doc_count" in row.keys() else 0,
        )

    # ------------------------------------------------------------------ files
    def get_file(self, path: str) -> FileRecord | None:
        """按绝对路径查文件记录。

        Args:
            path: 清单路径字符串。

        Returns:
            FileRecord 或 None。
        """
        with self._lock:
            row = self._conn.execute("SELECT * FROM files WHERE path=?", (str(path),)).fetchone()
        return self._file_from_row(row) if row else None

    def get_file_by_doc_id(self, doc_id: str) -> FileRecord | None:
        """按 doc_id 查未删除文件记录。

        Args:
            doc_id: 文档 ID。

        Returns:
            FileRecord 或 None。
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM files WHERE doc_id=? AND status!='deleted' LIMIT 1",
                (doc_id,),
            ).fetchone()
        return self._file_from_row(row) if row else None

    def list_files_page(
        self,
        page: int,
        page_size: int,
        *,
        file_type: str | None = None,
        q: str | None = None,
    ) -> tuple[list[FileRecord], int]:
        """按页查询文件列表，供资料库使用。

        Args:
            page: 1 起页码。
            page_size: 每页数量。
            file_type: 可选类型过滤。
            q: 可选路径子串过滤。

        Returns:
            ``(records, total)``；排序为 mtime 倒序、path 升序。
        """
        clauses = ["status != 'deleted'"]
        params: list = []
        if file_type:
            clauses.append("file_type = ?")
            params.append(file_type)
        if q:
            escaped = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            clauses.append("path LIKE ? ESCAPE '\\'")
            params.append(f"%{escaped}%")
        where = " WHERE " + " AND ".join(clauses)
        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) AS n FROM files" + where, params
            ).fetchone()["n"]
            rows = self._conn.execute(
                "SELECT * FROM files" + where + " ORDER BY mtime DESC LIMIT ? OFFSET ?",
                (*params, page_size, (page - 1) * page_size),
            ).fetchall()
        return [self._file_from_row(row) for row in rows], int(total)

    def list_files(self, status: str | None = None) -> list[FileRecord]:
        """按状态列出文件；status=None 返回全部。"""
        query = "SELECT * FROM files"
        params: tuple = ()
        if status:
            query += " WHERE status=?"
            params = (status,)
        with self._lock:
            rows = self._conn.execute(query + " ORDER BY path", params).fetchall()
        return [self._file_from_row(row) for row in rows]

    def upsert_pending(
        self,
        path: Path | str,
        root_id: int,
        file_type: str,
        size: int,
        mtime: float,
    ) -> None:
        """插入/更新 pending 记录；解析前调用。

        Args:
            path: 文件绝对路径。
            root_id: 所属根。
            file_type: 类型。
            size: 字节数。
            mtime: 修改时间。
        """
        resolved = str(Path(path).expanduser().resolve())
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO files(path, doc_id, root_id, file_type, size, mtime, status) "
                "VALUES(?, ?, ?, ?, ?, ?, 'pending') "
                "ON CONFLICT(path) DO UPDATE SET root_id=excluded.root_id, "
                "file_type=excluded.file_type, size=excluded.size, mtime=excluded.mtime, "
                "status='pending'",
                (resolved, make_doc_id(resolved), root_id, file_type, size, mtime),
            )

    def mark_indexing(self, path: str, run_id: str) -> None:
        """标记文件开始写入，记录 run_id。

        Args:
            path: 文件路径。
            run_id: 本次运行的 UUID。
        """
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE files SET status='indexing', run_id=?, last_error=NULL WHERE path=?",
                (run_id, str(path)),
            )

    def mark_indexed(
        self,
        path: str,
        *,
        chunk_count: int,
        doc_hash: str,
        size: int,
        mtime: float,
        run_id: str | None = None,
    ) -> None:
        """标记文件成功入库。

        Args:
            path: 文件路径。
            chunk_count: 文本 chunk 数。
            doc_hash: 内容 SHA-256。
            size: 字节数。
            mtime: 修改时间。
            run_id: 本次运行 ID。
        """
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE files SET status='indexed', chunk_count=?, doc_hash=?, size=?, "
                "mtime=?, run_id=?, last_indexed_at=datetime('now'), last_error=NULL "
                "WHERE path=?",
                (chunk_count, doc_hash, size, mtime, run_id, str(path)),
            )

    def mark_error(self, path: str, code: str, message: str) -> None:
        """标记文件失败；保留 last_error 供 UI/扫描重试。"""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE files SET status='error', last_error=? WHERE path=?",
                (f"{code}: {message}"[:500], str(path)),
            )

    def mark_deleted(self, path: str) -> None:
        """标记文件已删除（索引数据已清理）。"""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE files SET status='deleted', chunk_count=0 WHERE path=?",
                (str(path),),
            )

    def recover_indexing(self) -> list[str]:
        """启动时把残留 indexing 标记为 error，返回路径列表。"""
        with self._lock, self._conn:
            rows = self._conn.execute(
                "SELECT path FROM files WHERE status='indexing'"
            ).fetchall()
            paths = [row["path"] for row in rows]
            self._conn.execute(
                "UPDATE files SET status='error', last_error='INTERRUPTED: 上次入库中断' "
                "WHERE status='indexing'"
            )
        return paths

    @staticmethod
    def _file_from_row(row: sqlite3.Row) -> FileRecord:
        """把 SQLite 行转换为 FileRecord。

        Args:
            row: files 表行。

        Returns:
            FileRecord。
        """
        return FileRecord(
            path=row["path"],
            doc_id=row["doc_id"],
            root_id=row["root_id"],
            file_type=row["file_type"],
            size=row["size"],
            mtime=row["mtime"],
            doc_hash=row["doc_hash"],
            status=row["status"],
            chunk_count=row["chunk_count"],
            run_id=row["run_id"],
            last_indexed_at=row["last_indexed_at"],
            last_error=row["last_error"],
        )

    def needs_reindex(self, path: Path | str) -> bool:
        """判断文件是否需要重新解析/嵌入。

        Args:
            path: 文件路径。

        Returns:
            无记录、无 hash、hash 不一致或读取失败时返回 True。
        """
        record = self.get_file(str(Path(path).expanduser().resolve()))
        if record is None or not record.doc_hash:
            return True
        return sha256_file(path) != record.doc_hash

    # ------------------------------------------------------------------- scan
    def scan(
        self,
        settings: Settings,
        root_ids: list[int] | None = None,
        cancel=None,  # noqa: ANN001
    ) -> ScanDiff:
        """扫描启用根并计算 added/modified/deleted。

        Args:
            settings: 扫描配置（忽略规则、最大文件等）。
            root_ids: 只扫描指定根；None 表示全部启用根。
            cancel: 可选取消事件。

        Returns:
            ScanDiff；删除判定会在空根/遍历失败时保守跳过。

        Raises:
            JobCancelled: 用户取消。
        """
        started = time.perf_counter()
        diff = ScanDiff()
        logger.info("扫描开始 roots=%s", root_ids or "all")
        roots = [root for root in self.list_roots() if root.enabled]
        if root_ids:
            allowed = set(root_ids)
            roots = [root for root in roots if root.id in allowed]
        for root in roots:
            if cancel is not None and cancel.is_set():
                raise JobCancelled("任务已取消")
            seen: dict[str, tuple[int, float, str]] = {}
            root_path = Path(root.path)
            if not root_path.is_dir():
                diff.errors.append(
                    ScanError(path=root.path, message="索引根不存在或不可访问，跳过删除判定")
                )
                continue
            state = {"failed": False}
            try:
                for file_path in self._iter_files(root_path, settings, diff, state, cancel):
                    try:
                        stat = file_path.stat()
                    except OSError as exc:
                        diff.errors.append(ScanError(path=str(file_path), message=str(exc)))
                        continue
                    seen[str(file_path)] = (stat.st_size, stat.st_mtime, file_type_for(file_path))
            except OSError as exc:
                state["failed"] = True
                diff.errors.append(ScanError(path=root.path, message=str(exc)))
            walk_failed = bool(state["failed"])

            with self._lock:
                rows = self._conn.execute(
                    "SELECT path, size, mtime, status FROM files "
                    "WHERE root_id=? AND status!='deleted'",
                    (root.id,),
                ).fetchall()
            known = {
                row["path"]: (row["size"], row["mtime"], row["status"])
                for row in rows
            }

            for path, (size, mtime, file_type) in seen.items():
                candidate = FileCandidate(
                    path=path,
                    root_id=root.id,
                    size=size,
                    mtime=mtime,
                    file_type=file_type,
                )
                previous = known.get(path)
                if previous is None:
                    diff.added.append(candidate)
                elif previous[:2] != (size, mtime) or previous[2] in {
                    "pending",
                    "indexing",
                    "error",
                }:
                    diff.modified.append(candidate)
                else:
                    diff.unchanged += 1

            errored = {item.path for item in diff.errors}
            if not walk_failed and not seen and known:
                diff.errors.append(
                    ScanError(
                        path=root.path,
                        message="索引根为空或未挂载，跳过删除判定",
                    )
                )
                continue
            if not walk_failed:
                for path in known:
                    if path not in seen and path not in errored:
                        diff.deleted.append(path)

        diff.duration_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "扫描完成 added=%d modified=%d deleted=%d unchanged=%d errors=%d ms=%d",
            len(diff.added),
            len(diff.modified),
            len(diff.deleted),
            diff.unchanged,
            len(diff.errors),
            diff.duration_ms,
        )
        return diff

    @staticmethod
    def _searchignore_patterns(root: Path, settings: Settings) -> list[str]:
        """读取根目录和 data 目录下的 .searchignore（每行一个 glob）。"""
        patterns: list[str] = []
        for ignore_file in (root / ".searchignore", settings.data_dir / ".searchignore"):
            if not ignore_file.is_file():
                continue
            try:
                lines = ignore_file.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                logger.warning("读取忽略规则失败：%s（%s）", ignore_file, exc)
                continue
            patterns.extend(
                line.strip()
                for line in lines
                if line.strip() and not line.strip().startswith("#")
            )
        return patterns

    def _iter_files(
        self,
        root: Path,
        settings: Settings,
        diff: ScanDiff,
        state: dict,
        cancel=None,  # noqa: ANN001
    ):
        """遍历根下允许的文件，yield Path。

        Args:
            root: 根目录。
            settings: 配置。
            diff: 扫描差异，用于记录单文件错误。
            state: 遍历状态（walk_failed 等）。
            cancel: 可选取消事件。

        Yields:
            通过扩展名/大小/忽略规则的 Path。
        """
        def onerror(exc: OSError) -> None:
            """os.walk 级错误：标记遍历失败并记录路径。"""
            state["failed"] = True
            diff.errors.append(
                ScanError(path=str(getattr(exc, "filename", root)), message=str(exc))
            )

        ignore_patterns = self._searchignore_patterns(root, settings)
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=onerror):
            if cancel is not None and cancel.is_set():
                raise JobCancelled("任务已取消")
            dirnames[:] = [
                name
                for name in dirnames
                if name not in IGNORED_DIRS
                and not name.startswith(".")
                and not name.endswith(".app")
            ]
            for name in filenames:
                if name in IGNORED_NAMES or name.startswith(".") or name.startswith("~$"):
                    continue
                path = Path(dirpath) / name
                if path.suffix.lower() in IGNORED_SUFFIXES:
                    continue
                if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                try:
                    rel = path.relative_to(root).as_posix()
                except ValueError:
                    rel = name
                if any(
                    fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern)
                    for pattern in ignore_patterns
                ):
                    continue
                if path.is_symlink():
                    continue
                try:
                    size = path.stat().st_size
                except OSError as exc:
                    diff.errors.append(ScanError(path=str(path), message=str(exc)))
                    continue
                if size > settings.max_file_mb * 1024 * 1024:
                    diff.errors.append(
                        ScanError(
                            path=str(path),
                            message=f"超过 {settings.max_file_mb}MB，已跳过",
                        )
                    )
                    continue
                yield path

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        """返回清单统计：根数、文件数、按状态计数。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM files GROUP BY status"
            ).fetchall()
            roots = self._conn.execute("SELECT COUNT(*) AS n FROM roots WHERE enabled=1").fetchone()
            last = self._conn.execute(
                "SELECT finished_at FROM sync_runs WHERE status='done' "
                "ORDER BY finished_at DESC LIMIT 1"
            ).fetchone()
        by_status = {row["status"]: row["n"] for row in rows}
        return {
            "files": sum(by_status.values()),
            "by_status": by_status,
            "roots": roots["n"],
            "last_sync_at": last["finished_at"] if last else None,
        }

    def record_sync_run(
        self,
        run_id: str,
        kind: str,
        *,
        status: str,
        added: int = 0,
        updated: int = 0,
        deleted: int = 0,
        failed: int = 0,
    ) -> None:
        """记录一次 ingest/delete 运行，供状态页展示。

        Args:
            run_id: 运行 UUID。
            kind: ingest/delete/scan。
            status: done/failed。
            added: 新增数。
            updated: 更新数。
            deleted: 删除数。
            failed: 失败数。
        """
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO sync_runs(id, kind, started_at, finished_at, added, updated, "
                "deleted, failed, status) VALUES(?, ?, datetime('now'), datetime('now'), ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET finished_at=datetime('now'), added=excluded.added, "
                "updated=excluded.updated, deleted=excluded.deleted, failed=excluded.failed, "
                "status=excluded.status",
                (run_id, kind, added, updated, deleted, failed, status),
            )

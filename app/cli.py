"""命令行入口：scan / index / status / search。

与 Web 共用 Settings/Manifest/LanceDB/IngestService/SearchService；
scan 只读，index 走 write_lock，search 默认 auto（按 scope 选 full/hybrid）。

Step: STEP-06 / STEP-07
Doc: docs/steps/STEP-06-入库流水线.md §6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config import Settings, load_settings
from app.db import connect_db, db_stats, health_check, open_table
from app.embeddings import EmbeddingService
from app.image_index import ImageEmbeddingService, open_asset_table
from app.ingest import IngestResult, IngestService
from app.logging_setup import setup_logging
from app.manifest import Manifest


def _build(settings: Settings):
    """初始化日志/清单/LanceDB/模型，供 CLI 命令复用。

    Args:
        settings: 运行期配置。

    Returns:
        ``(manifest, table, embedder, db)``。

    Raises:
        DBError/ModelNotReady: 数据库或模型不可用。
    """
    settings.ensure_dirs()
    setup_logging(settings)
    manifest = Manifest(settings.manifest_path)
    db = connect_db(settings)
    embedder = EmbeddingService(settings)
    table = open_table(db, settings, embedder.func)
    return manifest, table, embedder, db


def _progress(done: int, total: int, message: str) -> None:
    """CLI 进度条：单行覆盖输出。

    Args:
        done: 已处理数。
        total: 总数。
        message: 当前文件/步骤。
    """
    print(f"\r[{done}/{total}] {message[:60]:60}", end="", flush=True)
    if done == total:
        print()


def cmd_scan(args: argparse.Namespace, settings: Settings) -> int:
    """CLI：扫描索引根并输出 diff。"""
    settings.ensure_dirs()
    manifest = Manifest(settings.manifest_path)
    try:
        root_ids = None
        if args.root:
            record = manifest.add_root(args.root)
            root_ids = [record.id]
        diff = manifest.scan(settings, root_ids=root_ids)
        payload = diff.to_dict()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2 if diff.errors else 0
    finally:
        manifest.close()


def cmd_index(args: argparse.Namespace, settings: Settings) -> int:
    """CLI：扫描/指定路径并入库。"""
    manifest, table, embedder, db = _build(settings)
    asset_table = None
    image_embedder = None
    if settings.embed_provider == "gemma" and settings.image_embed_enabled:
        asset_table = open_asset_table(db, settings)
        image_embedder = ImageEmbeddingService(settings)
    service = IngestService(
        settings,
        table,
        manifest,
        embedder,
        asset_table=asset_table,
        image_embedder=image_embedder,
    )
    try:
        if args.root:
            record = manifest.add_root(args.root)
            diff = manifest.scan(settings, root_ids=[record.id])
            paths = [item.path for item in diff.added + diff.modified]
            if args.force_all:
                # 旧切片/旧清洗规则不会因为代码更新自动生效；把该根下
                # 已索引文件也加入强制重建列表。
                root_path = record.path
                for file_record in manifest.list_files():
                    if file_record.status == "deleted":
                        continue
                    if file_record.path == root_path or file_record.path.startswith(
                        root_path + "/"
                    ):
                        paths.append(file_record.path)
                paths = list(dict.fromkeys(paths))
            deleted = diff.deleted
        else:
            paths = [str(Path(p).expanduser().resolve()) for p in args.paths]
            deleted = []
        if not paths and not (
            (args.delete_missing or args.delete_ignored) and deleted
        ):
            print("没有需要入库的文件。")
            return 0
        result = (
            service.ingest_paths(
                paths,
                progress=_progress,
                force=args.force or args.force_all,
            )
            if paths
            else IngestResult()
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        delete_result = None
        if (args.delete_missing or args.delete_ignored) and deleted:
            # delete-ignored 用于清掉“文件仍在磁盘、但已被扫描规则排除”的旧索引；
            # delete-missing 只处理确实缺失的文件。
            delete_result = service.delete_paths(
                deleted,
                progress=_progress,
                only_missing=not args.delete_ignored,
            )
            print(json.dumps(delete_result, ensure_ascii=False, indent=2))
        failed = result.failed + (delete_result["failed"] if delete_result else 0)
        return 0 if failed == 0 else 2
    finally:
        manifest.close()


def cmd_roots(args: argparse.Namespace, settings: Settings) -> int:
    """CLI：列出索引根；--repair 停用嵌套子根。"""
    manifest = Manifest(settings.manifest_path, validate_roots=False)
    db = None
    try:
        repaired = manifest.repair_nested_roots() if args.repair else []
        errors: list[str] = []
        if repaired:
            db = connect_db(settings)
            try:
                chunk_table = db.open_table(settings.table_name)
                for item in repaired:
                    chunk_table.update(
                        where=f"root_id = {item['root_id']}",
                        values={"root_id": item["parent_root_id"]},
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"chunks root_id 迁移失败：{exc}")
            try:
                asset_table = db.open_table("image_assets")
                for item in repaired:
                    asset_table.update(
                        where=f"root_id = {item['root_id']}",
                        values={"root_id": item["parent_root_id"]},
                    )
            except Exception as exc:  # noqa: BLE001
                # 没有 asset 表是正常情况；其他错误必须报告。
                if "not found" not in str(exc).lower():
                    errors.append(f"assets root_id 迁移失败：{exc}")
        roots = [
            {"id": root.id, "path": root.path, "enabled": root.enabled}
            for root in manifest.list_roots()
        ]
        print(json.dumps({"roots": roots, "repaired": repaired, "errors": errors}, ensure_ascii=False, indent=2))
        return 1 if errors else 0
    finally:
        manifest.close()


def cmd_status(args: argparse.Namespace, settings: Settings) -> int:
    """CLI：输出清单与库统计。"""
    manifest, table, _, _ = _build(settings)
    try:
        payload = {
            "manifest": manifest.stats(),
            "db": db_stats(settings, table),
            "health": health_check(table),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    finally:
        manifest.close()


def cmd_search(args: argparse.Namespace, settings: Settings) -> int:
    """CLI：执行检索并打印结果。"""
    try:
        from app.reranker import RerankerService
        from app.search import SearchService
    except ImportError:
        print("检索功能尚未实现（STEP-07）。")
        return 1

    manifest, table, embedder, db = _build(settings)
    asset_table = None
    image_embedder = None
    if settings.embed_provider == "gemma" and settings.image_embed_enabled:
        asset_table = open_asset_table(db, settings)
        image_embedder = ImageEmbeddingService(settings)
    try:
        service = SearchService(
            settings,
            table,
            embedder,
            RerankerService(settings),
            manifest,
            asset_table,
            image_embedder,
        )
        result = service.search(
            args.query,
            top_k=args.top_k,
            final_k=args.final_k,
            profile=args.profile,
            mode=args.mode,
            doc_ids=args.doc_id or None,
            file_paths=args.file_path or None,
            doc_level=args.doc_level,
        )
        for index, hit in enumerate(result.results, start=1):
            print(f"{index}. {hit.score:.4f}  {hit.file_name}  {hit.file_path}")
            print(f"   {hit.snippet[:160]}")
        return 0
    finally:
        manifest.close()


def main(argv: list[str] | None = None) -> int:
    """CLI 参数解析与命令分发。"""
    parser = argparse.ArgumentParser(prog="content-search")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="扫描索引根的变更")
    scan.add_argument("--root", help="添加并扫描一个目录")
    scan.set_defaults(func=cmd_scan)

    index = sub.add_parser("index", help="同步文件到索引")
    index.add_argument("--root", help="扫描并同步该目录")
    index.add_argument("--paths", nargs="*", default=[], help="指定文件路径")
    index.add_argument("--delete-missing", action="store_true", help="删除缺失文件")
    index.add_argument(
        "--delete-ignored",
        action="store_true",
        help="删除被忽略规则排除、但仍在磁盘上的旧索引",
    )
    index.add_argument("--force", action="store_true", help="忽略 hash，强制重新解析/嵌入")
    index.add_argument(
        "--force-all",
        action="store_true",
        help="配合 --root：把该根下已索引文件和新增文件全部强制重建",
    )
    index.set_defaults(func=cmd_index)

    roots = sub.add_parser("roots", help="查看/修复索引根")
    roots.add_argument("--repair", action="store_true", help="停用嵌套子根")
    roots.set_defaults(func=cmd_roots)

    status = sub.add_parser("status", help="查看索引状态")
    status.set_defaults(func=cmd_status)

    search = sub.add_parser("search", help="检索")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=20)
    search.add_argument("--final-k", type=int, default=5)
    search.add_argument("--profile", choices=["fast", "accurate"], default=None)
    search.add_argument(
        "--mode",
        choices=["auto", "full", "hybrid", "hybrid_rerank", "fts", "vector"],
        default="auto",
        help="auto：无 scope 用 full，有 doc/path scope 用 hybrid",
    )
    search.add_argument("--doc-id", action="append", default=[], help="限定 doc_id，可重复")
    search.add_argument("--file-path", action="append", default=[], help="限定文件绝对路径，可重复")
    search.add_argument("--doc-level", action="store_true", help="每个文档最多返回一个 chunk")
    search.set_defaults(func=cmd_search)

    args = parser.parse_args(argv)
    settings = load_settings()
    return args.func(args, settings)


if __name__ == "__main__":
    sys.exit(main())

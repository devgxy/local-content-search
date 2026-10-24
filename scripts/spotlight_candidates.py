"""用 macOS Spotlight 生成评测候选并对照本地索引。

用途：
1. 对每条 query 调 ``mdfind`` 找候选文件；
2. 对照 Manifest 判断：已索引 / 未索引 / 不支持 / 索引失败；
3. 输出 JSON + Markdown，供人工确认评测标签。

Spotlight 只做候选召回，不作为相关性金标准。

Step: 评测辅助
Doc: docs/15-Spotlight参照评测.md
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from app.config import load_settings
from app.manifest import Manifest
from app.parsing.registry import SUPPORTED_EXTENSIONS, file_type_for

logger = logging.getLogger(__name__)


def parse_mdfind_output(stdout: str, root: Path) -> list[Path]:
    """把 mdfind 的换行输出解析为存在的普通文件。

    Args:
        stdout: ``mdfind`` 标准输出。
        root: 索引根，用于过滤越界路径。

    Returns:
        去重、按路径排序的 Path 列表。
    """
    root = root.expanduser().resolve()
    found: dict[str, Path] = {}
    for line in stdout.splitlines():
        raw = line.strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not resolved.is_file():
            continue
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        found[str(resolved)] = resolved
    return [found[key] for key in sorted(found)]


def run_mdfind(root: Path, query: str, timeout: int = 30) -> list[Path]:
    """调用 macOS Spotlight 查询内容候选。

    Args:
        root: 只搜索该目录。
        query: 查询文本。
        timeout: 子进程超时秒数。

    Returns:
        候选文件路径。

    Raises:
        RuntimeError: mdfind 不存在或执行失败。
    """
    try:
        result = subprocess.run(
            ["mdfind", "-onlyin", str(root.expanduser().resolve()), query],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("当前系统没有 mdfind（仅 macOS 可用）") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"mdfind 超时：{query}") from exc
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "mdfind 执行失败").strip())
    return parse_mdfind_output(result.stdout, root)


def classify(
    path: Path,
    manifest: Manifest,
) -> dict[str, Any]:
    """判断文件在本地索引中的状态。

    Args:
        path: Spotlight 候选文件。
        manifest: SQLite 清单。

    Returns:
        包含 file_type/indexed/status/reason 的字典。
    """
    supported = path.suffix.lower() in SUPPORTED_EXTENSIONS
    file_type = file_type_for(path) if supported else "unknown"
    record = manifest.get_file(str(path))
    if not supported:
        state = "unsupported"
        reason = f"类型 {path.suffix or '未知'} 当前不解析"
    elif record is None:
        state = "not_indexed"
        reason = "清单中没有该路径"
    elif record.status == "indexed":
        state = "indexed"
        reason = f"chunks={record.chunk_count}"
    else:
        state = f"index_{record.status}"
        reason = record.last_error or record.status
    return {
        "path": str(path),
        "file_type": file_type,
        "indexed": state == "indexed",
        "state": state,
        "reason": reason,
    }


def load_queries(path: Path) -> list[dict[str, Any]]:
    """读取评测 query JSONL。

    Args:
        path: 每行 ``{qid, query, relevant, type}``。

    Returns:
        query 字典列表。
    """
    queries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        queries.append(json.loads(line))
    return queries


def build_report(
    root: Path,
    queries_path: Path,
    data_dir: Path,
    limit: int,
) -> dict[str, Any]:
    """构建 Spotlight 候选对照报告。

    Args:
        root: 索引根。
        queries_path: query JSONL。
        data_dir: Manifest 所在 data 目录。
        limit: 每个 query 最多保留候选数。

    Returns:
        JSON 可序列化报告。
    """
    settings = load_settings({"DATA_DIR": str(data_dir)})
    manifest = Manifest(settings.manifest_path)
    rows = []
    try:
        for item in load_queries(queries_path):
            query = str(item["query"])
            try:
                spotlight = run_mdfind(root, query)[:limit]
                error = ""
            except RuntimeError as exc:
                spotlight = []
                error = str(exc)
            candidates = [classify(path, manifest) for path in spotlight]
            rows.append(
                {
                    "qid": item.get("qid", ""),
                    "query": query,
                    "type": item.get("type", ""),
                    "spotlight_count": len(candidates),
                    "spotlight": candidates,
                    "not_indexed": [
                        item["path"] for item in candidates if not item["indexed"]
                    ],
                    "error": error,
                }
            )
            logger.info(
                "Spotlight query=%r candidates=%d not_indexed=%d",
                query,
                len(candidates),
                sum(1 for item in candidates if not item["indexed"]),
            )
    finally:
        manifest.close()
    return {
        "root": str(root.expanduser().resolve()),
        "queries": str(queries_path),
        "data_dir": str(data_dir),
        "results": rows,
    }


def write_markdown(report: dict[str, Any], out: Path) -> None:
    """写便于人工复核的 Markdown。"""
    lines = [
        "# Spotlight 候选对照",
        "",
        f"- 根目录：`{report['root']}`",
        f"- 查询集：`{report['queries']}`",
        "",
        "> Spotlight 仅用于候选召回/查漏，不是相关性金标准。",
        "",
        "| qid | query | Spotlight | 未索引/不支持 |",
        "|---|---|---|---|",
    ]
    for item in report["results"]:
        missing = "<br>".join(
            f"`{Path(path).name}`" for path in item["not_indexed"][:6]
        )
        lines.append(
            f"| {item['qid']} | {item['query']} | {item['spotlight_count']} | "
            f"{missing or '无'} |"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Spotlight 搜索根")
    parser.add_argument("--queries", required=True, help="评测 query JSONL")
    parser.add_argument("--data-dir", default="data", help="Manifest 所在 data 目录")
    parser.add_argument("--out", default="reports/spotlight-candidates.json")
    parser.add_argument("--markdown", default="reports/spotlight-candidates.md")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    report = build_report(
        Path(args.root),
        Path(args.queries),
        Path(args.data_dir),
        args.limit,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, Path(args.markdown))
    print(f"报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""启动前自检：模型、依赖、目录、MPS、OCR。

流程：Python/Pydantic/LanceDB 版本 → 模型目录与大小 → data 可写 →
MPS 可用 → OCR 可用 → sentencepiece（Gemma 分词）。任一 error 阻断启动。

Step: STEP-00
Doc: docs/steps/STEP-00-项目骨架与环境校验.md §4.4
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.config import Settings

MIN_EMBED_BYTES = 1_000_000_000
MIN_RERANK_BYTES = 1_500_000_000
EXPECTED_LANCEDB = "0.40.0"


@dataclass
class CheckItem:
    """单项自检结果。

    Attributes:
        name: 检查项名称。
        ok: 是否通过。
        level: error 阻断启动；warn 只提示。
        message: 当前版本/状态。
        hint: 修复建议。
    """

    name: str
    ok: bool
    level: Literal["error", "warn"] = "error"
    message: str = ""
    hint: str = ""


@dataclass
class PreflightReport:
    """自检报告。

    Attributes:
        items: 所有检查项。
    """

    items: list[CheckItem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """是否全部通过。"""
        return all(item.ok for item in self.items if item.level == "error")

    @property
    def errors(self) -> list[CheckItem]:
        """错误项列表。"""
        return [item for item in self.items if not item.ok and item.level == "error"]

    @property
    def warnings(self) -> list[CheckItem]:
        """警告项列表。"""
        return [item for item in self.items if not item.ok and item.level == "warn"]


def _version(package: str) -> str | None:
    """读取已安装包版本；未安装返回 None。"""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _check_python() -> CheckItem:
    """检查 Python >= 3.11。"""
    version = platform.python_version()
    ok = sys.version_info >= (3, 11)
    return CheckItem(
        "python",
        ok,
        message=version,
        hint="" if ok else "需要 Python 3.11 或 3.12",
    )


def _check_pydantic() -> CheckItem:
    """检查 pydantic v2（LanceDB 要求）。"""
    version = _version("pydantic")
    if version is None:
        return CheckItem("pydantic", False, message="未安装", hint="pip install 'pydantic>=2.7'")
    major = int(version.split(".", 1)[0])
    return CheckItem(
        "pydantic",
        major == 2,
        message=version,
        hint="" if major == 2 else "LanceDB 需要 pydantic v2；请使用项目 .venv",
    )


def _check_lancedb() -> CheckItem:
    """检查 lancedb 版本是否为 0.40.x。"""
    version = _version("lancedb")
    if version is None:
        return CheckItem("lancedb", False, message="未安装", hint="pip install 'lancedb==0.40.0'")
    ok = version == EXPECTED_LANCEDB
    return CheckItem(
        "lancedb",
        ok,
        level="error" if ok else "warn",
        message=version,
        hint="" if ok else f"建议锁定 {EXPECTED_LANCEDB}，API 变动较快",
    )


def _check_model(
    name: str,
    path: Path,
    weights_name: str,
    min_bytes: int,
) -> CheckItem:
    """检查本地模型目录/权重大小。

    Args:
        name: 显示名。
        path: 模型目录。
        min_bytes: 最小权重字节数。

    Returns:
        CheckItem；缺失/过小为 error。
    """
    """检查本地模型目录/权重大小。"""
    if not path.is_dir():
        return CheckItem(name, False, message="目录不存在", hint=f"检查 {path}")
    candidates = [path / weights_name, path / "model.safetensors"]
    weights = next((item for item in candidates if item.exists()), None)
    if weights is None:
        return CheckItem(name, False, message="权重缺失", hint=f"缺少 {weights_name}")
    size = weights.stat().st_size
    if size < min_bytes:
        return CheckItem(
            name,
            False,
            message=f"权重过小（{size / 1e9:.2f}GB）",
            hint="下载可能被截断，重新运行模型下载脚本",
        )
    return CheckItem(name, True, message=f"{path.name}（{size / 1e9:.2f}GB）")


def _check_ready_marker(settings: Settings) -> CheckItem:
    """检查 .models/READY.json 是否存在。"""
    """检查 .models/READY.json 是否存在。"""
    marker = settings.models_dir / "READY.json"
    if marker.exists():
        return CheckItem("ready_marker", True, message=str(marker))
    return CheckItem(
        "ready_marker",
        False,
        message=".models/READY.json 缺失",
        hint="运行 scripts/download_models.sh 或手动创建标记",
    )


def _check_data_dir(settings: Settings) -> CheckItem:
    """检查 data 目录可创建/可写。"""
    """检查 data 目录可创建/可写。"""
    try:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=settings.data_dir, delete=True):
            pass
        return CheckItem("data_dir", True, message=f"{settings.data_dir} 可写")
    except OSError as exc:
        return CheckItem("data_dir", False, message=str(exc), hint="检查目录权限")


def _check_mps() -> CheckItem:
    """检查 PyTorch MPS 是否可用（不可用允许 CPU）。"""
    """检查 PyTorch MPS 是否可用（不可用允许 CPU）。"""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return CheckItem("mps", False, level="warn", message=f"torch 不可用: {exc}")
    available = bool(torch.backends.mps.is_available())
    return CheckItem(
        "mps",
        available,
        level="warn",
        message="available" if available else "不可用，将回退 CPU",
    )


def _check_ocr() -> CheckItem:
    """检查 macOS Vision OCR 依赖。"""
    """检查 macOS Vision OCR 依赖。"""
    if importlib.util.find_spec("ocrmac") is None:
        return CheckItem(
            "ocr",
            False,
            level="warn",
            message="ocrmac 未安装",
            hint="图片/扫描 PDF 将不可用；pip install ocrmac",
        )
    return CheckItem("ocr", True, message="macOS Vision 可用")


def _check_sentencepiece() -> CheckItem:
    """检查 sentencepiece（Gemma 分词需要）。"""
    """检查 sentencepiece（Gemma 分词需要）。"""
    if importlib.util.find_spec("sentencepiece") is None:
        return CheckItem(
            "sentencepiece",
            False,
            message="未安装",
            hint="XLM-R 重排模型需要 sentencepiece",
        )
    return CheckItem("sentencepiece", True, message="ok")


def run_preflight(
    settings: Settings,
    *,
    check_mps: bool = True,
    check_ocr: bool = True,
) -> PreflightReport:
    """执行自检；不抛异常，返回结构化报告。"""
    items = [
        _check_python(),
        _check_pydantic(),
        _check_lancedb(),
        _check_ready_marker(settings),
        _check_model(
            "embed_model",
            settings.embed_model_path,
            "pytorch_model.bin",
            MIN_EMBED_BYTES,
        ),
        _check_model(
            "rerank_model",
            settings.rerank_model_path,
            "pytorch_model.bin",
            MIN_RERANK_BYTES,
        ),
        _check_sentencepiece(),
        _check_data_dir(settings),
    ]
    if settings.table_structure_enabled:
        items.append(
            _check_model(
                "table_detection_model",
                settings.table_detection_model_path,
                "model.safetensors",
                50_000_000,
            )
        )
        items.append(
            _check_model(
                "table_structure_model",
                settings.table_structure_model_path,
                "model.safetensors",
                50_000_000,
            )
        )
    if check_mps:
        items.append(_check_mps())
    if check_ocr:
        items.append(_check_ocr())
    return PreflightReport(items=items)


def print_report(report: PreflightReport) -> None:
    """把自检结果打印到 stdout。

    Args:
        report: 自检报告。
    """
    print(f"{'STATUS':8} {'ITEM':18} DETAIL")
    print("-" * 78)
    for item in report.items:
        status = "OK" if item.ok else ("WARN" if item.level == "warn" else "FAIL")
        print(f"{status:8} {item.name:18} {item.message}")
        if not item.ok and item.hint:
            print(f"{'':8} {'':18} -> {item.hint}")
    print("-" * 78)
    if report.ok:
        print("自检通过。")
    else:
        print(f"自检失败：{len(report.errors)} 个错误，{len(report.warnings)} 个警告。")

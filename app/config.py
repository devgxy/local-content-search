"""应用配置加载与校验。

所有路径在加载时解析为绝对路径，避免运行目录变化导致模型/数据找不到。
环境变量是唯一配置来源；`.env` 不会自动加载。

Step: STEP-00
Doc: docs/steps/STEP-00-项目骨架与环境校验.md §4.2
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_str(env: Mapping[str, str], key: str, default: str) -> str:
    """读取字符串环境变量；空字符串回退默认值。"""
    value = env.get(key)
    return value.strip() if value and value.strip() else default


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    """读取整数环境变量，非法值抛 ValueError。"""
    raw = env.get(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"环境变量 {key} 必须是整数，当前为 {raw!r}") from exc


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    """读取浮点环境变量，非法值抛 ValueError。"""
    raw = env.get(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"环境变量 {key} 必须是数字，当前为 {raw!r}") from exc


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    """读取布尔环境变量（true/false/1/0/yes/no/on/off）。"""
    raw = env.get(key)
    if raw is None or not str(raw).strip():
        return default
    value = str(raw).strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(f"环境变量 {key} 必须是布尔值（true/false），当前为 {raw!r}")


def _env_path(env: Mapping[str, str], key: str, default: Path) -> Path:
    """读取路径环境变量并展开 ~、解析为绝对路径。"""
    raw = env.get(key)
    if raw is None or not str(raw).strip():
        return default.expanduser().resolve()
    return Path(str(raw).strip()).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    """运行期配置（frozen dataclass）。

    所有字段都可由同名大写环境变量覆盖；路径字段加载时解析为绝对路径。
    字段单位见行内注释，例如 chunk_size 为字符数、max_file_mb 为 MB。
    """

    # Web
    host: str = "127.0.0.1"   # 仅本地
    port: int = 8765          # HTTP 端口

    # 路径
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / ".models"
    web_dir: Path = PROJECT_ROOT / "web"

    # 模型
    embed_model: str = "bge-large-zh-v1.5"
    embed_provider: str = "bge"          # bge | gemma
    embed_dtype: str = "float32"         # float32 | bfloat16
    embed_dim: int = 1024
    embed_query_prompt: str = ""         # 空则用 provider 默认
    embed_doc_template: str = ""         # 空则用 provider 默认
    image_embed_enabled: bool = True      # 仅 Gemma provider 生效
    rerank_model: str = "bge-reranker-large"
    table_name: str = "chunks"
    device: str = "mps"

    # 切片（单位：字符）
    chunk_size: int = 400       # 目标块长
    chunk_overlap: int = 50     # 相邻块重叠
    chunk_min: int = 80         # 低于此值尝试合并
    chunk_max: int = 600        # 硬上限
    chunk_adaptive: bool = True # 段落/标题优先，段落 <= chunk_max 不拆
    max_chunks_per_doc: int = 5000  # 单文档截断上限

    # 检索
    search_profile: str = "accurate"   # fast | accurate
    top_k: int = 20                    # 初筛数
    final_k: int = 5                   # 最终返回数
    per_doc_limit: int = 2             # full 模式每文档上限
    context_chars: int = 1200          # 结果上下文最大字符
    embed_batch_size: int = 32         # 编码 batch
    rerank_batch_size: int = 32        # 重排 batch
    rerank_max_length: int = 512       # CrossEncoder token 上限
    rerank_parent_context: bool = False # 重排时用同 section 父上下文（实验）

    # 解析/OCR
    max_file_mb: int = 200
    max_pdf_pages: int = 500
    ocr_enabled: bool = True
    ocr_min_image_px: int = 200
    pdf_ocr_text_threshold: int = 40
    pdf_strip_header_footer: bool = True  # 去重复页眉/页脚/页码
    pdf_table_extract: bool = True  # 用 PyMuPDF find_tables 提取带线表格
    special_context: bool = False  # 表格/代码前拼短上下文（默认关闭，A/B 显示会伤 chunk）
    table_structure_enabled: bool = False  # 扫描表格结构模型（默认关闭，避免启动加载）
    table_structure_device: str = "cpu"
    table_cell_ocr: bool = True
    table_confidence: float = 0.5
    table_tile_detection: bool = True
    table_detection_model_path: Path = PROJECT_ROOT / ".models" / "table-transformer-detection"
    table_structure_model_path: Path = PROJECT_ROOT / ".models" / "table-transformer-structure"
    ocr_min_confidence: float = 0.25
    ocr_deskew: bool = False
    ocr_max_skew: float = 15.0

    # 其它
    log_level: str = "INFO"
    auto_scan: bool = False

    @property
    def db_dir(self) -> Path:
        """LanceDB 数据目录。"""
        return self.data_dir / "lancedb"

    @property
    def manifest_path(self) -> Path:
        """SQLite 清单路径。"""
        return self.data_dir / "manifest.sqlite"

    @property
    def log_dir(self) -> Path:
        """日志目录。"""
        return self.data_dir / "logs"

    @property
    def schema_meta_path(self) -> Path:
        """schema 元数据文件路径。"""
        return self.data_dir / "schema.json"

    @property
    def eval_dir(self) -> Path:
        """评测输出目录。"""
        return self.data_dir / "eval"

    @property
    def embed_model_path(self) -> Path:
        """embedding 模型绝对路径。"""
        return self.models_dir / self.embed_model

    @property
    def rerank_model_path(self) -> Path:
        """reranker 模型绝对路径。"""
        return self.models_dir / self.rerank_model

    def ensure_dirs(self) -> None:
        """创建运行期目录（不创建模型目录，避免掩盖模型缺失）。"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


def _validate(settings: Settings) -> Settings:
    """校验跨字段约束并返回原 Settings。

    Args:
        settings: 待校验配置。

    Returns:
        通过校验的 settings。

    Raises:
        ValueError: 端口/维度/切片/批量等约束不满足。
    """
    if not (1 <= settings.port <= 65535):
        raise ValueError(f"APP_PORT 必须在 1..65535，当前为 {settings.port}")
    if settings.embed_dim <= 0:
        raise ValueError("embed_dim 必须大于 0")
    if settings.embed_provider not in {"bge", "gemma"}:
        raise ValueError("EMBED_PROVIDER 只能是 bge 或 gemma")
    if settings.embed_provider == "gemma" and settings.embed_dim not in {128, 256, 512, 768}:
        raise ValueError("Gemma MRL 维度只能是 128/256/512/768")
    if settings.embed_provider == "bge" and settings.embed_dim != 1024:
        raise ValueError("BGE 维度固定为 1024")
    if settings.embed_dtype not in {"float32", "bfloat16"}:
        raise ValueError("EMBED_DTYPE 只能是 float32 或 bfloat16（禁止 fp16）")
    if settings.embed_provider == "gemma" and settings.embed_dtype == "float32":
        # 允许 fp32 以便在 MPS 不支持 bf16 时回退；仅禁止 fp16。
        pass
    if settings.chunk_overlap < 0 or settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("CHUNK_OVERLAP 必须大于等于 0 且小于 CHUNK_SIZE")
    if not (settings.chunk_min <= settings.chunk_size <= settings.chunk_max):
        raise ValueError("必须满足 CHUNK_MIN <= CHUNK_SIZE <= CHUNK_MAX")
    if settings.search_profile not in {"fast", "accurate"}:
        raise ValueError("SEARCH_PROFILE 只能是 fast 或 accurate")
    if settings.top_k < 1:
        raise ValueError("TOP_K 必须大于 0")
    if not (1 <= settings.final_k <= settings.top_k):
        raise ValueError("必须满足 1 <= FINAL_K <= TOP_K")
    if settings.per_doc_limit < 1:
        raise ValueError("PER_DOC_LIMIT 必须大于 0")
    if settings.embed_batch_size < 1 or settings.rerank_batch_size < 1:
        raise ValueError("批量大小必须大于 0")
    if settings.max_file_mb < 1:
        raise ValueError("MAX_FILE_MB 必须大于 0")
    return settings


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """从环境变量加载配置并校验。

    Args:
        env: 可选环境变量映射；默认 os.environ。

    Returns:
        校验后的 Settings。

    Raises:
        ValueError: 任一环境变量格式/取值非法。
    """
    env = os.environ if env is None else env
    settings = Settings(
        host=_env_str(env, "APP_HOST", "127.0.0.1"),
        port=_env_int(env, "APP_PORT", 8765),
        data_dir=_env_path(env, "DATA_DIR", PROJECT_ROOT / "data"),
        models_dir=_env_path(env, "MODELS_DIR", PROJECT_ROOT / ".models"),
        web_dir=_env_path(env, "WEB_DIR", PROJECT_ROOT / "web"),
        embed_model=_env_str(env, "EMBED_MODEL", "bge-large-zh-v1.5"),
        embed_provider=_env_str(env, "EMBED_PROVIDER", "bge").lower(),
        embed_dtype=_env_str(env, "EMBED_DTYPE", "float32").lower(),
        embed_dim=_env_int(env, "EMBED_DIM", 1024),
        embed_query_prompt=_env_str(env, "EMBED_QUERY_PROMPT", ""),
        embed_doc_template=_env_str(env, "EMBED_DOC_TEMPLATE", ""),
        image_embed_enabled=_env_bool(env, "IMAGE_EMBED_ENABLED", True),
        rerank_model=_env_str(env, "RERANK_MODEL", "bge-reranker-large"),
        table_name=_env_str(env, "TABLE_NAME", "chunks"),
        device=_env_str(env, "DEVICE", "mps"),
        chunk_size=_env_int(env, "CHUNK_SIZE", 400),
        chunk_overlap=_env_int(env, "CHUNK_OVERLAP", 50),
        chunk_min=_env_int(env, "CHUNK_MIN", 80),
        chunk_max=_env_int(env, "CHUNK_MAX", 600),
        chunk_adaptive=_env_bool(env, "CHUNK_ADAPTIVE", True),
        max_chunks_per_doc=_env_int(env, "MAX_CHUNKS_PER_DOC", 5000),
        context_chars=_env_int(env, "CONTEXT_CHARS", 1200),
        search_profile=_env_str(env, "SEARCH_PROFILE", "accurate").lower(),
        top_k=_env_int(env, "TOP_K", 20),
        final_k=_env_int(env, "FINAL_K", 5),
        per_doc_limit=_env_int(env, "PER_DOC_LIMIT", 2),
        embed_batch_size=_env_int(env, "EMBED_BATCH_SIZE", 32),
        rerank_batch_size=_env_int(env, "RERANK_BATCH_SIZE", 32),
        rerank_parent_context=_env_bool(env, "RERANK_PARENT_CONTEXT", False),
        rerank_max_length=_env_int(env, "RERANK_MAX_LENGTH", 512),
        max_file_mb=_env_int(env, "MAX_FILE_MB", 200),
        max_pdf_pages=_env_int(env, "MAX_PDF_PAGES", 500),
        ocr_enabled=_env_bool(env, "OCR_ENABLED", True),
        ocr_min_image_px=_env_int(env, "OCR_MIN_IMAGE_PX", 200),
        pdf_ocr_text_threshold=_env_int(env, "PDF_OCR_TEXT_THRESHOLD", 40),
        pdf_strip_header_footer=_env_bool(env, "PDF_STRIP_HEADER_FOOTER", True),
        pdf_table_extract=_env_bool(env, "PDF_TABLE_EXTRACT", True),
        special_context=_env_bool(env, "CHUNK_SPECIAL_CONTEXT", False),
        table_structure_enabled=_env_bool(env, "TABLE_STRUCTURE_ENABLED", False),
        table_structure_device=_env_str(env, "TABLE_STRUCTURE_DEVICE", "cpu").lower(),
        table_cell_ocr=_env_bool(env, "TABLE_CELL_OCR", True),
        table_confidence=_env_float(env, "TABLE_CONFIDENCE", 0.5),
        table_tile_detection=_env_bool(env, "TABLE_TILE_DETECTION", True),
        table_detection_model_path=_env_path(
            env, "TABLE_DETECTION_MODEL_PATH",
            PROJECT_ROOT / ".models" / "table-transformer-detection",
        ),
        table_structure_model_path=_env_path(
            env, "TABLE_STRUCTURE_MODEL_PATH",
            PROJECT_ROOT / ".models" / "table-transformer-structure",
        ),
        ocr_min_confidence=_env_float(env, "OCR_MIN_CONFIDENCE", 0.25),
        ocr_deskew=_env_bool(env, "OCR_DESKEW", False),
        ocr_max_skew=_env_float(env, "OCR_MAX_SKEW", 15.0),
        log_level=_env_str(env, "LOG_LEVEL", "INFO").upper(),
        auto_scan=_env_bool(env, "AUTO_SCAN", False),
    )
    return _validate(settings)

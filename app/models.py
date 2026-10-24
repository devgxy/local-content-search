"""API 数据传输对象。

Step: STEP-08
Doc: docs/steps/STEP-08-API层.md §5
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

FileType = Literal["md", "txt", "pdf", "docx", "pptx", "html", "json", "csv", "image"]


class SearchRequest(BaseModel):
    """搜索 API 请求。"""
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500)       # 查询文本
    top_k: int = Field(default=20, ge=5, le=50)           # 初筛数
    final_k: int = Field(default=5, ge=1, le=20)          # 最终返回数
    file_types: list[FileType] = Field(default_factory=list)  # 类型过滤
    root_id: int | None = None                            # 索引根过滤
    doc_ids: list[str] = Field(default_factory=list, max_length=1000)  # 动态文件白名单
    file_paths: list[str] = Field(default_factory=list, max_length=1000)  # 动态路径白名单
    mode: Literal["auto", "full", "hybrid", "hybrid_rerank", "fts", "vector"] | None = None  # 检索方式
    profile: Literal["fast", "accurate"] | None = None    # 速度档位
    doc_level: bool = False                                # 每个文档最多返回一个 chunk

    @field_validator("final_k")
    @classmethod
    def _final_le_top(cls, value: int, info):  # noqa: ANN001
        """校验 final_k 不大于 top_k。

        Args:
            value: final_k。
            info: pydantic 校验上下文。

        Returns:
            通过校验的 final_k。
        """
        top_k = info.data.get("top_k", 20)
        if value > top_k:
            raise ValueError("final_k 不能大于 top_k")
        return value


class SearchHitDTO(BaseModel):
    """搜索 API 单条结果。"""
    chunk_id: str                 # chunk 主键
    doc_id: str                   # 文档主键
    chunk_index: int              # 文档内 chunk 序号；图片为 -1
    score: float                  # 当前模式主分
    rrf_score: float              # RRF 原始分（仅 hybrid）
    rerank_score: float | None    # 重排 0..1 分
    rank_before: int              # 重排前名次
    rank_after: int               # 最终名次
    file_name: str                # 文件名
    file_path: str                # 绝对路径
    file_type: str                # md/txt/pdf/...
    title: str                    # 文档标题
    heading: str                  # 命中标题路径
    snippet: str                  # section 级上下文
    summary: str = ""             # 抽取式摘要
    page: int                     # PDF 页码
    slide: int                    # PPTX 页码
    source_kind: str              # text/ocr/image/table
    score_kind: str = "rrf"       # rerank/rrf/bm25/distance/image
    table_html: str = ""          # 表格结构 HTML（仅 table）
    table_confidence: float | None = None  # 表格结构置信度


class SearchResponse(BaseModel):
    """搜索 API 响应。"""
    query: str
    took_ms: int
    candidates: int
    results: list[SearchHitDTO]
    warnings: list[str] = Field(default_factory=list)
    timings: dict = Field(default_factory=dict)
    highlight_terms: list[str] = Field(default_factory=list)


class LibraryItem(BaseModel):
    """资料库单文件 DTO。"""
    doc_id: str                    # 文档主键
    file_name: str                 # 文件名
    file_path: str                 # 绝对路径
    file_type: str                 # 文件类型
    size: int                      # 字节数
    mtime: float                   # 修改时间戳
    chunks: int                    # chunk 数
    indexed_at: str | None         # 最近索引时间
    status: str                    # pending/indexing/indexed/error/deleted
    last_error: str | None = None  # 最近错误


class LibraryResponse(BaseModel):
    """资料库分页响应。"""
    total: int
    page: int
    page_size: int
    items: list[LibraryItem]


class ChunkDTO(BaseModel):
    """文档详情 chunk DTO。"""
    chunk_id: str
    chunk_index: int
    text: str
    display_text: str
    heading: str
    page: int
    slide: int
    source_kind: str


class DocumentDetail(BaseModel):
    """文档详情响应。"""
    item: LibraryItem
    chunks: list[ChunkDTO]


class RootCreate(BaseModel):
    """添加索引根请求。"""
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=4096)


class RootDTO(BaseModel):
    """索引根 DTO。"""
    id: int
    path: str
    enabled: bool
    created_at: str
    doc_count: int


class ScanRequest(BaseModel):
    """扫描请求。"""
    model_config = ConfigDict(extra="forbid")
    root_ids: list[int] | None = None


class SyncRequest(BaseModel):
    """同步/删除请求。"""
    model_config = ConfigDict(extra="forbid")

    paths: list[str] = Field(default_factory=list, max_length=1000)
    deleted_paths: list[str] = Field(default_factory=list, max_length=1000)
    delete_missing: bool = False

    @field_validator("paths", "deleted_paths")
    @classmethod
    def _path_length(cls, values: list[str]) -> list[str]:
        """校验单个路径长度。

        Args:
            values: 路径列表。

        Returns:
            原列表；超长时抛 ValueError。
        """
        for value in values:
            if len(value) > 4096:
                raise ValueError("单个路径不能超过 4096 字符")
        return values


class ReindexRequest(BaseModel):
    """重建请求。"""
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=4096)


class OpenRequest(BaseModel):
    """打开文件请求。"""
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=4096)
    reveal: bool = False


class JobDTO(BaseModel):
    """后台任务 DTO。"""
    id: str
    kind: str
    status: str
    progress: float
    message: str
    processed: int
    total: int
    result: dict | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """健康检查响应。"""
    status: str
    version: str
    models: dict
    db: dict
    offline: bool


class StatsResponse(BaseModel):
    """统计响应。"""
    manifest: dict
    db: dict

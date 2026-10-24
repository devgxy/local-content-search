"""LanceDB 表结构。

Step: STEP-01
Doc: docs/00-总体方案.md §5.1 · docs/steps/STEP-01-数据层与索引.md §4.1
"""

from __future__ import annotations

from lancedb.pydantic import LanceModel, Vector

EMBED_DIM = 1024


def build_chunk_model(func, dim: int = EMBED_DIM):
    """用 embedding function 绑定 text → vector，返回 LanceModel 子类。

    注意：不要写 ``Vector(dim, source="text")``，LanceDB 0.40 没有 source 参数。
    """

    class Chunk(LanceModel):
        """LanceDB chunk 行模型（vector + FTS text）。

        Attributes:
            id: 稳定 chunk ID。
            doc_id: 文档 ID。
            root_id: 索引根。
            chunk_index: 文档内顺序。
            text: 参与 FTS/向量的文本（标题 + 正文）。
            vector: 固定维度向量。
            display_text: 展示正文（不含标题前缀）。
            file_name/file_path/file_type: 文件元数据。
            title: 文档标题。
            heading: 标题路径。
            page/slide: PDF/PPTX 页码。
            char_start/char_end: 在解析全文中的字符偏移。
            content_hash/doc_hash: 块/文件内容哈希。
            run_id: 入库运行 ID。
            indexed_at: 写入时间戳。
            source_kind: text/ocr 等。
            meta_json: 解析警告等扩展信息。
        """

        id: str
        doc_id: str
        root_id: int
        chunk_index: int
        text: str = func.SourceField()
        vector: Vector(dim) = func.VectorField()
        display_text: str
        file_name: str
        file_path: str
        file_type: str

        title: str = ""
        heading: str = ""
        page: int = 0
        slide: int = 0
        char_start: int = 0
        char_end: int = 0
        content_hash: str = ""
        doc_hash: str = ""
        run_id: str = ""
        indexed_at: float = 0.0
        source_kind: str = "text"
        meta_json: str = ""

    return Chunk

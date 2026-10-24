"""解析器注册表。

Step: STEP-03
Doc: docs/steps/STEP-03-解析层.md §4.5
"""

from __future__ import annotations

from pathlib import Path

from app.parsing.base import ParseError, Parser
from app.parsing.docx import DocxParser
from app.parsing.html import HtmlParser
from app.parsing.image_ocr import ImageParser
from app.parsing.pdf import PdfParser
from app.parsing.pptx import PptxParser
from app.parsing.structured import StructuredTextParser
from app.parsing.text_md import TextMarkdownParser

_PARSERS: list[Parser] = [
    TextMarkdownParser(),
    PdfParser(),
    DocxParser(),
    PptxParser(),
    StructuredTextParser(),
    HtmlParser(),
    ImageParser(),
]

_PARSER_BY_EXT: dict[str, Parser] = {}
_FILE_TYPE_BY_EXT: dict[str, str] = {}
for _parser in _PARSERS:
    _type = {
        TextMarkdownParser: "md",
        PdfParser: "pdf",
        DocxParser: "docx",
        PptxParser: "pptx",
        StructuredTextParser: "structured",
        HtmlParser: "html",
        ImageParser: "image",
    }[type(_parser)]
    for _ext in _parser.extensions:
        _PARSER_BY_EXT[_ext.lower()] = _parser
        if _type == "md" and _ext == ".txt":
            _FILE_TYPE_BY_EXT[_ext.lower()] = "txt"
        elif _type == "structured":
            # 类型过滤/资料库/API 使用真实类型，而不是统一 structured。
            _FILE_TYPE_BY_EXT[_ext.lower()] = {
                ".json": "json",
                ".csv": "csv",
                ".log": "txt",
            }[_ext.lower()]
        else:
            _FILE_TYPE_BY_EXT[_ext.lower()] = _type

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_PARSER_BY_EXT)


def is_supported(path: Path | str) -> bool:
    """判断扩展名是否受支持。

    Args:
        path: 文件路径。

    Returns:
        True 表示可解析。
    """
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def file_type_for(path: Path | str) -> str:
    """扩展名转文件类型。

    Args:
        path: 文件路径。

    Returns:
        md/txt/pdf/docx/pptx/html/image；未知返回 unknown。
    """
    suffix = Path(path).suffix.lower()
    if suffix not in _FILE_TYPE_BY_EXT:
        raise ParseError("UNSUPPORTED_TYPE", f"不支持的文件类型：{suffix}")
    return _FILE_TYPE_BY_EXT[suffix]


def get_parser(path: Path | str) -> Parser:
    """返回对应 Parser 实例。

    Args:
        path: 文件路径。

    Returns:
        Parser 实例。

    Raises:
        ParseError: 扩展名不支持。
    """
    suffix = Path(path).suffix.lower()
    parser = _PARSER_BY_EXT.get(suffix)
    if parser is None:
        raise ParseError(
            "UNSUPPORTED_TYPE",
            f"不支持的文件类型：{suffix or '<无扩展名>'}",
            f"支持：{', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )
    return parser

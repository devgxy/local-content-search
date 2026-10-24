# Local Content Search

macOS 本地内容混合检索系统：文件解析、OCR、结构感知切片、LanceDB 向量 + 中文 FTS、混合检索、重排与本地 Web UI。

## 特性

- 本地离线运行，文档和向量不上传云端；
- 支持 Markdown、TXT、PDF、DOCX、PPTX、HTML、JSON、CSV、LOG、图片 OCR；
- 向量检索 + ICU 中文全文检索 + RRF 融合；
- 可选 CrossEncoder 重排；
- 文件/目录 scope，`auto` 自动选择 `full` 或 `hybrid`；
- 可选本地 Table Transformer 表格检测/结构/单元格 OCR；
- Web UI：搜索、资料库、同步、设置；
- 176 个测试（工作区含生成本地 fixture；公开快照未提交 fixture 时为 174 passed / 2 skipped）。

## 环境

- macOS（Apple Silicon 推荐）
- Python 3.11
- 本地模型约 3.3GB

## 快速开始

```bash
git clone https://github.com/devgxy/local-content-search.git
cd local-content-search

python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# 按 .env 配置 DATA_DIR、模型路径等
```

模型准备：

```bash
scripts/download_models.sh
# 可选：表格模型
python scripts/download_table_models.py
```

启动：

```bash
python -m app
```

访问：

```text
http://127.0.0.1:8765
```

## 关键配置

| 变量 | 默认 | 说明 |
|---|---|---|
| `DATA_DIR` | `./data` | 索引、清单、日志目录 |
| `TABLE_STRUCTURE_ENABLED` | false | 是否加载本地表格模型 |
| `TABLE_STRUCTURE_DEVICE` | cpu | `cpu` / `mps` |
| `OCR_DESKEW` | false | 扫描页倾斜校正，实验性 |
| `RERANK_ENABLED` | true | 是否加载重排模型 |

完整配置见 `.env.example` 和 `docs/04-离线部署与运维.md`。

## 目录

```text
app/        # FastAPI、检索、解析、索引
web/        # 本地 Web UI
tests/      # 单元/集成测试
docs/       # 设计与实现文档
scripts/    # 安装、模型、评测脚本
examples/   # 最小评测样例
reports/    # 公开验收摘要（不含个人数据）
```

## 文档

- [总体方案](docs/00-总体方案.md)
- [代码结构与追溯矩阵](docs/02-代码结构与追溯矩阵.md)
- [评测方案](docs/03-评测方案.md)
- [工程实现与代码映射](docs/06-工程实现与代码映射.md)
- [v1.2 验收摘要](reports/v1.2-acceptance.md)
- [RAG 工程手册（文章目录）](docs/article/README.md)

## 隐私与数据

本仓库不包含：

- 用户文档、简历、图片、索引数据；
- 模型权重；
- 个人路径、姓名、电话、邮箱；
- 原始评测语料。

`data/`、`.models/`、`tests/fixtures/` 等通过 `.gitignore` 排除；`reports/evidence/` 仅保留脱敏聚合证据。

## 已知边界

- 复杂无线/合并表格仍受模型能力限制；
- OCR 版面在复杂扫描件上需要人工抽检；
- 表格结构指标不等于检索指标提升；
- 小样本评测数字只用于工程回归，不作为统计结论。

## License

未声明许可证；默认保留所有权利。使用第三方模型/数据前请核对各自许可证。

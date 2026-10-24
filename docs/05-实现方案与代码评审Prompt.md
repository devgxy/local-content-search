# 实现方案与代码评审 Prompt

> 文档编号：DOC-05  
> 版本：1.2.0.dev0（稳定版 v1.1）  
> 日期：2026-10-24  
> 用途：交给 codex 做**只读**代码审查；不要修改代码，只输出问题清单。  
> 审查对象：`/Users/example/Downloads/关键词检索`

---

## 1. 你的角色与任务

你是一名资深 Python / 检索系统 / macOS 本地应用架构师。  
请对当前工程做一次**只读**审查：

- 只读代码、配置、测试、文档；
- **不要修改任何文件**；
- 不要重装依赖、不要删除 `data/` 或 `.models/`；
- 可以运行只读检查（如 `pytest`、`ruff`、`python -m app --check`），但不要写仓库文件；
- 输出按严重级别排序的问题清单，必须带 `文件:行号`、原因、影响、建议修复方式；
- 不要泛泛而谈“可以优化”，要指出具体代码路径。

---

## 2. 系统目标与约束

本地 macOS 内容检索系统，单人使用，完全离线：

- 十类内容：MD / TXT / PDF / DOCX / PPTX / HTML / JSON / CSV / LOG / 图片 OCR；
- LanceDB 单库：1024 维 BGE 向量 + ICU 中文 FTS；
- 查询：hybrid 双路召回 + RRF → `bge-reranker-large` 精排；
- 文件选择：索引根 + 扫描 diff + 用户勾选后手动同步；
- 增量：先插新 chunk，再按 `run_id` 清理旧 chunk；
- 交付：FastAPI + 原生 HTML/CSS/JS 本地 Web UI；
- 运行期不请求任何云端 API。

关键不变量：

1. `vector` 必须绑定 `text`，维度 1024；
2. FTS 必须使用 `base_tokenizer="icu"`；
3. 不允许使用 Tantivy / `use_tantivy=True`；
4. 模型路径必须是绝对路径，缺失时快速失败；
5. 单文件失败不能中断整批；
6. 扫描阶段只做 `stat`，不自动写 LanceDB；
7. 删除缺失文件必须由用户显式勾选。

---

## 3. 代码地图

| 模块 | 文件 | 职责 |
|---|---|---|
| 配置/自检 | `app/config.py`、`app/preflight.py`、`app/logging_setup.py` | 路径、环境变量、模型/依赖检查 |
| ID/哈希 | `app/ids.py` | doc_id / chunk_id / content_hash |
| 数据层 | `app/schema.py`、`app/db.py` | LanceModel、建表、ICU FTS、健康检查 |
| 模型 | `app/embeddings.py`、`app/reranker.py` | BGE 编码、CrossEncoder 重排 |
| 解析 | `app/parsing/*` | TXT/MD、PDF、DOCX、PPTX、图片 OCR |
| 切片 | `app/chunking.py` | 400/50、结构感知、偏移、去重 |
| 清单 | `app/manifest.py` | SQLite roots/files/sync_runs、扫描 diff |
| 入库 | `app/ingest.py`、`app/jobs.py` | 批量入库、任务进度、恢复 |
| 检索 | `app/search.py` | hybrid + RRF + rerank + 去重/上下文 |
| API | `app/api/*`、`app/models.py` | 搜索/资料库/同步/系统 |
| UI | `web/index.html`、`web/app.js`、`web/styles.css` | 四页本地界面 |
| 评测 | `app/evaluation.py`、`scripts/benchmark.py` | Recall/MRR/nDCG、报告 |
| CLI | `app/cli.py` | scan/index/search/status |

数据流：

```text
文件 → Parser → ParsedDocument → Chunker → EmbeddingService
     → LanceDB merge_insert + run_id 清理 → manifest 状态
查询 → encode_query → hybrid(RRF) → top_k → rerank → 去重/上下文 → final_k
```

---

## 4. 当前验证状态

- `pytest`：176 个测试通过；
- `ruff check app tests scripts`：通过；
- `python -m app --check`：通过；
- `scripts/run_web.sh`：启动成功，健康检查 `status=ok`、`offline=true`；
- Playwright：搜索/资料库/扫描 UI 冒烟通过；
- 真实模型 smoke：E1024 命中 PDF，冷启动约 583ms；
- 评测 smoke（5 条 query）：hybrid+rerank nDCG@10 0.852，p50 152ms。

---

## 5. 已知限制（不要重复报为“未实现”）

- 音视频未实现；
- 图片只 OCR，不做图像语义；
- PDF 文本页内嵌图默认不逐张 OCR；
- 已有 v1.1 活动评测集（docs MD / PDF / 简历候选人 / 38 条 / T2 子集）；
- 单人 localhost，无登录/权限；
- PyMuPDF 为 AGPL。

---

## 6. 重点审查清单

### 6.1 正确性

- LanceDB 0.40 的 `create_index("text", config=FTS(...))`、`merge_insert`、`delete`、`table.search()` 用法是否正确；
- `run_id` 清理策略是否会在崩溃/并发/重试时丢数据或留重复；
- `manifest.scan` 是否会在权限错误、超大文件、symlink、网络盘离线时误删；
- `chunking` 的偏移、重叠、页/幻灯片边界、短块合并是否正确；
- 解析器对损坏/加密/空文件/编码异常是否可靠；
- 检索的 `rank_before/rank_after`、`rrf_score`、`rerank_score` 语义是否一致；
- `final_k > top_k`、空 query、过滤为空等边界。

### 6.2 一致性与并发

- SQLite 连接在多线程/任务中的使用；
- `JobManager` 单 worker 是否真的串行；
- 扫描/同步/搜索并发时的数据一致性；
- 应用关闭时任务、manifest、LanceDB 的清理；
- 长任务取消后的状态。

### 6.3 安全

- `POST /api/open` 的路径校验是否能被绕过（`..`、symlink、前缀混淆）；
- 前端是否有 XSS（`innerHTML`、高亮、路径）；
- API 是否只监听 `127.0.0.1`；
- 日志是否打印正文/敏感路径；
- 请求体大小/路径数量限制。

### 6.4 性能

- 每次请求是否重复创建模型/服务；
- 上下文扩展是否 N+1；
- 大批量入库的内存峰值；
- 10 万级 chunk 的检索/重排延迟；
- 启动预热是否完成后再提供请求；
- 是否有不必要的全表读取。

### 6.5 工程质量

- 测试是否覆盖关键失败路径；
- 异常是否被静默吞掉；
- 类型/接口/命名是否一致；
- 文档与代码是否不一致；
- 配置默认值是否安全。

---

## 7. 输出格式（必须遵守）

```markdown
# 代码审查结果

## P0 阻断
| # | 位置 | 问题 | 影响 | 建议 |
|---|---|---|---|---|

## P1 高
...

## P2 中
...

## P3 低/建议
...

## 测试缺口
...
```

要求：

- 每条必须给出 `path:line`；
- 区分“确定 bug”和“风险/建议”；
- 不要修改代码；
- 不要重复“已知限制”；
- 如果没发现问题，明确写“未发现 P0/P1 问题”，并列出仍然建议补的测试。

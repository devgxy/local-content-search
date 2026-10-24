# 从文件到答案：RAG 工程手册

这是一篇面向两类读者的 RAG 实战文档：

- 初学者：按“概念 → 例子 → 流程”阅读；
- 有经验开发者：直接看“工程细节、参数、失败案例、指标”。

主线：

```text
数据清洗 → Chunk → 向量化 → 存储 → 检索 → 结果呈现 → 评价指标
```

支线：

```text
静态文档 vs 动态文档
→ 版本、增量、删除、权限、评测快照
```

## 文章结构

### 第 0 章 导读

- RAG 是什么，不是什么；
- 两条阅读路线；
- 术语约定；
- 项目边界。

### 第 1 章 RAG 全局链路

- Retriever、Generator、Corpus、Query、Chunk；
- 召回与重排；
- 为什么 RAG 不是“向量库 + 大模型”。

### 第 2 章 数据清洗

- 文件发现、范围控制；
- 编码、页眉页脚、重复内容；
- 结构化解析；
- OCR、bbox、表格；
- 元数据、权限、隐私；
- 失败显式暴露。

### 第 3 章 Chunk

- 固定长度、递归、结构感知、语义切分；
- 父子 chunk、small-to-big、contextual retrieval；
- chunk size / overlap / top_k 的权衡；
- 表格、代码、问答对的特殊切法。

### 第 4 章 向量化

- dense / sparse；
- bi-encoder / cross-encoder；
- query/passage 指令；
- 维度、归一化、batch、截断；
- 模型版本与索引版本绑定。

### 第 5 章 存储

- 向量库、全文索引、元数据；
- Flat / HNSW / IVF / PQ；
- BM25、倒排索引、ICU 分词；
- prefilter / postfilter；
- Manifest、run_id、原子切换、tombstone。

### 第 6 章 检索

- vector / FTS / hybrid；
- RRF 公式；
- rerank；
- scope、auto、top_k、final_k；
- 去重、per-doc limit、doc-level；
- 查询改写、多 query、HyDE。

### 第 7 章 结果呈现

- 结果卡片；
- snippet、highlight、summary；
- 页码、路径、引用；
- 表格 HTML、置信度、warning；
- chunk-level 与 doc-level；
- 证据展示与“找不到就说找不到”。

### 第 8 章 评价指标

- Recall@K、Precision@K、MRR、nDCG、MAP、Hit Rate；
- 公式与手算例子；
- chunk-level 与 doc-level；
- RAG 生成指标；
- 业务指标；
- 小样本与统计风险。

### 第 9 章 静态文档 vs 动态文档

- 内容、结构、实时性三类动态文档；
- 索引方式、ID、删除、一致性、权限、评测差异；
- 为什么动态文档的工程复杂度更高。

### 第 10 章 项目实战

- 本地混合检索架构；
- Manifest / run_id / scope / auto；
- 表格与 OCR；
- 当前指标与失败复盘；
- 可复现命令。

### 第 11 章 隐私、开源与发布

- 绝对路径、姓名、电话、邮箱脱敏；
- Git 历史清理；
- 数据集/模型许可证；
- GitHub、CSDN、微信公众号发布顺序。

## 发布拆分

微信公众号建议拆成 6 篇：

1. RAG 全局链路与术语
2. 数据清洗与 Chunk
3. 向量化与存储
4. 检索、重排与结果呈现
5. 评价指标
6. 静态/动态文档与项目实战

CSDN 可合并为 2–3 篇长文，保留完整公式和代码。

## 写作约定

- 每章先给“一句话结论”；
- 术语首次出现时解释；
- 公式给一个手算例子；
- 工程细节用“进阶”标记；
- 不写真实姓名、公司、电话、邮箱、薪资、简历路径；
- 项目路径使用仓库相对路径；
- 指标必须注明数据规模与口径。

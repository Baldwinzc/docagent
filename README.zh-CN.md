# docagent — 面向本地文档的 agentic RAG 问答

[English](README.md) | **中文**

针对你自己的文件(Markdown / 文本 / PDF)用自然语言提问,得到**精确到来源行号**的带引用答案。基于 [LangGraph](https://langchain-ai.github.io/langgraph/) 构建。

docagent 不是一次性的 RAG demo。它在一条真正的检索管线(混合 dense+BM25、cross-encoder 重排、相关性阈值)之上跑 **agentic 检索循环**,强制答案带精确到行号的引用,提供检索 trace,并自带一套**量化评估**。

## 特性

- 🔁 **Agentic 检索** —— agent 可以检索、检查、改写查询、再检索,然后才作答。
- 🧪 **混合检索 + 重排** —— dense(bge embedding)**+** BM25,用 Reciprocal-Rank-Fusion 融合,再经 cross-encoder 重排,再用相关性阈值过滤(这也是它能诚实说"文档里没有"的机制)。
- 📎 **精确、强制的引用** —— 答案引用精确 locator,如 `tutorial-path-params.md:L65-91`,由 `Answer` 工具**强制**附带引用。
- 🧭 **意图路由** —— 前置路由先拒掉超范围问题。
- 🔭 **可观测性** —— 每一步检索都记录进 `trace`(CLI 加 `--trace` 查看)。
- 🛡️ **健壮性** —— 空知识库守卫、工具失败捕获、递归上限。
- 📊 **量化评估** —— 在带标注的 QA 集上算 意图/召回/答案/引用/拒答 指标(见 [评估](#评估))。
- 🔒 检索全程**本地 embedding、无需 API key**;只有作答 LLM 需要 key。

## 架构

```mermaid
flowchart TB
    START([START]) --> R[intent_router 意图路由]
    R -- 超范围 / 空库 --> D([END · 拒答])
    R -- 范围内 --> L[llm_call]
    L --> C{是否终止工具?}
    C -- search_docs --> E[environment 执行]
    E --> L
    C -- Answer --> X([END · 答案 + 引用])

    E -.调用.-> PIPE

    subgraph PIPE [混合检索管线]
      direction TB
      Q[query] --> DENSE[dense · bge]
      Q --> BM25[BM25]
      DENSE --> RRF[RRF 融合]
      BM25 --> RRF
      RRF --> RERANK[cross-encoder 重排]
      RERANK --> THRESH[相关性阈值]
      THRESH --> OUT2[top-k chunks + locators]
    end
```

## 快速开始

```bash
# 1. 环境（Python 3.11）
conda create -n docagent python=3.11 -c conda-forge
conda activate docagent
pip install -e .

# 2. 配置作答 LLM
cp .env.example .env          # 然后把 OPENAI_API_KEY 填进 .env
#   （或设 LLM_MODEL=ollama:llama3.1 完全本地运行）

# 3. 建立知识库（自带 FastAPI 文档，或把 --path 指向你自己的文件夹）
python -m docagent.ingest --path ./corpus/fastapi --reset

# 4. 提问
python -m docagent.ask --trace "How do I declare an integer path parameter?"
```

## 运行示例

**范围内问题** —— agent 先检索,再带精确到行的引用作答:

```console
$ python -m docagent.ask --trace "How do I declare a path parameter that must be an integer, and what does FastAPI do if the client sends a non-integer?"
🔎 Intent: IN_SCOPE — retrieving from knowledge base
=== trace ===
  1. search_docs  query='FastAPI path parameter integer non-integer validation'

=== Answer ===
Declare the path parameter with a Python type annotation, e.g. `item_id: int`.
FastAPI validates the value and returns a validation error if the client sends a
non-integer [tutorial-path-params.md:L65-91].

=== Citations ===
- tutorial-path-params.md:L65-91
- tutorial-path-params.md:L89-107
```

**超范围问题** —— 路由直接拒答,不浪费检索:

```console
$ python -m docagent.ask "What is the capital of France?"
🚫 Intent: OUT_OF_SCOPE — politely declining
This question is outside the scope of the local knowledge base, so I can't
answer it from the available documents.
```

也可以用探针脚本直接查看检索栈(无需 key):

```bash
python scripts/check_retrieval.py
```

## 检索管线

`search_docs` **不是**朴素的 top-k 余弦相似。对每个查询:

1. **Dense** 检索:`bge-small-en-v1.5` embedding(取 top *candidate_k*)。
2. **Sparse** 检索:**BM25** 关键词(取 top *candidate_k*)。
3. **RRF 融合**:合并两路排名(任一信号弱时仍鲁棒)。
4. **Cross-encoder 重排**(`ms-marco-MiniLM-L-6-v2`):对每个候选与查询打分。
5. **相关性阈值**:丢弃低分 chunk —— 所以域外查询返回空,这正是 agent 能答"文档里没有"的依据。

每个保留下来的 chunk 携带精确出处(`source:Lstart-Lend`,或 PDF 页码)用于引用。

## 评估

带标注的 QA 集(`src/docagent/eval/qa_dataset.py`)覆盖 单文档、多跳、超范围、无答案 四类问题。运行:

```bash
python -m docagent.eval.run_eval
```

在自带 FastAPI 语料(206 chunks / 12 文档)、作答 LLM `gpt-5.4-mini` 上的最新结果:

| 指标 | 结果 |
|---|---|
| 意图路由准确率 | **10/10 (100%)** |
| 检索召回(均值) | **0.94** |
| 答案正确率(LLM 评判) | **7/8 (88%)** |
| 引用准确率 | **7/8 (88%)** |
| 拒答准确率 | **2/2 (100%)** |

唯一失分是一个**多跳**问题:它需要同时用到两个文档的事实,agent 只检索到其中一个。多跳综合(子查询拆解)是后续改进方向。

## 目录结构

```
src/docagent/
├── agent.py            # LangGraph：intent_router + 应答循环 + trace/守卫
├── retriever.py        # 混合检索：dense+BM25 -> RRF -> 重排 -> 阈值
├── ingest.py           # CLI：加载 -> 切块(带行号出处) -> 向量化 -> Chroma
├── ask.py              # CLI：提问（--trace 查看检索步骤）
├── vectorstore.py      # 共享 embedding + Chroma 后端
├── configuration.py    # 可被环境变量覆盖的配置
├── prompts.py          # 意图 / agent 提示词
├── schemas.py          # 图状态（含 trace 通道）+ 结构化输出 schema
├── tools/
│   ├── base.py         # 工具注册中心
│   └── retrieval_tools.py  # search_docs, list_sources, Answer, Question
└── eval/
    ├── qa_dataset.py   # 带标注的 QA 用例
    └── run_eval.py     # 指标：意图 / 召回 / 答案 / 引用 / 拒答
corpus/fastapi/         # 示例语料（FastAPI 文档子集，MIT — 见 SOURCE.md）
scripts/check_retrieval.py  # 检索栈开发探针（无需 key）
tests/                  # 检索测试（无需 key）+ LLM 端到端测试
```

## 测试

```bash
python tests/run_all_tests.py          # 仅检索测试（无需 API key）
python tests/run_all_tests.py --all    # + LLM 端到端（需要 API key）
```

`tests/test_retrieval.py` 测试混合检索器(命中、精确行号 locator、阈值),**无需 key**,CI 中运行。

## 配置

在 `.env` 中设置(见 `.env.example`):

| 变量 | 默认值 | 用途 |
|---|---|---|
| `OPENAI_API_KEY` | — | 作答模型的 key |
| `LLM_MODEL` | `openai:gpt-4.1` | 任意 `init_chat_model` id,如 `ollama:llama3.1` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | 本地 dense embedding 模型 |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | cross-encoder 重排模型 |
| `TOP_K` / `CANDIDATE_K` | `4` / `20` | 最终命中数 / 每路候选数 |
| `SCORE_THRESHOLD` | `0.0` | 保留 chunk 的最低重排分 |
| `CHROMA_PATH` / `CHROMA_COLLECTION` | `./chroma_db` / `docagent` | 向量库 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | 索引切块参数 |

## 技术栈

LangGraph · LangChain · Chroma · sentence-transformers (bge) · rank-bm25 ·
cross-encoder · pypdf

## 许可证

MIT（本项目）。`corpus/fastapi/` 下的示例语料是 FastAPI 文档的子集,依其 MIT 许可再分发 —— 见 `corpus/fastapi/SOURCE.md`。

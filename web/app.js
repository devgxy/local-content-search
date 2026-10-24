/* 本地内容检索 · 前端逻辑（无框架、无构建） */
"use strict";

const state = {
  view: "search",
  page: 1,
  pageSize: 20,
  roots: [],
  diff: null,
  currentDoc: null,
  currentJob: null,
  highlightTerms: [],
  selectedDocs: new Set(),
  libraryItems: [],
  currentHit: null,
  currentOriginal: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function badge(text, className = "") {
  return el("span", `badge ${className}`.trim(), text);
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatTime(value) {
  if (!value) return "—";
  const date = typeof value === "number" ? new Date(value * 1000) : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("zh-CN", { hour12: false });
}

function toast(message, kind = "info") {
  const node = $("#toast");
  node.textContent = message;
  node.className = `toast ${kind}`;
  window.clearTimeout(toast._timer);
  toast._timer = window.setTimeout(() => node.classList.add("hidden"), 4000);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch (error) {
    payload = null;
  }
  if (!response.ok) {
    const detail = payload && payload.detail ? payload.detail : {};
    const message = detail.message || `请求失败（HTTP ${response.status}）`;
    const hint = detail.hint ? `：${detail.hint}` : "";
    throw new Error(`${message}${hint}`);
  }
  return payload;
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function previewAround(text, terms, limit = 280) {
  if (!text || text.length <= limit) return text;
  const candidates = (terms || []).filter((term) => term && term.length >= 2);
  let position = -1;
  for (const term of candidates) {
    const found = text.toLowerCase().indexOf(term.toLowerCase());
    if (found >= 0 && (position < 0 || found < position)) position = found;
  }
  if (position < 0) return `${text.slice(0, limit).trim()}…`;
  let start = Math.max(0, position - Math.floor(limit * 0.25));
  let end = Math.min(text.length, start + limit);
  if (end - start < limit) start = Math.max(0, end - limit);
  return `${start > 0 ? "…" : ""}${text.slice(start, end).trim()}${end < text.length ? "…" : ""}`;
}

function highlight(text, terms) {
  const fragment = document.createDocumentFragment();
  const cleaned = (terms || []).filter((term) => term && term.length >= 2);
  if (!cleaned.length) {
    fragment.appendChild(document.createTextNode(text));
    return fragment;
  }
  cleaned.sort((a, b) => b.length - a.length);
  const pattern = new RegExp(`(${cleaned.map(escapeRegExp).join("|")})`, "gi");
  let last = 0;
  let match;
  let count = 0;
  while ((match = pattern.exec(text)) !== null && count < 50) {
    if (match.index > last) {
      fragment.appendChild(document.createTextNode(text.slice(last, match.index)));
    }
    const mark = document.createElement("mark");
    mark.textContent = match[0];
    fragment.appendChild(mark);
    last = match.index + match[0].length;
    count += 1;
    if (match[0].length === 0) pattern.lastIndex += 1;
  }
  if (last < text.length) {
    fragment.appendChild(document.createTextNode(text.slice(last)));
  }
  return fragment;
}

/* ------------------------------------------------------------------ 导航 */
function switchView(view) {
  state.view = view;
  $$(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.view === view));
  $$(".view").forEach((section) => section.classList.remove("active"));
  const target = $(`#view-${view}`);
  if (target) target.classList.add("active");
  if (view === "library") loadLibrary(1);
  if (view === "settings") {
    loadHealth();
    loadStats();
  }
  history.replaceState(null, "", `#${view}`);
}

/* ------------------------------------------------------------------ 状态 */
async function loadHealth() {
  try {
    const health = await api("/api/health");
    const dot = $("#status-dot");
    dot.className = `dot ${health.status}`;
    $("#status-text").textContent = health.status === "ok" ? "就绪" : "降级";
    const embed = health.models.embedder || {};
    const rerank = health.models.reranker || {};
    $("#status-model").textContent =
      `向量 ${embed.device || "?"} · 重排 ${rerank.ready ? "就绪" : "待加载"}`;

    const cards = $("#health-cards");
    cards.replaceChildren();
    const items = [
      ["服务", health.status],
      ["版本", health.version],
      ["向量模型", embed.ready ? "已加载" : embed.available ? "可用" : "缺失"],
      ["向量设备", embed.device || "—"],
      ["重排模型", rerank.ready ? "已加载" : rerank.available ? "可用" : "缺失"],
      ["FTS 索引", health.db.fts_index ? "正常" : "缺失"],
      ["Chunk 数", health.db.chunks],
      ["离线模式", health.offline ? "是" : "未设置 HF_HUB_OFFLINE"],
    ];
    for (const [label, value] of items) {
      const card = el("div", "card-item");
      card.appendChild(el("div", "label", label));
      card.appendChild(el("div", "value", String(value)));
      cards.appendChild(card);
    }
  } catch (error) {
    toast(`健康检查失败：${error.message}`, "error");
  }
}

async function loadStats() {
  try {
    const payload = await api("/api/stats");
    const manifest = payload.manifest || {};
    const db = payload.db || {};
    $("#status-stats").textContent =
      `文档 ${manifest.files || 0} · Chunk ${db.chunks || 0} · 上次同步 ${formatTime(manifest.last_sync_at)}`;

    const cards = $("#stats-cards");
    cards.replaceChildren();
    const statuses = manifest.by_status || {};
    const items = [
      ["文档", manifest.files || 0],
      ["Chunk", db.chunks || 0],
      ["索引根", manifest.roots || 0],
      ["库大小", `${db.db_size_mb || 0} MB`],
      ["已索引", statuses.indexed || 0],
      ["错误", statuses.error || 0],
      ["待同步", statuses.pending || 0],
    ];
    for (const [label, value] of items) {
      const card = el("div", "card-item");
      card.appendChild(el("div", "label", label));
      card.appendChild(el("div", "value", String(value)));
      cards.appendChild(card);
    }
  } catch (error) {
    toast(`统计加载失败：${error.message}`, "error");
  }
}

/* ------------------------------------------------------------------ 搜索 */
async function runSearch() {
  const query = $("#query").value.trim();
  if (!query) {
    toast("请输入搜索内容", "warn");
    return;
  }
  const button = $("#search-btn");
  button.disabled = true;
  button.textContent = "搜索中…";
  $("#search-meta").textContent = "";
  try {
    const payload = await api("/api/search", {
      method: "POST",
      body: JSON.stringify({
        query,
        top_k: Number($("#top-k").value) || 20,
        final_k: Number($("#final-k").value) || 5,
        file_types: $("#filter-type").value ? [$("#filter-type").value] : [],
        root_id: $("#filter-root").value ? Number($("#filter-root").value) : null,
        doc_ids: state.selectedDocs.size ? Array.from(state.selectedDocs) : [],
        mode: $("#search-mode").value || null,
        profile: $("#search-profile").value || null,
        doc_level: $("#doc-level").checked,
      }),
    });
    renderResults(payload);
  } catch (error) {
    toast(`搜索失败：${error.message}`, "error");
  } finally {
    button.disabled = false;
    button.textContent = "搜索";
  }
}

function renderResults(payload) {
  const container = $("#results");
  container.replaceChildren();
  state.highlightTerms = payload.highlight_terms || [];
  const meta = [];
  meta.push(`${payload.took_ms} ms`);
  meta.push(`${payload.candidates} 候选`);
  if (payload.warnings && payload.warnings.length) meta.push(payload.warnings.join(" / "));
  $("#search-meta").textContent = meta.join(" · ");

  if (!payload.results.length) {
    container.appendChild(el("div", "empty", "没有找到结果。试试更短的关键词，或放宽过滤条件。"));
    return;
  }

  for (const hit of payload.results) {
    const card = el("article", "result");
    const head = el("div", "result-head");
    head.appendChild(badge(hit.file_type.toUpperCase(), `type-${hit.file_type}`));
    if (hit.source_kind === "ocr") head.appendChild(badge("OCR", "ocr"));
    if (hit.source_kind === "table") head.appendChild(badge("表格", "table"));
    const name = el("span", "file-name", hit.file_name);
    head.appendChild(name);
    if (hit.page) head.appendChild(el("span", "muted", `第 ${hit.page} 页`));
    if (hit.slide) head.appendChild(el("span", "muted", `第 ${hit.slide} 页幻灯片`));
    const baseLabel = hit.score_kind === "rrf"
      ? "RRF"
      : hit.score_kind === "bm25"
        ? "BM25"
        : hit.score_kind === "distance"
          ? "相似"
          : hit.score_kind === "image"
            ? "图片"
            : "分";
    const score = hit.rerank_score !== null
      ? `重排 ${(hit.rerank_score * 100).toFixed(0)}%`
      : `${baseLabel} ${hit.score.toFixed(4)}`;
    head.appendChild(el("span", "score", score));
    if (hit.rerank_score !== null) {
      head.appendChild(el("span", "muted", `${baseLabel} ${hit.rrf_score.toFixed(4)}`));
    }
    card.appendChild(head);

    const body = el("div", "result-body");
    const thumb = resultThumb(hit);
    thumb.addEventListener("click", () => openOriginal(hit));
    body.appendChild(thumb);
    const content = el("div", "result-content");
    content.appendChild(el("div", "path", hit.file_path));
    if (hit.heading) content.appendChild(el("div", "heading", hit.heading));
    const summary = el("div", "summary");
    summary.appendChild(badge("摘要", "summary-badge"));
    const summaryText = hit.summary || previewAround(hit.snippet, state.highlightTerms, 240);
    summary.appendChild(highlight(summaryText, state.highlightTerms));
    content.appendChild(summary);
    if (hit.table_html) {
      const tableWrap = el("div", "table-preview");
      const safeTable = safeTableFromHtml(hit.table_html);
      if (!safeTable) {
        tableWrap.appendChild(el("div", "muted", "表格结构无效"));
      } else {
        tableWrap.appendChild(safeTable);
      }
      content.appendChild(tableWrap);
      if (typeof hit.table_confidence === "number") {
        content.appendChild(
          el(
            "div",
            hit.table_confidence < 0.5 ? "muted table-low" : "muted",
            `表格结构置信度 ${(hit.table_confidence * 100).toFixed(0)}%`
          )
        );
      }
    }
    const actions = el("div", "actions");
    const preview = el("button", "primary", "查看命中分块");
    preview.addEventListener("click", () => openHit(hit));
    const original = el("button", "", "原文件预览");
    original.addEventListener("click", () => openOriginal(hit));
    const open = el("button", "", "打开原文件");
    open.addEventListener("click", () => openPath(hit.file_path, false));
    const reveal = el("button", "", "在 Finder 中显示");
    reveal.addEventListener("click", () => openPath(hit.file_path, true));
    actions.append(preview, original, open, reveal);
    content.appendChild(actions);
    body.appendChild(content);
    card.appendChild(body);
    container.appendChild(card);
  }
}

function safeTableFromHtml(source) {
  if (typeof source !== "string") return null;
  const template = document.createElement("template");
  template.innerHTML = source;
  const sourceTable = template.content.querySelector("table");
  if (!sourceTable) return null;
  const table = document.createElement("table");
  for (const sourceRow of sourceTable.querySelectorAll("tr")) {
    const row = document.createElement("tr");
    for (const sourceCell of sourceRow.querySelectorAll(":scope > th, :scope > td")) {
      const cell = document.createElement(sourceCell.tagName === "TH" ? "th" : "td");
      const colspan = Number(sourceCell.getAttribute("colspan"));
      const rowspan = Number(sourceCell.getAttribute("rowspan"));
      if (Number.isInteger(colspan) && colspan > 1 && colspan <= 20) cell.colSpan = colspan;
      if (Number.isInteger(rowspan) && rowspan > 1 && rowspan <= 20) cell.rowSpan = rowspan;
      cell.textContent = sourceCell.textContent || "";
      row.appendChild(cell);
    }
    table.appendChild(row);
  }
  return table;
}

function resultThumb(hit) {
  const box = el("div", "result-thumb");
  if (hit.file_type === "pdf") {
    const image = document.createElement("img");
    image.src = `/api/documents/${encodeURIComponent(hit.doc_id)}/preview?page=${hit.page || 1}`;
    image.alt = "PDF 缩略图";
    image.loading = "lazy";
    box.appendChild(image);
  } else if (hit.file_type === "image") {
    const image = document.createElement("img");
    image.src = `/api/documents/${encodeURIComponent(hit.doc_id)}/file`;
    image.alt = hit.file_name;
    image.loading = "lazy";
    box.appendChild(image);
  } else {
    box.appendChild(el("span", "thumb-placeholder", hit.file_type.toUpperCase()));
  }
  box.title = "点击预览原文件";
  return box;
}

function openOriginal(hit) {
  state.currentOriginal = hit;
  $("#original-title").textContent = hit.file_name;
  $("#original-meta").textContent = `${hit.file_path}${hit.page ? ` · 第 ${hit.page} 页` : ""}`;
  const body = $("#original-body");
  body.replaceChildren();
  if (hit.file_type === "pdf") {
    const image = document.createElement("img");
    image.src = `/api/documents/${encodeURIComponent(hit.doc_id)}/preview?page=${hit.page || 1}`;
    image.alt = `${hit.file_name} 第 ${hit.page || 1} 页`;
    body.appendChild(image);
    body.appendChild(el("div", "muted", "当前页预览；完整 PDF 请点“打开原文件”。"));
  } else if (hit.file_type === "image") {
    const image = document.createElement("img");
    image.src = `/api/documents/${encodeURIComponent(hit.doc_id)}/file`;
    image.alt = hit.file_name;
    body.appendChild(image);
  } else if (hit.file_type === "md" || hit.file_type === "txt") {
    const pre = el("pre", "original-text", "加载中…");
    body.appendChild(pre);
    fetch(`/api/documents/${encodeURIComponent(hit.doc_id)}/file`)
      .then((response) => response.text())
      .then((text) => {
        pre.textContent = text.slice(0, 50000);
      })
      .catch(() => {
        pre.textContent = "读取失败，请点“打开原文件”。";
      });
  } else {
    body.appendChild(el("div", "empty", "该格式暂不支持浏览器内嵌预览，请点“打开原文件”。"));
  }
  $("#original-dialog").showModal();
}

function openHit(hit) {
  state.currentHit = hit;
  $("#hit-title").textContent = hit.file_name;
  const bits = [hit.file_path];
  if (hit.page) bits.push(`第 ${hit.page} 页`);
  if (hit.slide) bits.push(`第 ${hit.slide} 页幻灯片`);
  if (hit.heading) bits.push(hit.heading);
  $("#hit-meta").textContent = bits.join(" · ");
  const box = $("#hit-context");
  box.replaceChildren();
  const context = el("div", "hit-snippet");
  context.appendChild(
    highlight(previewAround(hit.snippet, state.highlightTerms, 800), state.highlightTerms),
  );
  box.appendChild(context);
  $("#hit-dialog").showModal();
}

async function openPath(path, reveal) {
  try {
    await api("/api/open", {
      method: "POST",
      body: JSON.stringify({ path, reveal: Boolean(reveal) }),
    });
  } catch (error) {
    toast(`打开失败：${error.message}`, "error");
  }
}

/* ------------------------------------------------------------------ 资料库 */
function updateScopeUI() {
  const count = state.selectedDocs.size;
  $("#scope-count").textContent = String(count);
  $("#scope-label").textContent = count ? `范围：已选 ${count} 个文件` : "范围：全部文件";
  $("#clear-scope").classList.toggle("hidden", count === 0);
}

function toggleSelected(docId, checked) {
  if (checked) state.selectedDocs.add(docId);
  else state.selectedDocs.delete(docId);
  updateScopeUI();
}

async function loadLibrary(page = 1) {
  state.page = page;
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(state.pageSize),
  });
  const query = $("#library-query").value.trim();
  if (query) params.set("q", query);
  try {
    const payload = await api(`/api/library?${params.toString()}`);
    const body = $("#library-body");
    body.replaceChildren();
    state.libraryItems = payload.items;
    if (!payload.items.length) {
      const row = el("tr");
      const cell = el("td", "empty", "资料库为空，请到“同步”添加文件夹。");
      cell.colSpan = 7;
      row.appendChild(cell);
      body.appendChild(row);
    }
    for (const item of payload.items) {
      const row = el("tr");
      const checkCell = el("td", "check");
      const check = document.createElement("input");
      check.type = "checkbox";
      check.checked = state.selectedDocs.has(item.doc_id);
      check.addEventListener("change", () => toggleSelected(item.doc_id, check.checked));
      checkCell.appendChild(check);
      row.appendChild(checkCell);
      const nameCell = el("td");
      const nameButton = el("button", "link", item.file_name);
      nameButton.addEventListener("click", () => openDocument(item.doc_id));
      nameCell.appendChild(nameButton);
      nameCell.appendChild(el("div", "path small", item.file_path));
      row.appendChild(nameCell);
      row.appendChild(el("td", "", item.file_type));
      row.appendChild(el("td", "", String(item.chunks)));
      const statusCell = el("td");
      statusCell.appendChild(badge(item.status, `status-${item.status}`));
      if (item.last_error) statusCell.title = item.last_error;
      row.appendChild(statusCell);
      row.appendChild(el("td", "muted", formatTime(item.indexed_at)));
      const actions = el("td", "actions");
      const reindex = el("button", "", "重建");
      reindex.addEventListener("click", () => reindexPath(item.file_path));
      actions.appendChild(reindex);
      row.appendChild(actions);
      body.appendChild(row);
    }
    $("#library-total").textContent = `共 ${payload.total} 个文档`;
    updateScopeUI();
    $("#library-page").textContent = String(payload.page);
    $("#library-prev").disabled = payload.page <= 1;
    $("#library-next").disabled = payload.page * payload.page_size >= payload.total;
  } catch (error) {
    toast(`资料库加载失败：${error.message}`, "error");
  }
}

async function openDocument(docId, page = 1) {
  try {
    const payload = await api(`/api/documents/${encodeURIComponent(docId)}`);
    state.currentDoc = payload.item;
    $("#doc-title").textContent = payload.item.file_name;
    $("#doc-meta").textContent =
      `${payload.item.file_path} · ${formatBytes(payload.item.size)} · ${payload.item.chunks} chunks`;
    renderDocPreview(payload.item, page);
    const list = $("#doc-chunks");
    list.replaceChildren();
    for (const chunk of payload.chunks) {
      const block = el("div", "chunk");
      const head = el("div", "chunk-head");
      head.appendChild(el("span", "muted", `#${chunk.chunk_index}`));
      if (chunk.page) head.appendChild(el("span", "muted", `第 ${chunk.page} 页`));
      if (chunk.slide) head.appendChild(el("span", "muted", `第 ${chunk.slide} 页幻灯片`));
      if (chunk.heading) head.appendChild(el("span", "heading", chunk.heading));
      block.appendChild(head);
      block.appendChild(el("div", "chunk-text", chunk.display_text));
      list.appendChild(block);
    }
    $("#doc-dialog").showModal();
  } catch (error) {
    toast(`文档加载失败：${error.message}`, "error");
  }
}

function renderDocPreview(item, page = 1) {
  const box = $("#doc-preview");
  box.replaceChildren();
  box.classList.remove("hidden");
  const fileUrl = `/api/documents/${encodeURIComponent(item.doc_id)}/file`;
  if (item.file_type === "pdf") {
    const image = document.createElement("img");
    image.src = `/api/documents/${encodeURIComponent(item.doc_id)}/preview?page=${page}`;
    image.alt = `${item.file_name} 第 ${page} 页`;
    image.onerror = () => {
      box.replaceChildren();
      const frame = document.createElement("iframe");
      frame.src = `${fileUrl}#page=${page}`;
      frame.title = "PDF 预览";
      box.appendChild(frame);
    };
    box.appendChild(image);
    box.appendChild(el("div", "muted", `第 ${page} 页（点“打开文件”查看完整 PDF）`));
    return;
  }
  if (item.file_type === "image") {
    const image = document.createElement("img");
    image.src = fileUrl;
    image.alt = item.file_name;
    box.appendChild(image);
    return;
  }
  const note = el(
    "div",
    "muted",
    item.file_type === "md" || item.file_type === "txt"
      ? "文本提取预览（下方为切片内容）"
      : "该格式暂不支持浏览器内嵌预览，下方为提取文本；可点“打开文件”用系统应用查看。",
  );
  box.appendChild(note);
}

async function reindexPath(path) {
  try {
    const payload = await api("/api/reindex", {
      method: "POST",
      body: JSON.stringify({ path }),
    });
    toast("已提交重建任务");
    await pollJob(payload.job_id, () => {});
    toast("重建完成");
    loadLibrary(state.page);
    loadStats();
  } catch (error) {
    toast(`重建失败：${error.message}`, "error");
  }
}

/* ------------------------------------------------------------------ 同步 */
async function loadRoots() {
  try {
    state.roots = await api("/api/roots");
    const list = $("#roots-list");
    list.replaceChildren();
    if (!state.roots.length) {
      list.appendChild(el("li", "muted", "还没有索引根。添加一个文件夹开始。"));
    }
    for (const root of state.roots) {
      const item = el("li", "root-item");
      item.appendChild(el("span", "path", root.path));
      item.appendChild(el("span", "muted", `${root.doc_count} 文档`));
      if (!root.enabled) item.appendChild(badge("已停用", "warn"));
      const remove = el("button", "", "移除");
      remove.addEventListener("click", () => removeRoot(root.id));
      item.appendChild(remove);
      list.appendChild(item);
    }
    const select = $("#filter-root");
    const current = select.value;
    select.replaceChildren(new Option("全部", ""));
    for (const root of state.roots.filter((item) => item.enabled)) {
      select.appendChild(new Option(root.path, String(root.id)));
    }
    select.value = current;
  } catch (error) {
    toast(`索引根加载失败：${error.message}`, "error");
  }
}

async function addRoot() {
  const path = $("#root-input").value.trim();
  if (!path) {
    toast("请输入文件夹路径", "warn");
    return;
  }
  try {
    await api("/api/roots", { method: "POST", body: JSON.stringify({ path }) });
    $("#root-input").value = "";
    await loadRoots();
    toast("已添加索引根");
  } catch (error) {
    toast(`添加失败：${error.message}`, "error");
  }
}

async function removeRoot(rootId) {
  if (!window.confirm("移除索引根并删除其索引数据？不会删除电脑上的原文件。")) return;
  try {
    const payload = await api(`/api/roots/${rootId}?delete_files=true`, { method: "DELETE" });
    if (payload.job_id) {
      const job = await pollJob(payload.job_id, (current) => updateProgress(current));
      if (job.status !== "done") throw new Error(job.error || job.status);
    }
    await Promise.all([loadRoots(), loadStats(), loadLibrary(1)]);
    toast("已移除索引根及其索引数据");
  } catch (error) {
    toast(`移除失败：${error.message}`, "error");
  }
}

async function pickFolder() {
  const button = $("#pick-root-btn");
  button.disabled = true;
  try {
    const payload = await api("/api/pick-folder", { method: "POST" });
    $("#root-input").value = payload.path;
    toast("已选择文件夹，点击“添加”");
  } catch (error) {
    if (!error.message.includes("取消")) toast(`选择失败：${error.message}`, "error");
  } finally {
    button.disabled = false;
  }
}

async function scanChanges() {
  const button = $("#scan-btn");
  button.disabled = true;
  try {
    const payload = await api("/api/scan", { method: "POST", body: "{}" });
    const job = await pollJob(payload.job_id, (current) => updateProgress(current));
    if (job.status !== "done") throw new Error(job.error || job.status);
    state.diff = job.result;
    renderDiff(job.result);
  } catch (error) {
    toast(`检查失败：${error.message}`, "error");
  } finally {
    button.disabled = false;
    $("#job-progress").classList.add("hidden");
  }
}

function renderDiff(diff) {
  const added = diff.added || [];
  const modified = diff.modified || [];
  const deleted = diff.deleted || [];
  const errors = diff.errors || [];
  $("#diff-summary").textContent =
    `新增 ${added.length} · 修改 ${modified.length} · 删除 ${deleted.length} · 未变 ${diff.unchanged || 0}` +
    (errors.length ? ` · 错误 ${errors.length}` : "");
  const errorBox = $("#diff-errors");
  errorBox.replaceChildren();
  errorBox.hidden = errors.length === 0;
  if (errors.length) {
    errorBox.appendChild(el("div", "error-title", "扫描不完整：以下路径被跳过"));
    for (const error of errors.slice(0, 20)) {
      errorBox.appendChild(el("div", "error-item", `${error.path}：${error.message}`));
    }
  }
  $("#diff-grid").hidden = false;
  renderDiffList("#diff-added", added, true, "added");
  renderDiffList("#diff-modified", modified, true, "modified");
  renderDiffList("#diff-deleted", deleted, false, "deleted");
  $("#count-added").textContent = String(added.length);
  $("#count-modified").textContent = String(modified.length);
  $("#count-deleted").textContent = String(deleted.length);
  $("#delete-missing").checked = false;
  $("#sync-btn").disabled = !(added.length || modified.length || deleted.length);
  const pending = added.length + modified.length;
  const badgeNode = $("#sync-badge");
  badgeNode.textContent = String(pending);
  badgeNode.classList.toggle("hidden", pending === 0);
}

function renderDiffList(selector, items, checked, kind) {
  const container = $(selector);
  container.replaceChildren();
  if (!items.length) {
    container.appendChild(el("div", "muted", "无"));
    return;
  }
  for (const item of items) {
    const row = el("label", "diff-item");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = checked;
    box.disabled = kind === "deleted";
    box.dataset.kind = kind;
    box.dataset.path = kind === "deleted" ? item : item.path;
    row.appendChild(box);
    row.appendChild(el("span", "path", kind === "deleted" ? item : item.path));
    if (item.size) row.appendChild(el("span", "muted", formatBytes(item.size)));
    container.appendChild(row);
  }
}

async function syncSelected() {
  if (!state.diff) return;
  const paths = [];
  const deleted = [];
  $$("#diff-grid input[type=checkbox]").forEach((box) => {
    if (!box.checked) return;
    if (box.dataset.kind === "deleted") deleted.push(box.dataset.path);
    else paths.push(box.dataset.path);
  });
  const deleteMissing = $("#delete-missing").checked;
  if (!paths.length && !(deleteMissing && deleted.length)) {
    toast("请勾选要同步的文件", "warn");
    return;
  }
  const button = $("#sync-btn");
  button.disabled = true;
  try {
    const payload = await api("/api/sync", {
      method: "POST",
      body: JSON.stringify({ paths, deleted_paths: deleted, delete_missing: deleteMissing }),
    });
    const job = await pollJob(payload.job_id, (current) => updateProgress(current));
    if (job.status !== "done") throw new Error(job.error || job.status);
    toast(`同步完成：索引 ${job.result.indexed || 0}，失败 ${job.result.failed || 0}`);
    state.diff = null;
    $("#diff-grid").hidden = true;
    $("#diff-summary").textContent = "同步完成。可以再次检查更新。";
    $("#sync-badge").classList.add("hidden");
    await Promise.all([loadStats(), loadLibrary(1), loadRoots()]);
  } catch (error) {
    toast(`同步失败：${error.message}`, "error");
  } finally {
    button.disabled = false;
    $("#job-progress").classList.add("hidden");
  }
}

function updateProgress(job) {
  const wrapper = $("#job-progress");
  wrapper.classList.remove("hidden");
  $("#job-bar").style.width = `${Math.round((job.progress || 0) * 100)}%`;
  $("#job-message").textContent =
    job.message || `${job.processed || 0}/${job.total || 0} · ${job.status}`;
}

async function pollJob(jobId, onUpdate, timeoutMs = 300000) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const job = await api(`/api/jobs/${jobId}`);
    if (onUpdate) onUpdate(job);
    if (["done", "failed", "cancelled"].includes(job.status)) return job;
    if (Date.now() > deadline) throw new Error("任务超时，请查看日志或重试");
    await new Promise((resolve) => window.setTimeout(resolve, 1000));
  }
}

async function shutdownService() {
  if (!window.confirm("关闭服务？关闭后网页不可用，需要重新启动。")) return;
  const button = $("#shutdown-btn");
  button.disabled = true;
  button.textContent = "正在关闭…";
  try {
    await api("/api/shutdown", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Local-Action": "1" },
    });
    toast("服务已关闭，可以关闭此页面");
    document.body.classList.add("service-stopped");
    button.textContent = "服务已关闭";
  } catch (error) {
    button.disabled = false;
    button.textContent = "关闭服务";
    toast(`关闭失败：${error.message}`, "error");
  }
}

/* ------------------------------------------------------------------ 初始化 */
function init() {
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => switchView(tab.dataset.view));
  });
  $("#status").addEventListener("click", () => switchView("settings"));
  $("#search-btn").addEventListener("click", runSearch);
  $("#query").addEventListener("keydown", (event) => {
    if (event.key === "Enter") runSearch();
  });
  $("#filter-root").addEventListener("change", runSearch);
  $("#pick-root-btn").addEventListener("click", pickFolder);
  $("#add-root-btn").addEventListener("click", addRoot);
  $("#root-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter") addRoot();
  });
  $("#scan-btn").addEventListener("click", scanChanges);
  $("#sync-btn").addEventListener("click", syncSelected);
  $("#delete-missing").addEventListener("change", () => {
    const enabled = $("#delete-missing").checked;
    $$("#diff-deleted input[type=checkbox]").forEach((box) => {
      box.disabled = !enabled;
      box.checked = enabled;
    });
  });
  $("#library-select-all").addEventListener("change", () => {
    const checked = $("#library-select-all").checked;
    $$("#library-body input[type=checkbox]").forEach((box) => {
      box.checked = checked;
    });
    for (const item of state.libraryItems) {
      toggleSelected(item.doc_id, checked);
    }
  });
  $("#clear-scope").addEventListener("click", () => {
    state.selectedDocs.clear();
    $$("#library-body input[type=checkbox]").forEach((box) => {
      box.checked = false;
    });
    $("#library-select-all").checked = false;
    updateScopeUI();
  });
  $("#library-prev").addEventListener("click", () => loadLibrary(Math.max(1, state.page - 1)));
  $("#library-next").addEventListener("click", () => loadLibrary(state.page + 1));
  $("#library-query").addEventListener("keydown", (event) => {
    if (event.key === "Enter") loadLibrary(1);
  });
  $("#shutdown-btn").addEventListener("click", shutdownService);
  $("#refresh-health").addEventListener("click", () => {
    loadHealth();
    loadStats();
  });
  $("#original-close").addEventListener("click", () => $("#original-dialog").close());
  $("#original-open").addEventListener("click", () => {
    if (state.currentOriginal) openPath(state.currentOriginal.file_path, false);
  });
  $("#hit-close").addEventListener("click", () => $("#hit-dialog").close());
  $("#hit-open").addEventListener("click", () => {
    if (state.currentHit) openPath(state.currentHit.file_path, false);
  });
  $("#hit-finder").addEventListener("click", () => {
    if (state.currentHit) openPath(state.currentHit.file_path, true);
  });
  $("#doc-close").addEventListener("click", () => $("#doc-dialog").close());
  $("#doc-open").addEventListener("click", () => {
    if (state.currentDoc) openPath(state.currentDoc.file_path, false);
  });
  $("#doc-reindex").addEventListener("click", () => {
    if (state.currentDoc) reindexPath(state.currentDoc.file_path);
  });

  const initial = (location.hash || "#search").slice(1);
  switchView(["search", "library", "sync", "settings"].includes(initial) ? initial : "search");
  loadRoots();
  loadHealth();
  loadStats();
  const deepLink = new URLSearchParams(location.search).get("q");
  if (deepLink) {
    $("#query").value = deepLink;
    runSearch();
  }
  window.setInterval(loadHealth, 30000);
}

document.addEventListener("DOMContentLoaded", init);

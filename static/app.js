/* ==========================================================================
   Multimodal Q&A Pro — static/app.js
   Vanilla JS only, per locked stack (master PRD Section 5). No build step.

   IMPORTANT — Section 16 compliance:
   This file NEVER fabricates a reasoning trace and presents it as if it
   came from a real agent.stream() run. Where backend isn't
   reachable yet, functions fall back to clearly-labeled demo content and
   the "Demo Mode" badge in the header stays visible. The moment a real
   fetch()/EventSource call succeeds, demo mode turns off and only real
   server data is rendered.
   ========================================================================== */

const API_BASE = ""; // same-origin, per Section 7 — FastAPI serves these static files too
let backendReachable = false;
let sessionId = "session-" + Math.random().toString(36).slice(2, 10);
let currentHybridMode = "chat"; 
// Declared here (not next to their upload handlers further down) so
// updateHybridSourcesHint() can safely reference them the moment the page
// loads (setActiveTab("hybrid-chat") below runs before this file reaches the
// Document Q&A / Image Studio sections) — referencing a `let` before its own
// declaration throws, so these can't stay declared further down.
let uploadedDocCount = 0;
let uploadedImageCount = 0;

// ---------------------------------------------------------------------------
// Indexing lock — while a PDF is being chunked/embedded or an image is being
// uploaded, chat inputs must be disabled. Sending a question mid-index either
// silently returns nothing (document not in Chroma yet) or answers against a
// stale/incomplete session state — so this is a correctness fix, not just
// polish. A counter (not a boolean) supports multiple concurrent uploads:
// the lock only lifts once every in-flight upload has finished.
// ---------------------------------------------------------------------------
let pendingIndexingCount = 0;
const LOCKABLE_CHAT_INPUTS = [
  { input: "hybrid-chat-input", send: "hybrid-chat-send" },
  { input: "report-chat-input", send: "report-chat-send" },
  { input: "docqa-chat-input", send: "docqa-chat-send" },
  { input: "image-chat-input", send: "image-chat-send" },
];
const INDEXING_LOCK_PLACEHOLDER = "Indexing in progress — please wait…";
const _originalPlaceholders = {};

function updateIndexingLockUI() {
  const locked = pendingIndexingCount > 0;
  LOCKABLE_CHAT_INPUTS.forEach(({ input, send }) => {
    const inputEl = document.getElementById(input);
    const sendEl = document.getElementById(send);
    if (!inputEl || !sendEl) return;
    inputEl.disabled = locked;
    sendEl.disabled = locked;
    sendEl.classList.toggle("opacity-50", locked);
    sendEl.classList.toggle("cursor-not-allowed", locked);
    if (locked) {
      if (!(input in _originalPlaceholders)) {
        _originalPlaceholders[input] = inputEl.placeholder;
      }
      inputEl.placeholder = INDEXING_LOCK_PLACEHOLDER;
    } else if (input in _originalPlaceholders) {
      inputEl.placeholder = _originalPlaceholders[input];
    }
  });
}

function beginIndexing() {
  pendingIndexingCount += 1;
  updateIndexingLockUI();
}

function endIndexing() {
  pendingIndexingCount = Math.max(0, pendingIndexingCount - 1);
  updateIndexingLockUI();
}

// ---------------------------------------------------------------------------
// Onboarding modal — explains what each tab is for and that uploads are
// strictly per-session. Shown on every fresh page load (a refresh is a brand
// new session, per the cleanup logic below) unless the person has checked
// "Don't show this again", which is remembered across sessions via
// localStorage (a real persistent preference, not session-scoped data).
// ---------------------------------------------------------------------------
function initOnboarding() {
  const overlay = document.getElementById("onboarding-overlay");
  const dismissBtn = document.getElementById("onboarding-dismiss-btn");
  const dontShowCheckbox = document.getElementById("onboarding-dont-show");
  if (!overlay || !dismissBtn) return;

  let suppressed = false;
  try {
    suppressed = localStorage.getItem("mqa_onboarding_dismissed") === "1";
  } catch (e) {
    suppressed = false; // localStorage unavailable (e.g. private mode) — just show it
  }

  if (!suppressed) {
    overlay.classList.remove("hidden");
    overlay.classList.add("flex");
  }

  dismissBtn.addEventListener("click", () => {
    overlay.classList.add("hidden");
    overlay.classList.remove("flex");
    if (dontShowCheckbox && dontShowCheckbox.checked) {
      try {
        localStorage.setItem("mqa_onboarding_dismissed", "1");
      } catch (e) {
        /* ignore — non-fatal if storage isn't available */
      }
    }
  });
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) dismissBtn.click();
  });
}
initOnboarding();

// ---------------------------------------------------------------------------
// Session cleanup — a page refresh/close gets a brand-new sessionId (see the
// `let sessionId = ...` above, re-run on every load), so the OLD session's
// uploaded chunks/images need to actually be deleted server-side, not just
// left orphaned and unreachable. `pagehide` fires reliably on refresh, tab
// close, and navigation; fetch's `keepalive` flag lets the request survive
// the page unloading.
// ---------------------------------------------------------------------------
function endCurrentSession() {
  try {
    fetch(`${API_BASE}/api/session/${sessionId}`, { method: "DELETE", keepalive: true });
  } catch (e) {
    // best-effort only — nothing meaningful to do if this fails on unload
  }
}
window.addEventListener("pagehide", endCurrentSession);

// ---------------------------------------------------------------------------
// Lightweight toast — used for the "OCR takes a while" patience notice.
// Non-blocking (doesn't stop the user doing anything else), auto-dismisses,
// and is only ever shown once per page load so it doesn't nag on every
// OCR'd page of the same upload.
// ---------------------------------------------------------------------------
let _toastShownKeys = new Set();
function showToast(message, { key, durationMs = 9000 } = {}) {
  if (key) {
    if (_toastShownKeys.has(key)) return;
    _toastShownKeys.add(key);
  }
  let container = document.getElementById("toast-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "toast-container";
    container.style.cssText =
      "position:fixed;bottom:20px;right:20px;z-index:9999;display:flex;flex-direction:column;gap:8px;max-width:360px;";
    document.body.appendChild(container);
  }
  const toast = document.createElement("div");
  toast.className = "bg-surface-container border border-outline-variant/40 rounded-xl shadow-lg p-md text-body-md text-on-surface";
  toast.style.cssText = "animation: fadeInUp .2s ease-out;";
  toast.innerHTML = `
    <div class="flex items-start gap-sm">
      <span class="material-symbols-outlined text-cobalt text-[18px] mt-[1px]">hourglass_top</span>
      <p class="flex-1">${escapeHtml(message)}</p>
      <button aria-label="Dismiss" class="text-on-surface-variant hover:text-on-surface" style="line-height:1">&times;</button>
    </div>`;
  toast.querySelector("button").addEventListener("click", () => toast.remove());
  container.appendChild(toast);
  setTimeout(() => toast.remove(), durationMs);
}

// ---------------------------------------------------------------------------
// Enter-to-send / Ctrl+Enter-for-newline on every chat input in the app.
// Plain Enter submits the message (matches most chat UIs); Ctrl+Enter (or
// Cmd+Enter on Mac) inserts a newline instead of sending.
// ---------------------------------------------------------------------------
function wireEnterToSend(textareaId, sendButtonId) {
  const textarea = document.getElementById(textareaId);
  const sendButton = document.getElementById(sendButtonId);
  if (!textarea || !sendButton) return;
  textarea.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    if (e.ctrlKey || e.metaKey) {
      // Ctrl/Cmd+Enter -> newline. Textareas do this natively already, so
      // just let the default happen (no preventDefault).
      return;
    }
    // Plain Enter -> send, and don't also insert a newline into the box.
    e.preventDefault();
    if (pendingIndexingCount > 0) return; // input is locked while indexing
    sendButton.click();
  });
}

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------
const TAB_IDS = ["hybrid-chat", "document-qa", "image-studio"];

function setActiveTab(tabId) {
  TAB_IDS.forEach((id) => {
    const panel = document.getElementById("tab-" + id);
    const navBtn = document.querySelector(`[data-tab-target="${id}"]`);
    if (!panel || !navBtn) return;
    if (id === tabId) {
      panel.classList.add("active");
      navBtn.classList.add("text-primary", "border-primary");
      navBtn.classList.remove("text-on-surface-variant", "border-transparent");
    } else {
      panel.classList.remove("active");
      navBtn.classList.remove("text-primary", "border-primary");
      navBtn.classList.add("text-on-surface-variant", "border-transparent");
    }
  });
  if (typeof updateHybridSourcesHint === "function") updateHybridSourcesHint();
}

document.querySelectorAll(".nav-tab-link").forEach((btn) => {
  btn.addEventListener("click", () => setActiveTab(btn.dataset.tabTarget));
});
setActiveTab("hybrid-chat");

// ---------------------------------------------------------------------------
// Hybrid Chat sources hint — Hybrid Chat has no upload UI of its own, so this
// is the only feedback a user gets, inside that tab, about whether anything
// uploaded via Document Q&A / Image Studio is actually attached to their
// session (uploads carry over automatically — same shared `sessionId` — this
// just makes that visible instead of silent).
// ---------------------------------------------------------------------------
function updateHybridSourcesHint() {
  const statusEl = document.getElementById("hybrid-sources-status");
  if (!statusEl) return;
  const docCount = typeof uploadedDocCount !== "undefined" ? uploadedDocCount : 0;
  const imgCount = typeof uploadedImageCount !== "undefined" ? uploadedImageCount : 0;

  if (docCount === 0 && imgCount === 0) {
    statusEl.textContent = "No documents or images attached yet —";
    return;
  }
  const parts = [];
  if (docCount > 0) parts.push(`${docCount} document${docCount === 1 ? "" : "s"}`);
  if (imgCount > 0) parts.push(`${imgCount} image${imgCount === 1 ? "" : "s"}`);
  statusEl.textContent = `Attached: ${parts.join(" · ")} — also try`;
}

// ---------------------------------------------------------------------------
// Backend reachability check — determines whether the Demo Mode badge shows.
// POST /api/chat, /api/chat/report,
// /api/upload/pdf, /api/upload/image, GET /api/stream/{session_id}
// ---------------------------------------------------------------------------
async function checkBackend() {
  try {
    const res = await fetch(`${API_BASE}/api/chat`, { method: "OPTIONS" });
    backendReachable = res.ok || res.status === 405; // 405 still proves the route exists
  } catch (e) {
    backendReachable = false;
  }
  const badge = document.getElementById("demo-mode-badge");
  if (badge) badge.style.display = backendReachable ? "none" : "inline-flex";
}
checkBackend();

// ---------------------------------------------------------------------------
// Hybrid Chat — Chat Mode / Report Mode toggle
// ---------------------------------------------------------------------------
const chatModeBtn = document.getElementById("mode-chat-btn");
const reportModeBtn = document.getElementById("mode-report-btn");
const chatModeView = document.getElementById("chat-mode-view");
const reportModeView = document.getElementById("report-mode-view");

function setHybridMode(mode) {
  currentHybridMode = mode;
  const isChat = mode === "chat";
  chatModeView.classList.toggle("hidden", !isChat);
  reportModeView.classList.toggle("hidden", isChat);
  chatModeBtn.classList.toggle("bg-primary-container", isChat);
  chatModeBtn.classList.toggle("text-on-primary-container", isChat);
  chatModeBtn.classList.toggle("text-on-surface-variant", !isChat);
  reportModeBtn.classList.toggle("bg-primary-container", !isChat);
  reportModeBtn.classList.toggle("text-on-primary-container", !isChat);
  reportModeBtn.classList.toggle("text-on-surface-variant", isChat);
}
chatModeBtn.addEventListener("click", () => setHybridMode("chat"));
reportModeBtn.addEventListener("click", () => setHybridMode("report"));

function pillarPillClass(sourceType) {
  if (sourceType === "document") return "pill-document";
  if (sourceType === "web") return "pill-web";
  if (sourceType === "vision") return "pill-vision";
  return "pill-document";
}
function pillarIcon(sourceType) {
  if (sourceType === "document") return "description";
  if (sourceType === "web") return "public";
  if (sourceType === "vision") return "visibility";
  return "description";
}

function renderReport(data) {
  document.getElementById("report-title").textContent = data.title;
  document.getElementById("report-subtitle").textContent = data.subtitle;
  document.getElementById("report-conclusion").textContent = data.conclusion;

  const conflictBadge = document.getElementById("report-conflict-badge");
  conflictBadge.classList.toggle("hidden", data.conflicts.length === 0);
  conflictBadge.classList.toggle("flex", data.conflicts.length > 0);

  const findingsEl = document.getElementById("report-findings");
  findingsEl.innerHTML = "";
  data.findings.forEach((f) => {
    const row = document.createElement("div");
    row.className = "space-y-xs";
    row.innerHTML = `
      <p class="text-body-lg text-on-surface leading-relaxed">${f.claim}</p>
      <span class="inline-flex items-center gap-1 px-sm py-0.5 rounded-full text-label-caps uppercase ${pillarPillClass(f.source_type)}">
        <span class="material-symbols-outlined text-[12px]">${pillarIcon(f.source_type)}</span>
        ${f.source_type === "document" ? "Document" : "Web"}: ${f.source_detail}
      </span>`;
    findingsEl.appendChild(row);
  });

  // Conflicts — the section itself only renders when non-empty. Per Section
  // 16: "Do not let conflicts populate when sources actually agree" and
  // Dhanya's UI checklist: confirm empty conflicts renders cleanly with no
  // empty amber box.
  const conflictsSection = document.getElementById("report-conflicts-section");
  const conflictsEl = document.getElementById("report-conflicts");
  conflictsSection.classList.toggle("hidden", data.conflicts.length === 0);
  conflictsEl.innerHTML = "";
  data.conflicts.forEach((c) => {
    const block = document.createElement("div");
    block.className = "space-y-sm";
    block.innerHTML = `
      <p class="font-label-caps text-label-caps text-on-surface-variant uppercase">${c.topic}</p>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-gutter">
        <div class="space-y-xs p-sm bg-surface-container/50 border border-outline-variant/30 rounded-lg">
          <div class="flex items-center gap-xs text-cobalt text-label-caps uppercase"><span class="material-symbols-outlined text-[14px]">description</span> Document claim</div>
          <p class="text-body-md text-on-surface">${c.document_claim}</p>
        </div>
        <div class="space-y-xs p-sm bg-surface-container/50 border border-outline-variant/30 rounded-lg">
          <div class="flex items-center gap-xs text-teal text-label-caps uppercase"><span class="material-symbols-outlined text-[14px]">public</span> Web claim</div>
          <p class="text-body-md text-on-surface">${c.web_claim}</p>
        </div>
      </div>
      <div class="p-sm border-l-2 border-amber bg-amber/10 rounded-r-lg">
        <p class="font-code-inline text-code-inline text-amber"><span class="font-bold">Note:</span> ${c.note}</p>
      </div>`;
    conflictsEl.appendChild(block);
  });

  // Trace column — same Electric Cyan token as Chat Mode's accordion.
  const traceEl = document.getElementById("report-trace-steps");
  const steps = [
    { label: "Data Ingestion", text: "Retrieved relevant documents" + (data.findings.some(f => f.source_type === "web") ? " and web results." : ".") },
    { label: "Cross-Referencing", text: "Comparing claims across pillars for the same sub-topic." },
  ];
  if (data.conflicts.length > 0) steps.push({ label: "Conflict Detection", text: "Detected a numeric/claim mismatch between pillars.", isConflict: true });
  steps.push({ label: "Final Synthesis", text: "Compiling the Investigation Report." });

  traceEl.innerHTML = "";
  steps.forEach((s, i) => {
    const isLast = i === steps.length - 1;
    const node = document.createElement("div");
    node.className = "relative pl-gutter";
    node.innerHTML = `
      ${!isLast ? '<div class="absolute left-[7px] top-4 bottom-[-24px] trace-line"></div>' : ""}
      <div class="absolute left-0 top-0 w-4 h-4 rounded-full border-2 bg-background z-10" style="border-color:#22D3EE;"></div>
      <div class="space-y-xs">
        <span class="text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-tighter border ${s.isConflict ? 'text-amber border-amber/30 bg-amber/10' : 'text-cyan border-cyan/30 bg-cyan/10'}">${s.label}</span>
        <p class="text-body-md ${s.isConflict ? 'text-amber' : 'text-on-surface'}">${s.text}</p>
      </div>`;
    traceEl.appendChild(node);
  });
}

// Real call — populates the report the moment the backend responds.
async function fetchReport(message) {
  try {
    const res = await fetch(`${API_BASE}/api/chat/report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: message, session_id: sessionId })
    });
    if (!res.ok) throw new Error("report endpoint returned " + res.status);
    return await res.json();
  } catch (e) {
    console.warn("[demo mode] /api/chat/report unavailable:", e.message);
    return null; // caller shows a clear "backend unavailable" state instead
  }
}

//  /api/chat/report returns { session_id, report: { findings, conflicts,
// conclusion } } — no title/subtitle field (confirmed directly against her
// main.py + synthesis.py; it was never part of her schema). renderReport's UI
// needs those two fields, so we generate them client-side rather than asking
// her to add fields her synthesis layer has no natural source for.
function normalizeReport(real, query) {
  const r = real.report || {};
  return {
    title: `Investigation Report: ${query}`,
    subtitle: "Automated audit comparing available sources for this query.",
    findings: r.findings || [],
    conflicts: r.conflicts || [],
    conclusion: r.conclusion || ""
  };
}

// ---------------------------------------------------------------------------
// Hybrid Chat — send handler (Chat Mode)
// ---------------------------------------------------------------------------
document.getElementById("hybrid-chat-send").addEventListener("click", async () => {
  if (pendingIndexingCount > 0) return; // input is locked while indexing
  const input = document.getElementById("hybrid-chat-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";

  const chatEmptyState = document.getElementById("chat-empty-state");
  if (chatEmptyState) chatEmptyState.remove();

  appendUserBubble("chat-thread", text);

  // Single source of truth for Chat Mode: the SSE trace stream now drives
  // BOTH the reasoning trace UI and the final answer bubble. We no longer
  // also fire a second POST /api/chat here — that was triggering two full,
  // independent agent runs per message (2x Groq calls, 2x tool calls:
  // web search, vision, doc search), and since tool results like web
  // search are non-deterministic, the reasoning trace shown and the final
  // answer shown could in principle come from two different runs. The
  // last `ai_message` event received before `done` is used as the answer.
  beginLiveTrace("chat-thread", text, (finalAnswer, connected, alreadyStreamed) => {
    if (!connected) {
      console.warn("[demo mode] trace stream unavailable — no live SSE connection.");
      appendAiBubble("chat-thread", "(Demo mode — backend not connected yet. This is where the agent's real synthesized answer will appear.)");
      return;
    }
    // finalAnswer is null when the answer already streamed live token-by-
    // token into its own bubble (the normal case) — only append a fresh
    // bubble here if nothing streamed (e.g. a non-streaming model response)
    // or the run produced no content at all.
    if (!alreadyStreamed) {
      appendAiBubble("chat-thread", finalAnswer || "(The agent stream ended without a final message.)");
    }
  });
});

// ---------------------------------------------------------------------------
// Report Mode — send handler (dedicated input, separate from Chat Mode)
// ---------------------------------------------------------------------------
document.getElementById("report-chat-send").addEventListener("click", async () => {
  if (pendingIndexingCount > 0) return; // input is locked while indexing
  const input = document.getElementById("report-chat-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";

  document.getElementById("report-empty-state").classList.add("hidden");
  document.getElementById("report-body").classList.remove("hidden");

  const real = await fetchReport(text);
  if (real) {
    renderReport(normalizeReport(real, text));
  } else {
    renderReportUnavailable(text);
  }
});

// Shown when /api/chat/report can't be reached — an honest "no report" state
// rather than fabricated findings (keeps with Section 16: never fake a trace
// or a result that didn't really come back from the backend).
function renderReportUnavailable(query) {
  document.getElementById("report-title").textContent = `Investigation Report: ${query}`;
  document.getElementById("report-subtitle").textContent = "Backend not reachable — no report could be generated for this query.";
  document.getElementById("report-conflict-badge").classList.add("hidden");
  document.getElementById("report-findings").innerHTML =
    '<p class="text-on-surface-variant">(Demo mode — connect the backend to generate a real, cited report here.)</p>';
  document.getElementById("report-conflicts-section").classList.add("hidden");
  document.getElementById("report-conclusion").textContent = "";
  document.getElementById("report-trace-steps").innerHTML = "";
}

function appendUserBubble(containerId, text) {
  const el = document.createElement("div");
  el.className = "flex flex-col items-end gap-xs max-w-[85%] ml-auto";
  el.innerHTML = `<div class="bg-primary text-on-primary p-md rounded-2xl rounded-tr-none shadow-lg"><p class="text-body-md">${escapeHtml(text)}</p></div>`;
  document.getElementById(containerId).appendChild(el);
  el.scrollIntoView({ behavior: "smooth" });
}
function appendAiBubble(containerId, text) {
  const el = document.createElement("div");
  el.className = "flex flex-col items-start max-w-[95%]";
  el.innerHTML = `<div class="bg-surface-container rounded-2xl rounded-tl-none p-gutter border border-outline-variant/30"><div class="text-body-md text-on-surface">${renderMarkdown(text)}</div></div>`;
  document.getElementById(containerId).appendChild(el);
  el.scrollIntoView({ behavior: "smooth" });
}
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// Minimal, safe Markdown -> HTML for AI answer text. Escapes first (so raw
// HTML/script in a model response can never inject), then layers on just
// the handful of Markdown constructs these answers actually use: **bold**,
// "- " bullet lists, and blank-line-separated paragraphs. Not a full
// Markdown parser on purpose -- this only needs to cover what the agent's
// synthesis layer actually produces.
function renderMarkdown(str) {
  const escaped = escapeHtml(str || "");
  const withBold = escaped.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  const lines = withBold.split(/\n/);
  let html = "";
  let inList = false;
  const closeList = () => {
    if (inList) {
      html += "</ul>";
      inList = false;
    }
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (/^-\s+/.test(line)) {
      if (!inList) {
        html += '<ul class="list-disc pl-lg space-y-1">';
        inList = true;
      }
      html += `<li>${line.replace(/^-\s+/, "")}</li>`;
    } else if (line === "") {
      closeList();
    } else {
      closeList();
      html += `<p class="mb-2 last:mb-0">${line}</p>`;
    }
  }
  closeList();
  return html;
}

// ---------------------------------------------------------------------------
// Document Q&A — upload flow (empty -> uploading -> populated)
// ---------------------------------------------------------------------------
const pdfDropzone = document.getElementById("pdf-dropzone");
const pdfFileInput = document.getElementById("pdf-file-input");
const docList = document.getElementById("doc-list");
const docEmptyState = document.getElementById("doc-empty-state");
const docCountLabel = document.getElementById("doc-count-label");

pdfDropzone.addEventListener("click", () => pdfFileInput.click());
["dragover", "dragenter"].forEach((evt) =>
  pdfDropzone.addEventListener(evt, (e) => { e.preventDefault(); pdfDropzone.classList.add("drag-over"); })
);
["dragleave", "drop"].forEach((evt) =>
  pdfDropzone.addEventListener(evt, (e) => { e.preventDefault(); pdfDropzone.classList.remove("drag-over"); })
);
pdfDropzone.addEventListener("drop", (e) => {
  Array.from(e.dataTransfer.files || []).forEach((file) => handlePdfUpload(file));
});
pdfFileInput.addEventListener("change", (e) => {
  Array.from(e.target.files || []).forEach((file) => handlePdfUpload(file));
  pdfFileInput.value = ""; // allow re-selecting the same file(s) later
});

function handlePdfUpload(file) {
  docEmptyState.style.display = "none";
  const card = document.createElement("div");
  card.className = "bg-surface-container p-md rounded-xl border border-outline-variant";
  card.innerHTML = `
    <div class="flex justify-between items-start mb-xs">
      <span class="material-symbols-outlined text-cobalt">description</span>
      <span class="px-xs py-[2px] bg-cobalt/10 border border-cobalt/30 text-cobalt rounded text-[10px] font-bold uppercase" data-status>Indexing…</span>
    </div>
    <h4 class="text-body-md font-bold text-on-surface truncate">${escapeHtml(file.name)}</h4>
    <div class="w-full h-1.5 bg-surface-container-highest rounded-full overflow-hidden mt-sm">
      <div class="h-full bg-cobalt transition-[width] duration-300 ease-out" data-progress-bar style="width:4%"></div>
    </div>
    <p class="mt-xs text-on-surface-variant font-code-inline text-[11px]" data-progress-text>Starting…</p>`;
  docList.appendChild(card);
  document.getElementById("docqa-indexing-status").textContent = "INDEXING: ACTIVE";

  uploadPdf(file, card);
}

// ---------------------------------------------------------------------------
// Real-time indexing progress — GET /api/upload/progress/{upload_id} (SSE),
// driven by the actual per-page progress_cb snapshots from rag/ingest.py's
// worker thread. Not a timed/simulated bar: if the backend goes silent, the
// bar simply stops moving, which is an honest reflection of reality.
// ---------------------------------------------------------------------------
function connectUploadProgress(uploadId, card) {
  const barEl = card.querySelector("[data-progress-bar]");
  const textEl = card.querySelector("[data-progress-text]");
  if (!barEl || !textEl) return null;

  let sawOcr = false;
  let es;
  try {
    es = new EventSource(`${API_BASE}/api/upload/progress/${uploadId}`);
  } catch (e) {
    return null; // EventSource unavailable — the bar just stays at its start state
  }

  es.onmessage = (event) => {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch (e) {
      return;
    }
    const { stage, current, total, detail, done, error } = data;

    if (!sawOcr && (stage === "ocr_start" || stage === "ocr_done")) {
      sawOcr = true;
      showToast(
        "This PDF has scanned/slide pages that need OCR — that goes through a rate-limited " +
          "vision API, so a multi-page file can take a couple of minutes. Hang tight, it will finish.",
        { key: "ocr-patience" }
      );
    }

    // Rough overall progress: extraction/OCR is the dominant cost, so give
    // it most of the bar (5–85%), then chunking/embedding fill the rest.
    let pct = 4;
    if (total > 0) pct = 5 + Math.round((current / total) * 80);
    if (stage === "chunking") pct = 88;
    if (stage === "embedding") pct = 94;
    if (stage === "done") pct = 100;
    if (stage === "error") pct = 100;
    barEl.style.width = `${Math.min(100, Math.max(4, pct))}%`;
    barEl.classList.toggle("bg-red-500", stage === "error");
    barEl.classList.toggle("bg-cobalt", stage !== "error");

    if (stage === "ocr_start") {
      textEl.textContent = `OCR: ${detail || `page ${current}/${total}`} — this page needs vision OCR, may take a bit`;
    } else if (stage === "text") {
      textEl.textContent = `Reading ${detail || `page ${current}/${total}`}`;
    } else if (stage === "chunking") {
      textEl.textContent = "Splitting extracted text into chunks…";
    } else if (stage === "embedding") {
      textEl.textContent = detail || "Embedding chunks…";
    } else if (stage === "error") {
      textEl.textContent = `Failed: ${detail || error || "unknown error"}`;
    } else if (stage === "done") {
      textEl.textContent = "Finalizing…";
    } else if (detail) {
      textEl.textContent = detail;
    }

    if (done) es.close();
  };
  es.onerror = () => {
    es.close(); // connection issue — the final REST response still finishes the upload either way
  };
  return es;
}

async function uploadPdf(file, card) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("session_id", sessionId); 
  const uploadId =
    (crypto.randomUUID && crypto.randomUUID()) || `upload-${Math.random().toString(36).slice(2)}`;
  formData.append("upload_id", uploadId);
  beginIndexing();
  const progressStream = connectUploadProgress(uploadId, card);
  try {
    const res = await fetch(`${API_BASE}/api/upload/pdf`, { method: "POST", body: formData });
    if (!res.ok) throw new Error(String(res.status));
    const data = await res.json();
    const ocrNote = data.ocr_page_count > 0
      ? ` (${data.ocr_page_count} page${data.ocr_page_count === 1 ? "" : "s"} read via OCR)`
      : "";
    markDocIndexed(card, `${data.chunks_ingested ?? "?"} chunks indexed${ocrNote}`); // key is chunks_ingested, not chunk_count
    endIndexing();
  } catch (e) {
    console.warn("[demo mode] /api/upload/pdf unavailable, simulating indexing locally:", e.message);
    setTimeout(() => {
      markDocIndexed(card, "Indexed (demo — not yet sent to backend)");
      endIndexing();
    }, 1200);
  } finally {
    if (progressStream) progressStream.close();
  }
}
function markDocIndexed(card, statusText) {
  const statusEl = card.querySelector("[data-status]");
  const progressText = card.querySelector("[data-progress-text]");
  const progressBar = card.querySelector("[data-progress-bar]");
  statusEl.textContent = "✓ Indexed";
  progressText.textContent = statusText;
  if (progressBar) {
    progressBar.style.width = "100%";
    progressBar.classList.remove("bg-red-500");
    progressBar.classList.add("bg-cobalt");
  }
  uploadedDocCount += 1;
  docCountLabel.textContent = `(${uploadedDocCount} file${uploadedDocCount === 1 ? "" : "s"})`;
  document.getElementById("docqa-indexing-status").textContent = "INDEXING: INACTIVE";
  document.getElementById("docqa-ready-state").classList.add("hidden");
  document.getElementById("docqa-chat-thread").classList.remove("hidden");
  updateHybridSourcesHint();
}

document.getElementById("docqa-chat-send").addEventListener("click", async () => {
  if (pendingIndexingCount > 0) return; // input is locked while indexing
  const input = document.getElementById("docqa-chat-input");
  const text = input.value.trim();
  if (!text) return;
  document.getElementById("docqa-ready-state").classList.add("hidden");
  document.getElementById("docqa-chat-thread").classList.remove("hidden");
  appendUserBubble("docqa-chat-thread", text);
  input.value = "";

  beginLiveTrace(
    "docqa-chat-thread",
    text,
    (finalAnswer, connected, alreadyStreamed) => {
      if (!connected) {
        console.warn("[demo mode] /api/stream unavailable:", "no live SSE connection.");
        appendAiBubble("docqa-chat-thread", "(Demo mode — upload a PDF and connect the backend to get real, cited answers here.)");
        return;
      }
      if (!alreadyStreamed) {
        appendAiBubble("docqa-chat-thread", finalAnswer || "(The agent stream ended without a final message.)");
      }
    },
    { scope: "documents_only" }
  );
});

// ---------------------------------------------------------------------------
// Image Studio — upload flow
// ---------------------------------------------------------------------------
const imageEmptyState = document.getElementById("image-empty-state");
const imagePopulatedState = document.getElementById("image-populated-state");
const imageFileInput = document.getElementById("image-file-input");
const imageGallery = document.getElementById("image-gallery");
const imageCountLabel = document.getElementById("image-count-label");
const imageAddMoreBtn = document.getElementById("image-add-more-btn");

imageEmptyState.addEventListener("click", () => imageFileInput.click());
if (imageAddMoreBtn) imageAddMoreBtn.addEventListener("click", () => imageFileInput.click());
imageFileInput.addEventListener("change", (e) => {
  Array.from(e.target.files || []).forEach((file) => handleImageUpload(file));
  imageFileInput.value = ""; // allow re-selecting the same file(s) later
});

function handleImageUpload(file) {
  const url = URL.createObjectURL(file);
  imageEmptyState.classList.add("hidden");
  imagePopulatedState.classList.remove("hidden");
  imagePopulatedState.classList.add("flex");

  const thumb = document.createElement("div");
  thumb.className = "image-gallery-thumb";
  thumb.innerHTML = `
    <img src="${url}" alt="${escapeHtml(file.name)}">
    <span class="thumb-status">Uploading…</span>`;
  imageGallery.appendChild(thumb);

  uploadImage(file, thumb);
}

async function uploadImage(file, thumb) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("session_id", sessionId); 
  const statusEl = thumb.querySelector(".thumb-status");
  beginIndexing();
  try {
    const res = await fetch(`${API_BASE}/api/upload/image`, { method: "POST", body: formData });
    if (!res.ok) throw new Error(String(res.status));
    const data = await res.json();
    // main.py now stores every image uploaded in the session (not just the
    // most recent one), so image_count reflects the real running total.
    uploadedImageCount = data.image_count ?? uploadedImageCount + 1;
    if (statusEl) statusEl.remove();
    updateImageCountLabel();
    updateHybridSourcesHint();
    endIndexing();
  } catch (e) {
    console.warn("[demo mode] /api/upload/image unavailable:", e.message);
    if (statusEl) statusEl.textContent = "Demo mode";
    uploadedImageCount += 1;
    updateImageCountLabel();
    updateHybridSourcesHint();
    endIndexing();
  }
}

function updateImageCountLabel() {
  if (!imageCountLabel) return;
  imageCountLabel.textContent = `${uploadedImageCount} image${uploadedImageCount === 1 ? "" : "s"} attached`;
}

document.getElementById("image-chat-send").addEventListener("click", async () => {
  if (pendingIndexingCount > 0) return; // input is locked while indexing
  const input = document.getElementById("image-chat-input");
  const text = input.value.trim();
  if (!text) return;
  const crossReference = document.getElementById("cross-reference-toggle").checked;
  appendUserBubble("image-chat-thread", text);
  input.value = "";

  // Real reasoning trace + live-typed answer now (previously a static,
  // hand-built "describe_image / search_documents" markup that didn't
  // reflect what the agent actually did, backed by a separate blocking
  // /api/chat POST). beginLiveTrace drives both from the one real run.
  beginLiveTrace(
    "image-chat-thread",
    text,
    (finalAnswer, connected, alreadyStreamed) => {
      if (!connected) {
        console.warn("[demo mode] /api/stream unavailable:", "no live SSE connection.");
        appendAiBubble("image-chat-thread", "(Demo mode — the real vision + cross-reference answer will render here once the backend is connected.)");
        return;
      }
      if (!alreadyStreamed) {
        appendAiBubble("image-chat-thread", finalAnswer || "(The agent stream ended without a final message.)");
      }
    },
    { crossReferenceDocuments: crossReference }
  );
});

// ---------------------------------------------------------------------------
// Reasoning trace SSE client — GET /api/stream/{session_id} (Section 6.7 / 7)
// Real EventSource only. If it can't connect, we do NOT fabricate a fake
// live trace (Section 16) — we simply leave the static demo trace markup
// already in the page (clearly a mockup, per the Demo Mode badge) alone.
// ---------------------------------------------------------------------------
function connectTraceStream(query, onEvent, onError, extraParams = {}) {
  try {
    
    // required query string param (
    // omitting it 422s.
    const params = new URLSearchParams({ query, ...extraParams });
    const url = `${API_BASE}/api/stream/${sessionId}?${params.toString()}`;
    const es = new EventSource(url);
    let gotAnyMessage = false;
    es.onmessage = (event) => {
      gotAnyMessage = true;
      try {
        onEvent(JSON.parse(event.data));
      } catch (err) {
        console.error("Malformed trace event:", err);
      }
    };
    es.onerror = () => {
      console.warn("[demo mode] trace stream unavailable — no live SSE connection.");
      es.close();
      // Only treat this as a connection failure (and trigger the demo-mode
      // fallback) if we never received a single event. If the stream had
      // already delivered events and then errored/closed normally, the
      // 'done' handler in beginLiveTrace already resolved the answer.
      if (!gotAnyMessage && onError) onError();
    };
    return es;
  } catch (e) {
    console.warn("[demo mode] EventSource not available:", e.message);
    if (onError) onError();
    return null;
  }
}

// Builds the same "REASONING TRACE (N STEPS)" accordion used in the static
// Chat Mode mockup, but drives it from real connectTraceStream() events
// instead of hardcoded copy. Per Section 16, if the stream never connects
// this accordion simply never appears — it does not fall back to a fake
// trace.
// onFinalAnswer(answerText, connected) is called exactly once, either when
// the stream sends 'done' (connected=true, answerText = last ai_message
// content seen) or when the stream never connects at all (connected=false,
// so the caller can show the demo-mode fallback bubble instead).
function beginLiveTrace(threadId, query, onFinalAnswer, options = {}) {
  const thread = document.getElementById(threadId);
  const wrapper = document.createElement("div");
  wrapper.className = "flex flex-col items-start gap-sm max-w-[95%] mb-sm";
  wrapper.innerHTML = `
    <div class="trace-container">
      <button class="flex items-center gap-sm text-[12px] trace-label hover:opacity-100 transition-colors" onclick="this.nextElementSibling.classList.toggle('hidden'); this.querySelector('.arrow').classList.toggle('rotate-180')">
        <span class="font-code-inline font-medium" data-trace-count>REASONING TRACE (0 STEPS)</span>
        <span class="material-symbols-outlined text-[16px] arrow transition-transform">expand_more</span>
      </button>
      <div class="hidden flex flex-col gap-sm py-xs mt-xs" data-trace-steps></div>
    </div>`;
  thread.appendChild(wrapper);

  const stepsEl = wrapper.querySelector("[data-trace-steps]");
  const countEl = wrapper.querySelector("[data-trace-count]");
  let stepCount = 0;
  let lastAiMessage = null;
  let resolved = false;

  // Live-typing answer bubble — created lazily on the first real token, so
  // nothing empty flashes on screen if the run is pure tool-calling with no
  // streamed text yet. Text is appended token-by-token as real SSE "token"
  // events arrive (see agent/graph.py's stream_mode=["updates","messages"]
  // and main.py's _serialize_stream_item) — never a client-side simulated
  // typing animation over an already-complete answer.
  let answerBubbleEl = null;
  let answerTextEl = null;
  let streamedText = "";
  let lastError = null;

  // Translates the raw exception string the backend forwards (main.py's
  // `except Exception as exc: q.put(("error", str(exc)))`) into something
  // a user should actually see, instead of the generic "stream ended
  // without a final message" fallback. Rate limiting on the LLM provider
  // is the one case worth calling out by name, since "try again shortly"
  // is genuinely actionable advice here -- everything else gets a generic
  // but still honest message rather than a raw stack-trace-flavored string.
  function friendlyErrorMessage(rawError) {
    const msg = String(rawError || "").toLowerCase();
    if (msg.includes("429") || msg.includes("rate_limit") || msg.includes("rate limit")) {
      return "We're being rate-limited by the AI provider right now. Please wait a few seconds and try again.";
    }
    return "Something went wrong while generating a response. Please try again in a moment.";
  }

  function ensureAnswerBubble() {
    if (answerBubbleEl) return;
    answerBubbleEl = document.createElement("div");
    answerBubbleEl.className = "flex flex-col items-start max-w-[95%]";
    answerBubbleEl.innerHTML = `<div class="bg-surface-container rounded-2xl rounded-tl-none p-gutter border border-outline-variant/30"><p class="text-body-md text-on-surface"><span data-answer-text></span><span class="typing-cursor" data-typing-cursor>▍</span></p></div>`;
    thread.appendChild(answerBubbleEl);
    answerTextEl = answerBubbleEl.querySelector("[data-answer-text]");
    answerBubbleEl.scrollIntoView({ behavior: "smooth", block: "end" });
  }

  function appendToken(text) {
    ensureAnswerBubble();
    streamedText += text;
    answerTextEl.textContent = streamedText;
  }

  function finishTyping() {
    if (answerBubbleEl) {
      const cursor = answerBubbleEl.querySelector("[data-typing-cursor]");
      if (cursor) cursor.remove();
    }
  }

  // Tokens stream in as plain text (appendToken above) so a Markdown marker
  // split across two token chunks (e.g. "**" arriving as "*" then "*")
  // never renders half-converted mid-stream. Once the run is done and the
  // full text is known, re-render it once through renderMarkdown so bold/
  // lists show up correctly in the final, settled answer.
  function finalizeAnswerBubble(fullText) {
    if (answerTextEl && fullText) {
      answerTextEl.innerHTML = renderMarkdown(fullText);
    }
  }

  function addStep(text) {
    stepCount += 1;
    countEl.textContent = `REASONING TRACE (${stepCount} STEP${stepCount === 1 ? "" : "S"})`;
    const row = document.createElement("div");
    row.className = "flex items-start gap-sm";
    row.innerHTML = `
      <div class="w-1.5 h-1.5 rounded-full trace-node-dot mt-1.5"></div>
      <p class="text-body-md text-on-surface-variant">${escapeHtml(text)}</p>`;
    stepsEl.appendChild(row);
  }

  function resolveOnce(answer, connected) {
    if (resolved) return;
    resolved = true;
    finishTyping();
    const alreadyStreamed = Boolean(streamedText);
    if (onFinalAnswer) onFinalAnswer(alreadyStreamed ? null : answer, connected, alreadyStreamed);
  }

  const extraParams = {};
  if (options.scope) extraParams.scope = options.scope;
  if (options.crossReferenceDocuments !== undefined && options.crossReferenceDocuments !== null) {
    extraParams.cross_reference_documents = String(options.crossReferenceDocuments);
  }

  const es = connectTraceStream(
    query,
    (traceEvent) => {
      if (traceEvent.type === "tool_call") {
        addStep(`Calling ${traceEvent.tool}…`);
      } else if (traceEvent.type === "tool_result") {
        addStep(`${traceEvent.tool} returned results.`);
      } else if (traceEvent.type === "token") {
        appendToken(traceEvent.content);
      } else if (traceEvent.type === "ai_message" && traceEvent.content) {
        // Still tracked as a trace step (keeps the accordion's reasoning
        // narrative intact) and as a fallback final answer for runs where,
        // for whatever reason, no token events came through.
        addStep(traceEvent.content);
        lastAiMessage = traceEvent.content;
      } else if (traceEvent.type === "error") {
        lastError = traceEvent.content;
        console.warn("[agent error]", traceEvent.content);
        addStep(`⚠ ${friendlyErrorMessage(traceEvent.content)}`);
      } else if (traceEvent.type === "done") {
        if (es) es.close();
        const completedAnswer = traceEvent.content || lastAiMessage || null;
        const finalText = completedAnswer || streamedText || null;
        if (finalText) {
          finalizeAnswerBubble(finalText);
          if (!completedAnswer && lastError) {
            // Text streamed in, but the run errored before a proper final
            // answer arrived (e.g. rate-limited mid-response) -- flag that
            // explicitly rather than silently presenting a cut-off answer
            // as if it were complete.
            const note = document.createElement("p");
            note.className = "text-body-sm text-on-surface-variant italic mt-xs px-gutter";
            note.textContent = `⚠ ${friendlyErrorMessage(lastError)}`;
            answerBubbleEl.appendChild(note);
          }
          resolveOnce(finalText, true);
        } else {
          resolveOnce(lastError ? friendlyErrorMessage(lastError) : null, true);
        }
      }
    },
    () => resolveOnce(null, false), // connection never established
    extraParams
  );

  return wrapper;
}
// connectTraceStream((traceEvent) => { /* render real tool-call events here */ });

// Wire Enter-to-send / Ctrl+Enter-newline for every chat input in the app.
wireEnterToSend("hybrid-chat-input", "hybrid-chat-send");
wireEnterToSend("report-chat-input", "report-chat-send");
wireEnterToSend("docqa-chat-input", "docqa-chat-send");
wireEnterToSend("image-chat-input", "image-chat-send");
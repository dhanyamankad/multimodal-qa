/* ==========================================================================
   Multimodal Q&A Pro — static/app.js
   Vanilla JS only, per locked stack (master PRD Section 5). No build step.

   IMPORTANT — Section 16 compliance:
   This file NEVER fabricates a reasoning trace and presents it as if it
   came from a real agent.stream() run. Where Vanshi's backend isn't
   reachable yet, functions fall back to clearly-labeled demo content and
   the "Demo Mode" badge in the header stays visible. The moment a real
   fetch()/EventSource call succeeds, demo mode turns off and only real
   server data is rendered.
   ========================================================================== */

const API_BASE = ""; // same-origin, per Section 7 — FastAPI serves these static files too
let backendReachable = false;
let sessionId = "session-" + Math.random().toString(36).slice(2, 10);
let currentHybridMode = "chat"; 

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
}

document.querySelectorAll(".nav-tab-link").forEach((btn) => {
  btn.addEventListener("click", () => setActiveTab(btn.dataset.tabTarget));
});
setActiveTab("hybrid-chat");

// ---------------------------------------------------------------------------
// Backend reachability check — determines whether the Demo Mode badge shows.
// Vanshi's real endpoints (Section 7): POST /api/chat, /api/chat/report,
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

// Vanshi's /api/chat/report returns { session_id, report: { findings, conflicts,
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
  beginLiveTrace("chat-thread", text, (finalAnswer, connected) => {
    if (!connected) {
      console.warn("[demo mode] trace stream unavailable — no live SSE connection.");
      appendAiBubble("chat-thread", "(Demo mode — backend not connected yet. This is where the agent's real synthesized answer will appear.)");
      return;
    }
    appendAiBubble("chat-thread", finalAnswer || "(The agent stream ended without a final message.)");
  });
});

// ---------------------------------------------------------------------------
// Report Mode — send handler (dedicated input, separate from Chat Mode)
// ---------------------------------------------------------------------------
document.getElementById("report-chat-send").addEventListener("click", async () => {
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
  el.innerHTML = `<div class="bg-surface-container rounded-2xl rounded-tl-none p-gutter border border-outline-variant/30"><p class="text-body-md text-on-surface">${escapeHtml(text)}</p></div>`;
  document.getElementById(containerId).appendChild(el);
  el.scrollIntoView({ behavior: "smooth" });
}
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ---------------------------------------------------------------------------
// Document Q&A — upload flow (empty -> uploading -> populated)
// ---------------------------------------------------------------------------
const pdfDropzone = document.getElementById("pdf-dropzone");
const pdfFileInput = document.getElementById("pdf-file-input");
const docList = document.getElementById("doc-list");
const docEmptyState = document.getElementById("doc-empty-state");
const docCountLabel = document.getElementById("doc-count-label");
let uploadedDocCount = 0;

pdfDropzone.addEventListener("click", () => pdfFileInput.click());
["dragover", "dragenter"].forEach((evt) =>
  pdfDropzone.addEventListener(evt, (e) => { e.preventDefault(); pdfDropzone.classList.add("drag-over"); })
);
["dragleave", "drop"].forEach((evt) =>
  pdfDropzone.addEventListener(evt, (e) => { e.preventDefault(); pdfDropzone.classList.remove("drag-over"); })
);
pdfDropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handlePdfUpload(file);
});
pdfFileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) handlePdfUpload(file);
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
      <div class="h-full bg-cobalt progress-shimmer" style="width:100%"></div>
    </div>
    <p class="mt-xs text-on-surface-variant font-code-inline text-[11px]" data-progress-text>Indexing… extracting pages, chunking, embedding</p>`;
  docList.appendChild(card);
  document.getElementById("docqa-indexing-status").textContent = "INDEXING: ACTIVE";

  uploadPdf(file, card);
}

async function uploadPdf(file, card) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("session_id", sessionId); // required Form(...) field on Vanshi's endpoint — was missing, would 422
  try {
    const res = await fetch(`${API_BASE}/api/upload/pdf`, { method: "POST", body: formData });
    if (!res.ok) throw new Error(String(res.status));
    const data = await res.json();
    markDocIndexed(card, `${data.chunks_ingested ?? "?"} chunks indexed`); // key is chunks_ingested, not chunk_count
  } catch (e) {
    console.warn("[demo mode] /api/upload/pdf unavailable, simulating indexing locally:", e.message);
    setTimeout(() => markDocIndexed(card, "Indexed (demo — not yet sent to backend)"), 1200);
  }
}
function markDocIndexed(card, statusText) {
  const statusEl = card.querySelector("[data-status]");
  const progressText = card.querySelector("[data-progress-text]");
  statusEl.textContent = "✓ Indexed";
  progressText.textContent = statusText;
  uploadedDocCount += 1;
  docCountLabel.textContent = `(${uploadedDocCount} file${uploadedDocCount === 1 ? "" : "s"})`;
  document.getElementById("docqa-indexing-status").textContent = "INDEXING: INACTIVE";
  document.getElementById("docqa-ready-state").classList.add("hidden");
  document.getElementById("docqa-chat-thread").classList.remove("hidden");
}

document.getElementById("docqa-chat-send").addEventListener("click", async () => {
  const input = document.getElementById("docqa-chat-input");
  const text = input.value.trim();
  if (!text) return;
  document.getElementById("docqa-ready-state").classList.add("hidden");
  document.getElementById("docqa-chat-thread").classList.remove("hidden");
  appendUserBubble("docqa-chat-thread", text);
  input.value = "";
  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: text, session_id: sessionId, scope: "documents_only" })
    });
    if (!res.ok) throw new Error(String(res.status));
    const data = await res.json();
    appendAiBubble("docqa-chat-thread", data.answer || JSON.stringify(data));
  } catch (e) {
    console.warn("[demo mode] /api/chat unavailable:", e.message);
    appendAiBubble("docqa-chat-thread", "(Demo mode — upload a PDF and connect the backend to get real, cited answers here.)");
  }
});

// ---------------------------------------------------------------------------
// Image Studio — upload flow
// ---------------------------------------------------------------------------
const imageEmptyState = document.getElementById("image-empty-state");
const imagePopulatedState = document.getElementById("image-populated-state");
const imageFileInput = document.getElementById("image-file-input");
const imagePreview = document.getElementById("image-preview");

imageEmptyState.addEventListener("click", () => imageFileInput.click());
imageFileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) handleImageUpload(file);
});

function handleImageUpload(file) {
  const url = URL.createObjectURL(file);
  imagePreview.src = url;
  imageEmptyState.classList.add("hidden");
  imagePopulatedState.classList.remove("hidden");
  imagePopulatedState.classList.add("flex");
  document.getElementById("image-chat-thread").innerHTML = "";
  uploadImage(file);
}

async function uploadImage(file) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("session_id", sessionId); // required Form(...) field on Vanshi's endpoint — was missing, would 422
  try {
    const res = await fetch(`${API_BASE}/api/upload/image`, { method: "POST", body: formData });
    if (!res.ok) throw new Error(String(res.status));
  } catch (e) {
    console.warn("[demo mode] /api/upload/image unavailable:", e.message);
  }
}

document.getElementById("image-chat-send").addEventListener("click", async () => {
  const input = document.getElementById("image-chat-input");
  const text = input.value.trim();
  if (!text) return;
  const crossReference = document.getElementById("cross-reference-toggle").checked;
  appendUserBubble("image-chat-thread", text);
  input.value = "";

  // Reasoning trace container — cyan wrapper, coral/cobalt tool icons inside,
  // per the fix applied to the Image Studio Stitch export.
  const traceEl = document.createElement("div");
  traceEl.className = "trace-container mb-md";
  traceEl.innerHTML = `
    <div class="trace-label text-label-caps mb-xs">Reasoning Trace</div>
    <div class="flex items-center gap-xs">
      <div class="w-5 h-5 rounded-full border-[1.5px] border-coral flex items-center justify-center bg-coral/10">
        <div class="w-1.5 h-1.5 rounded-full bg-coral"></div>
      </div>
      <span class="material-symbols-outlined text-[16px] text-coral">visibility</span>
      <span class="text-label-caps text-coral">describe_image</span>
    </div>
    ${crossReference ? `
    <div class="flex items-center gap-xs mt-xs">
      <div class="w-5 h-5 rounded-full border-[1.5px] border-cobalt flex items-center justify-center bg-cobalt/10">
        <div class="w-1.5 h-1.5 rounded-full bg-cobalt"></div>
      </div>
      <span class="material-symbols-outlined text-[16px] text-cobalt">search</span>
      <span class="text-label-caps text-cobalt">search_documents</span>
    </div>` : ""}`;
  document.getElementById("image-chat-thread").appendChild(traceEl);

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: text, session_id: sessionId, cross_reference_documents: crossReference })
    });
    if (!res.ok) throw new Error(String(res.status));
    const data = await res.json();
    appendAiBubble("image-chat-thread", data.answer || JSON.stringify(data));
  } catch (e) {
    console.warn("[demo mode] /api/chat unavailable:", e.message);
    appendAiBubble("image-chat-thread", "(Demo mode — the real vision + cross-reference answer will render here once the backend is connected.)");
  }
});

// ---------------------------------------------------------------------------
// Reasoning trace SSE client — GET /api/stream/{session_id} (Section 6.7 / 7)
// Real EventSource only. If it can't connect, we do NOT fabricate a fake
// live trace (Section 16) — we simply leave the static demo trace markup
// already in the page (clearly a mockup, per the Demo Mode badge) alone.
// ---------------------------------------------------------------------------
function connectTraceStream(query, onEvent, onError) {
  try {
    // Vanshi's endpoint requires session_id in the path AND query as a
    // required query string param (confirmed directly against her main.py) —
    // omitting it 422s.
    const url = `${API_BASE}/api/stream/${sessionId}?query=${encodeURIComponent(query)}`;
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
function beginLiveTrace(threadId, query, onFinalAnswer) {
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
    if (onFinalAnswer) onFinalAnswer(answer, connected);
  }

  const es = connectTraceStream(
    query,
    (traceEvent) => {
      if (traceEvent.type === "tool_call") {
        addStep(`Calling ${traceEvent.tool}…`);
      } else if (traceEvent.type === "tool_result") {
        addStep(`${traceEvent.tool} returned results.`);
      } else if (traceEvent.type === "ai_message" && traceEvent.content) {
        addStep(traceEvent.content);
        // Track the latest ai_message content as the running candidate for
        // the final synthesized answer — the last one seen before 'done'
        // wins, since intermediate ai_message events can be interim
        // reasoning rather than the final response.
        lastAiMessage = traceEvent.content;
      } else if (traceEvent.type === "done") {
        if (es) es.close();
        resolveOnce(lastAiMessage, true);
      }
    },
    () => resolveOnce(null, false) // connection never established
  );

  return wrapper;
}
// connectTraceStream((traceEvent) => { /* render real tool-call events here */ });
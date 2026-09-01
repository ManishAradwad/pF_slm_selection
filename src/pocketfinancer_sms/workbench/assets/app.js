"use strict";

const params = new URLSearchParams(window.location.search);
const incomingToken = params.get("token");
if (incomingToken) {
  sessionStorage.setItem("workbenchToken", incomingToken);
  history.replaceState({}, "", "/");
}
const token = sessionStorage.getItem("workbenchToken") || "";
const state = {
  offset: 0,
  limit: 50,
  total: 0,
  selectedId: null,
  selectedRecord: null,
  spans: {},
  events: [],
  correctionRevision: 0,
};

const classes = ["posted_candidate", "financial_non_posted", "non_financial", "ambiguous", "invalid_outgoing"];
const eventStates = ["posted", "not_posted", "no_event", "unknown"];
const families = ["", "bank_transfer", "bill_payment", "card_purchase", "cash_deposit", "cash_withdrawal", "fee_charge", "insurance", "interest", "investment", "loan", "merchant_payment", "refund", "salary_income", "upi_transfer", "wallet", "other_financial", "unknown"];
const rails = ["", "bank_internal", "card", "cash", "imps", "nach", "neft", "other", "rtgs", "upi", "wallet", "unknown"];

function el(id) { return document.getElementById(id); }
function reviewer() {
  const value = el("reviewerId").value.trim();
  if (!value) throw new Error("Enter a local reviewer name first.");
  localStorage.setItem("workbenchReviewer", value);
  return value;
}
function optionList(target, values, firstLabel) {
  target.textContent = "";
  values.forEach((value, index) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = index === 0 && value === "" ? firstLabel : value.replaceAll("_", " ");
    target.append(option);
  });
}
optionList(el("classFilter"), ["", ...classes], "All classes");
optionList(el("railFilter"), rails, "All rails");
optionList(el("operationalClass"), ["", ...classes], "Choose…");
optionList(el("eventState"), ["", ...eventStates], "Choose…");
optionList(el("financialFamily"), families, "None");
optionList(el("paymentRail"), rails, "None");
el("reviewerId").value = localStorage.getItem("workbenchReviewer") || "";

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {"X-Workbench-Token": token, ...(options.headers || {})},
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "Local workbench request failed.");
  return body;
}
async function post(path, body) {
  return api(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
}
function toast(message, error = false) {
  const node = el("toast");
  node.textContent = message;
  node.className = error ? "show error" : "show";
  setTimeout(() => { node.className = ""; }, 3500);
}
function queryString(extra = {}) {
  const values = {reviewer_id: reviewer(), ...extra};
  const query = new URLSearchParams();
  Object.entries(values).forEach(([key, value]) => {
    if (value !== "" && value !== null && value !== undefined) query.set(key, value);
  });
  return query.toString();
}

async function loadProgress() {
  const value = await api(`/api/progress?${queryString()}`);
  el("progress").textContent = "";
  const items = [
    ["Corpus", value.total],
    ["Submitted", value.review_states.submitted || 0],
    ["Adjudicated", value.review_states.adjudicated || 0],
    ["Drafts", value.review_states.draft || 0],
  ];
  items.forEach(([label, count]) => {
    const span = document.createElement("span");
    const strong = document.createElement("strong");
    strong.textContent = Number(count).toLocaleString();
    span.append(`${label} `, strong);
    el("progress").append(span);
  });
}
function filters() {
  return {
    pool: el("poolFilter").value,
    operational_class: el("classFilter").value,
    payment_rail: el("railFilter").value,
    disposition: el("dispositionFilter").value,
    review_state: el("reviewFilter").value,
    search: el("search").value,
    sort: el("sort").value,
    descending: String(el("descending").checked),
    limit: state.limit,
    offset: state.offset,
  };
}
async function loadRows() {
  const result = await api(`/api/rows?${queryString(filters())}`);
  state.total = result.total;
  el("rowCount").textContent = `${result.total.toLocaleString()} rows`;
  el("pageLabel").textContent = `${Math.floor(state.offset / state.limit) + 1} / ${Math.max(1, Math.ceil(state.total / state.limit))}`;
  el("previousPage").disabled = state.offset === 0;
  el("nextPage").disabled = state.offset + state.limit >= state.total;
  const list = el("messageList");
  list.textContent = "";
  result.rows.forEach((row) => {
    const button = document.createElement("button");
    button.className = "message-card" + (row.source_id === state.selectedId ? " active" : "");
    const snippet = document.createElement("span");
    snippet.className = "snippet";
    snippet.textContent = row.body;
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = [row.pool, row.review_state, row.blind_locked ? "blind" : row.disposition].filter(Boolean).join(" · ");
    button.append(snippet, meta);
    button.addEventListener("click", () => selectRow(row.source_id));
    list.append(button);
  });
}

async function selectRow(sourceId) {
  const record = await api(`/api/row?${queryString({source_id: sourceId})}`);
  state.selectedId = sourceId;
  state.selectedRecord = record;
  state.spans = {};
  state.events = [];
  el("emptyDetail").hidden = true;
  el("detailContent").hidden = false;
  el("detailMeta").textContent = `${record.pool} · ${record.source_metadata.timestamp || "unknown time"}`;
  el("reviewBadge").textContent = record.review_state;
  el("senderText").textContent = record.source.sender;
  el("messageText").textContent = record.source.body;
  el("blindBanner").hidden = !record.blind_locked;
  el("revealButton").hidden = !record.can_reveal;
  el("analysisPanel").hidden = record.blind_locked;
  el("revisionLabel").textContent = record.latest_annotation ? `Revision ${record.latest_annotation.revision}` : "No revision";
  populateAnnotation(record.latest_annotation);
  renderAnalysis(record);
  renderSpans();
  renderEvents();
  await loadRows();
}
function populateAnnotation(latest) {
  const payload = latest ? latest.payload : {};
  el("decision").value = payload.decision || "";
  el("operationalClass").value = payload.operational_class || "";
  el("eventState").value = payload.event_state || "";
  el("financialFamily").value = payload.financial_family || "";
  el("paymentRail").value = payload.payment_rail || "";
  el("uncertain").checked = Boolean(payload.uncertain);
  el("notes").value = payload.notes || "";
  state.events = Array.isArray(payload.events) ? structuredClone(payload.events) : [];
}
function renderAnalysis(record) {
  const root = el("analysisContent");
  root.textContent = "";
  if (!record.analysis) return;
  const reasons = section("Queue reasons");
  const reasonList = document.createElement("div");
  reasonList.className = "reason-list";
  (record.weak_facets.reason_codes || []).forEach((reason) => {
    const node = document.createElement("span"); node.className = "reason"; node.textContent = reason; reasonList.append(node);
  });
  reasons.append(reasonList); root.append(reasons);
  const candidates = section("Grounded candidates");
  record.analysis.candidates.forEach((candidate) => {
    const item = document.createElement("div"); item.className = "candidate-item";
    const label = document.createElement("span");
    label.textContent = candidate.explicit_absence ? `${candidate.kind}: explicit absent` : `${candidate.kind}: ${candidate.evidence.text}`;
    const id = document.createElement("span"); id.className = "candidate-id"; id.textContent = candidate.candidate_id;
    item.append(label, id);
    if (candidate.evidence) {
      const use = document.createElement("button"); use.className = "secondary"; use.textContent = "Use";
      use.addEventListener("click", () => setSpan(`${candidate.kind}_span`, {start_char: candidate.evidence.start_char, end_char: candidate.evidence.end_char}));
      item.append(use);
    }
    candidates.append(item);
  });
  root.append(candidates);
  const cues = section("Structural cues");
  record.analysis.cues.forEach((cue) => {
    const item = document.createElement("div"); item.className = "cue-item"; item.textContent = `${cue.kind} · ${cue.reason_code} · ${cue.evidence.text}`; cues.append(item);
  });
  root.append(cues);
  if (record.processing_trace) {
    const trace = section("Processing trace");
    const pre = document.createElement("pre"); pre.textContent = JSON.stringify(record.processing_trace, null, 2); trace.append(pre); root.append(trace);
  }
}
function section(title) { const node = document.createElement("div"); node.className = "analysis-section"; const h = document.createElement("h3"); h.textContent = title; node.append(h); return node; }

function selectedSpan() {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount !== 1 || selection.isCollapsed) throw new Error("Select exact message text first.");
  const range = selection.getRangeAt(0);
  const root = el("messageText");
  if (!root.contains(range.commonAncestorContainer)) throw new Error("Selection must be inside the message.");
  const before = range.cloneRange();
  before.selectNodeContents(root);
  before.setEnd(range.startContainer, range.startOffset);
  const start = before.toString().length;
  const end = start + range.toString().length;
  return {start_char: start, end_char: end};
}
function setSpan(field, value) { state.spans[field] = value; renderSpans(); toast(`${field.replace("_span", "")} evidence selected.`); }
function spanText(span) { return span ? state.selectedRecord.source.body.slice(span.start_char, span.end_char) : "not selected"; }
function renderSpans() {
  const root = el("spanSummary"); root.textContent = "";
  ["amount_span", "direction_span", "account_span", "counterparty_span"].forEach((field) => {
    const node = document.createElement("span"); node.textContent = `${field.replace("_span", "")}: ${spanText(state.spans[field])}`; root.append(node);
  });
}
function currentEvent() {
  return {
    amount_span: state.spans.amount_span || null,
    currency: el("currency").value.trim().toUpperCase(),
    currency_provenance: el("currencyProvenance").value,
    direction: el("direction").value,
    direction_span: state.spans.direction_span || null,
    account_state: el("accountState").value,
    account_span: el("accountState").value === "present" ? state.spans.account_span || null : null,
    counterparty_state: el("counterpartyState").value,
    counterparty_span: el("counterpartyState").value === "present" ? state.spans.counterparty_span || null : null,
    financial_family: el("financialFamily").value || null,
    payment_rail: el("paymentRail").value || null,
  };
}
function addEvent() {
  const event = currentEvent();
  if (!event.amount_span || !event.direction_span) throw new Error("Amount and direction evidence are required for an event.");
  if (event.account_state === "present" && !event.account_span) throw new Error("Select account evidence or mark it absent/unknown.");
  if (event.counterparty_state === "present" && !event.counterparty_span) throw new Error("Select counterparty evidence or mark it absent/unknown.");
  state.events.push(event); state.spans = {}; renderSpans(); renderEvents();
}
function renderEvents() {
  const root = el("eventList"); root.textContent = "";
  state.events.forEach((event, index) => {
    const node = document.createElement("div"); node.className = "event-item";
    const text = document.createElement("span"); text.textContent = `Event ${index + 1}: ${event.currency} · ${event.direction} · ${spanText(event.amount_span)}`;
    const remove = document.createElement("button"); remove.className = "secondary"; remove.textContent = "Remove";
    remove.addEventListener("click", () => { state.events.splice(index, 1); renderEvents(); });
    node.append(text, remove); root.append(node);
  });
}
function buildPayload() {
  const decision = el("decision").value;
  let events = structuredClone(state.events);
  if ((decision === "posted" || decision === "multiple_event") && events.length === 0) events = [currentEvent()];
  return {
    decision,
    operational_class: el("operationalClass").value,
    event_state: el("eventState").value,
    financial_family: el("financialFamily").value || null,
    payment_rail: el("paymentRail").value || null,
    events,
    uncertain: el("uncertain").checked,
    notes: el("notes").value,
  };
}
function currentRevision() { return state.selectedRecord.latest_annotation ? state.selectedRecord.latest_annotation.revision : 0; }
async function save(path) {
  if (!state.selectedId) throw new Error("Choose a message first.");
  await post(path, {source_id: state.selectedId, reviewer_id: reviewer(), expected_revision: currentRevision(), payload: buildPayload()});
  toast(path === "/api/draft" ? "Draft saved." : "Label saved.");
  await selectRow(state.selectedId); await loadProgress();
}

el("refreshButton").addEventListener("click", () => run(async () => { state.offset = 0; await loadRows(); }));
el("previousPage").addEventListener("click", () => run(async () => { state.offset = Math.max(0, state.offset - state.limit); await loadRows(); }));
el("nextPage").addEventListener("click", () => run(async () => { state.offset += state.limit; await loadRows(); }));
document.querySelectorAll("[data-span-field]").forEach((button) => button.addEventListener("click", () => run(() => setSpan(button.dataset.spanField, selectedSpan()))));
el("addEventButton").addEventListener("click", () => run(addEvent));
el("saveDraftButton").addEventListener("click", () => run(() => save("/api/draft")));
el("submitButton").addEventListener("click", () => run(() => save("/api/submit")));
el("adjudicateButton").addEventListener("click", () => run(() => save("/api/adjudicate")));
el("revealButton").addEventListener("click", () => run(async () => { await post("/api/reveal", {source_id: state.selectedId, reviewer_id: reviewer()}); await selectRow(state.selectedId); toast("Deterministic analysis revealed."); }));
el("previewButton").addEventListener("click", () => run(async () => { const value = await api(`/api/preview?${queryString({source_id: state.selectedId})}`); el("preview").textContent = JSON.stringify(value, null, 2); }));
el("correctionButton").addEventListener("click", () => run(async () => {
  const facets = {operational_class: el("operationalClass").value, event_state: el("eventState").value, financial_family: el("financialFamily").value || null, payment_rail: el("paymentRail").value || null, reason: el("correctionReason").value};
  const value = await post("/api/correction", {source_id: state.selectedId, reviewer_id: reviewer(), expected_revision: state.correctionRevision, facets});
  state.correctionRevision = value.revision; toast("Weak segregation correction saved separately.");
}));
el("backupButton").addEventListener("click", () => run(async () => { const value = await post("/api/backup", {}); toast(`Backup created: ${value.backup}`); }));
el("exportButton").addEventListener("click", () => run(async () => { const value = await post("/api/export", {}); toast(`Export ${value.export_id} created with ${value.label_count} labels.`); }));
el("decision").addEventListener("change", () => {
  const decision = el("decision").value;
  const defaults = {posted: ["posted_candidate", "posted"], not_posted: ["financial_non_posted", "not_posted"], non_financial: ["non_financial", "no_event"], ambiguous: ["ambiguous", "unknown"], multiple_event: ["posted_candidate", "posted"]};
  if (defaults[decision]) [el("operationalClass").value, el("eventState").value] = defaults[decision];
  el("eventEditor").hidden = !["posted", "multiple_event"].includes(decision);
});

async function run(task) { try { await task(); } catch (error) { toast(error.message || "Operation failed.", true); } }
run(async () => { await loadProgress(); await loadRows(); });

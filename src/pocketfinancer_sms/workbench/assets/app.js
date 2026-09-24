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
  rowIds: [],
  focusedSpans: {},
  focusedDirty: false,
  activeFocusedField: null,
  selectedId: null,
  selectedRecord: null,
  selectedReviewer: null,
  correctionRevision: 0,
  hasDisagreement: false,
  exportSelections: [],
  listHistory: [],
  groupFilters: {
    normalized_template_group: null,
    sender_family_group: null,
    sender_template_group: null,
  },
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
optionList(el("eventStateFilter"), ["", ...eventStates], "All states");
optionList(el("familyFilter"), families, "All families");
optionList(el("railFilter"), rails, "All rails");
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

function exportIdentity(annotation, sourceId) {
  if (!annotation || !["submitted", "adjudicated"].includes(annotation.status) ||
      !annotation.canonical_label || !annotation.revision_hash) return null;
  return {
    source_id: sourceId, reviewer_id: annotation.reviewer_id || state.selectedReviewer,
    revision: annotation.revision, revision_hash: annotation.revision_hash,
  };
}
function sameExportRevision(first, second) {
  return first.source_id === second.source_id &&
    first.reviewer_id === second.reviewer_id && first.revision === second.revision;
}
function updateExportControls() {
  el("exportButton").textContent = `Export selected (${state.exportSelections.length})`;
  const current = state.selectedRecord && exportIdentity(
    state.selectedRecord.latest_annotation, state.selectedId,
  );
  el("selectExportButton").disabled = !current;
  el("selectExportButton").textContent = current && state.exportSelections.some(
    (item) => sameExportRevision(item, current)
  ) ? "Remove this revision from export" : "Select this revision for export";
}
function toggleExportRevision(annotation, sourceId) {
  const selected = exportIdentity(annotation, sourceId);
  if (!selected) throw new Error("Only submitted or adjudicated revisions can be exported.");
  const index = state.exportSelections.findIndex((item) => sameExportRevision(item, selected));
  if (index >= 0) state.exportSelections.splice(index, 1);
  else state.exportSelections.push(selected);
  updateExportControls();
}

async function loadProgress() {
  const coverage = el("coverageTables");
  coverage.textContent = "Loading coverage…";
  let value;
  try {
    value = await api(`/api/progress?${queryString()}`);
  } catch (error) {
    coverage.textContent = error.message || "Coverage data could not be loaded.";
    throw error;
  }
  el("progress").textContent = "";
  const items = [
    ["Corpus", value.total],
    ["Submitted", value.review_states.submitted || 0],
    ["Adjudicated", value.review_states.adjudicated || 0],
    ["Drafts", value.review_states.draft || 0],
    ["My remaining", value.my_remaining || 0],
    ["Disagreements", value.disagreements || 0],
    ["Validation failures", value.validation_failures || 0],
    ["Submitted last hour", value.submitted_last_hour || 0],
    ["Submitted last day", value.submitted_last_day || 0],
  ];
  items.forEach(([label, count]) => {
    const span = document.createElement("span");
    const strong = document.createElement("strong");
    strong.textContent = Number(count).toLocaleString();
    span.append(`${label} `, strong);
    el("progress").append(span);
  });
  coverage.textContent = "";
  coverage.append(coverageTable("Annotation pools", value.pool_coverage));
  coverage.append(coverageTable("Weak operational classes", value.class_coverage));
}
function coverageTable(title, values) {
  const section = document.createElement("section");
  const heading = document.createElement("h3"); heading.textContent = title; section.append(heading);
  const table = document.createElement("table");
  const head = document.createElement("tr");
  ["Category", "Reviewed", "Total"].forEach((label) => { const cell = document.createElement("th"); cell.textContent = label; head.append(cell); });
  table.append(head);
  Object.entries(values || {}).sort().forEach(([name, counts]) => {
    const row = document.createElement("tr");
    [name.replaceAll("_", " "), Number(counts.reviewed).toLocaleString(), Number(counts.total).toLocaleString()].forEach((value) => { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); });
    table.append(row);
  });
  section.append(table); return section;
}
function filters() {
  return {
    pool: el("poolFilter").value,
    operational_class: el("classFilter").value,
    event_state: el("eventStateFilter").value,
    financial_family: el("familyFilter").value,
    payment_rail: el("railFilter").value,
    disposition: el("dispositionFilter").value,
    selector_action: el("selectorActionFilter").value,
    review_state: el("reviewFilter").value,
    reviewer_state: el("reviewerStateFilter").value,
    disagreement: el("disagreementFilter").value,
    candidate_coverage: el("candidateCoverageFilter").value,
    imported_feedback: el("importedFeedbackFilter").value,
    time_group: el("timeGroupFilter").value,
    time_from: el("timeFromFilter").value,
    time_to: el("timeToFilter").value,
    ...state.groupFilters,
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
  state.rowIds = result.rows.map((row) => row.source_id);
  updateFocusedPosition();
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
    button.addEventListener("click", () => run(() => selectRow(row.source_id)));
    list.append(button);
  });
}

const resumeControlIds = {
  pool: "poolFilter", operational_class: "classFilter",
  event_state: "eventStateFilter", financial_family: "familyFilter",
  payment_rail: "railFilter", disposition: "dispositionFilter",
  selector_action: "selectorActionFilter", review_state: "reviewFilter",
  reviewer_state: "reviewerStateFilter",
  disagreement: "disagreementFilter",
  candidate_coverage: "candidateCoverageFilter",
  imported_feedback: "importedFeedbackFilter",
  time_group: "timeGroupFilter",
  time_from: "timeFromFilter", time_to: "timeToFilter",
};
async function saveResume() {
  const current = filters();
  const savedFilters = {};
  Object.keys(resumeControlIds).forEach((key) => { savedFilters[key] = current[key]; });
  Object.assign(savedFilters, state.groupFilters);
  await post("/api/resume", {
    reviewer_id: state.selectedReviewer, source_id: state.selectedId,
    offset: state.offset, filters: savedFilters, search: current.search,
    sort: current.sort, descending: current.descending === "true",
  });
}
async function resumeQueue() {
  state.listHistory = [];
  const saved = await api("/api/resume?" + queryString());
  if (saved) {
    Object.entries(resumeControlIds).forEach(([key, id]) => {
      el(id).value = saved.filters[key] || "";
    });
    state.groupFilters = {
      normalized_template_group: saved.filters.normalized_template_group || null,
      sender_family_group: saved.filters.sender_family_group || null,
      sender_template_group: saved.filters.sender_template_group || null,
    };
    el("search").value = saved.search || "";
    el("sort").value = saved.sort || "timestamp";
    el("descending").checked = Boolean(saved.descending);
    state.offset = saved.offset;
  } else {
    Object.values(resumeControlIds).forEach((id) => { el(id).value = ""; });
    state.groupFilters = {
      normalized_template_group: null,
      sender_family_group: null,
      sender_template_group: null,
    };
    el("reviewerStateFilter").value = "unfinished";
    el("search").value = "";
    el("sort").value = "timestamp";
    el("descending").checked = false;
    state.offset = 0;
  }
  await loadRows();
  if (state.total && !state.rowIds.length) {
    state.offset = 0;
    await loadRows();
  }
  let target = saved && state.rowIds.includes(saved.source_id) ? saved.source_id : null;
  if (target) {
    const record = await api("/api/row?" + queryString({source_id: target}));
    const status = record.latest_annotation && record.latest_annotation.status;
    if (status === "submitted" || status === "adjudicated") {
      target = null;
      el("reviewerStateFilter").value = "unfinished";
      el("reviewFilter").value = "";
      state.offset = 0;
      await loadRows();
    }
  }
  if (!target) target = state.rowIds[0];
  if (target) {
    await selectRow(target);
  } else {
    state.selectedId = null;
    state.selectedRecord = null;
    el("detailContent").hidden = true;
    el("emptyDetail").hidden = false;
    updateExportControls();
  }
}

async function selectRow(sourceId) {
  if (state.selectedId && state.selectedId !== sourceId) await flushFocusedDraft();
  const record = await api(`/api/row?${queryString({source_id: sourceId})}`);
  state.selectedId = sourceId;
  state.selectedRecord = record;
  state.selectedReviewer = reviewer();
  state.hasDisagreement = false;
  state.correctionRevision = record.latest_weak_correction ? record.latest_weak_correction.revision : 0;
  el("emptyDetail").hidden = true;
  el("detailContent").hidden = false;
  el("detailContent").closest(".detail").scrollTop = 0;
  el("detailMeta").textContent = `${record.pool} · ${record.source_metadata.timestamp || "unknown time"}`;
  el("reviewBadge").textContent = record.review_state;
  el("senderText").textContent = record.source.sender;
  el("messageText").textContent = record.source.body;
  el("blindBanner").hidden = !record.blind_locked;
  el("revealButton").hidden = !record.can_reveal;
  el("analysisPanel").hidden = record.blind_locked;
  el("revisionLabel").textContent = record.latest_annotation ? `Revision ${record.latest_annotation.revision}` : "No revision";
  populateFocused(record);
  updateExportControls();
  renderAnalysis(record);
  renderGroupNavigation(record);
  renderFocusedEvidence();
  await loadDisagreements(record);
  await loadRows();
  updateFocusedPosition();
  await saveResume();
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
  if (record.candidate_coverage) {
    const coverage = section("Candidate-oracle coverage");
    const counts = record.candidate_coverage.field_candidate_counts;
    const summary = document.createElement("p");
    summary.textContent = `Amount ${counts.amount} · direction ${counts.direction} · account ${counts.account} · counterparty ${counts.counterparty} · complete core clauses ${record.candidate_coverage.complete_core_clause_count}`;
    coverage.append(summary); root.append(coverage);
  }
  const candidates = section("Analyzer suggestions");
  record.analysis.candidates.forEach((candidate) => {
    const item = document.createElement("div"); item.className = "candidate-item";
    const label = document.createElement("span");
    label.textContent = candidate.explicit_absence ? `${candidate.kind}: explicit absent` : `${candidate.kind}: ${candidate.evidence.text}`;
    const id = document.createElement("span"); id.className = "candidate-id"; id.textContent = candidate.candidate_id;
    item.append(label, id);
    if (candidate.evidence) {
      const use = document.createElement("button"); use.className = "secondary"; use.textContent = "Use analyzer suggestion";
      use.addEventListener("click", () => run(() => setFocusedSpan(`${candidate.kind}_span`, {start_char: candidate.evidence.start_char, end_char: candidate.evidence.end_char})));
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
  if (Array.isArray(record.native_suggestions) && record.native_suggestions.length > 0) {
    const suggestions = section("Imported native correction evidence (review before use)");
    record.native_suggestions.forEach((suggestion) => {
      const item = document.createElement("div"); item.className = "candidate-item";
      const label = document.createElement("span");
      label.textContent = `${suggestion.source_platform} correction · ${suggestion.field}: ${suggestion.evidence.text}`;
      const use = document.createElement("button"); use.className = "secondary";
      use.textContent = "Use native evidence";
      use.addEventListener("click", () => run(() => setFocusedSpan(
        `${suggestion.field}_span`, suggestion.evidence,
      )));
      item.append(label, use); suggestions.append(item);
    });
    root.append(suggestions);
  }
  if (Array.isArray(record.native_traces) && record.native_traces.length > 0) {
    const traces = section("Imported native traces");
    record.native_traces.forEach((trace, index) => {
      const heading = document.createElement("p");
      const coverage = trace.candidate_coverage;
      heading.textContent = `${index + 1}. ${trace.source_platform} · candidates ${coverage.deterministic_candidate_count} · selected ${coverage.selected_candidate_count} · misses ${coverage.candidate_miss_count}`;
      const pre = document.createElement("pre");
      pre.textContent = trace.record_json;
      traces.append(heading, pre);
    });
    root.append(traces);
  }
  if (Array.isArray(record.annotation_history) && record.annotation_history.length > 0) {
    const history = section("Your annotation revision history");
    record.annotation_history.forEach((revision) => {
      const item = document.createElement("p");
      item.textContent = `Revision ${revision.revision} · ${revision.status} · ${revision.revision_hash}`;
      if (exportIdentity(revision, record.source_id)) {
        const button = document.createElement("button");
        button.className = "secondary";
        button.textContent = "Select this revision for export";
        button.addEventListener("click", () => run(() => toggleExportRevision(revision, record.source_id)));
        item.append(" ", button);
      }
      history.append(item);
    });
    root.append(history);
  }
}
function section(title) { const node = document.createElement("div"); node.className = "analysis-section"; const h = document.createElement("h3"); h.textContent = title; node.append(h); return node; }

function renderGroupNavigation(record) {
  const root = el("groupNavigation"); root.textContent = "";
  if (!record || !record.grouping) return;
  const groups = [
    ["normalized_template_group", record.grouping.normalized_template_hash, "Show this template family"],
    ["sender_family_group", record.grouping.sender_family_hash, "Show this sender family"],
    ["sender_template_group", record.grouping.sender_template_group_hash, "Show this sender-template group"],
  ];
  groups.forEach(([key, value, label]) => {
    const button = document.createElement("button"); button.className = "secondary"; button.textContent = label;
    button.addEventListener("click", () => run(async () => {
      const previous = {filters: filters(), selectedId: state.selectedId};
      state.listHistory.push(previous);
      state.groupFilters = {normalized_template_group: null, sender_family_group: null, sender_template_group: null};
      state.groupFilters[key] = value;
      await showQueuePage(0);
      toast(`${label} filter applied.`);
    }));
    root.append(button);
  });
  if (state.listHistory.length) {
    const back = document.createElement("button"); back.className = "secondary"; back.textContent = "Back to previous list";
    back.addEventListener("click", () => run(async () => {
      const previous = state.listHistory.at(-1);
      Object.entries(resumeControlIds).forEach(([key, id]) => { el(id).value = previous.filters[key] || ""; });
      state.groupFilters = {
        normalized_template_group: previous.filters.normalized_template_group || null,
        sender_family_group: previous.filters.sender_family_group || null,
        sender_template_group: previous.filters.sender_template_group || null,
      };
      el("search").value = previous.filters.search || "";
      el("sort").value = previous.filters.sort || "timestamp";
      el("descending").checked = previous.filters.descending === "true";
      await showQueuePage(previous.filters.offset, previous.selectedId);
      state.listHistory.pop();
      renderGroupNavigation(state.selectedRecord);
    }));
    root.append(back);
  }
  const clear = document.createElement("button"); clear.className = "secondary"; clear.textContent = "Clear group filter";
  clear.addEventListener("click", () => run(async () => {
    state.listHistory = [];
    state.groupFilters = {normalized_template_group: null, sender_family_group: null, sender_template_group: null};
    await showQueuePage(0);
  }));
  root.append(clear);
}

async function loadDisagreements(record) {
  const root = el("disagreementContent"); root.textContent = "";
  el("focusAdjudicate").disabled = true;
  if (record.blind_locked) { root.textContent = "Agreement details remain hidden during blind review."; return; }
  const value = await api(`/api/disagreements?${queryString({source_id: record.source_id})}`);
  state.hasDisagreement = value.has_disagreement;
  el("focusAdjudicate").disabled = !value.has_disagreement;
  if (value.review_count === 0) { root.textContent = "No submitted reviews yet."; return; }
  value.annotations.forEach((annotation) => {
    const item = document.createElement("p");
    const decision = annotation.canonical_label ? annotation.canonical_label.decision : "unavailable";
    item.textContent = `${annotation.reviewer_id} · ${decision} · revision ${annotation.revision} · ${annotation.revision_hash}`;
    root.append(item);
  });
}

function selectedSpan() {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount !== 1 || selection.isCollapsed) throw new Error("Select exact message text first.");
  const range = selection.getRangeAt(0);
  const root = el("messageText");
  if (!root.contains(range.commonAncestorContainer)) throw new Error("Selection must be inside the message.");
  const before = range.cloneRange();
  before.selectNodeContents(root);
  before.setEnd(range.startContainer, range.startOffset);
  const start = Array.from(before.toString()).length;
  const end = start + Array.from(range.toString()).length;
  return {start_char: start, end_char: end};
}
function currentRevision() {
  return state.selectedRecord.latest_annotation ? state.selectedRecord.latest_annotation.revision : 0;
}

async function showQueuePage(offset, preferredId = null) {
  await flushFocusedDraft();
  state.offset = offset;
  await loadRows();
  if (state.rowIds[0]) {
    await selectRow(preferredId && state.rowIds.includes(preferredId) ? preferredId : state.rowIds[0]);
  } else {
    state.selectedId = null;
    state.selectedRecord = null;
    el("detailContent").hidden = true;
    el("emptyDetail").hidden = false;
    el("groupNavigation").textContent = "";
    updateFocusedPosition();
  }
}
el("refreshButton").addEventListener("click", () => run(() => showQueuePage(0)));
el("previousPage").addEventListener("click", () => run(() => showQueuePage(Math.max(0, state.offset - state.limit))));
el("nextPage").addEventListener("click", () => run(() => showQueuePage(state.offset + state.limit)));
el("revealButton").addEventListener("click", () => run(async () => { await post("/api/reveal", {source_id: state.selectedId, reviewer_id: state.selectedReviewer}); await selectRow(state.selectedId); toast("Deterministic analysis revealed."); }));
el("correctionButton").addEventListener("click", () => run(async () => {
  const facets = {operational_class: el("focusClass").value, event_state: el("focusState").value, financial_family: el("focusFamily").value || null, payment_rail: el("focusRail").value || null, reason: el("correctionReason").value};
  const value = await post("/api/correction", {source_id: state.selectedId, reviewer_id: state.selectedReviewer, expected_revision: state.correctionRevision, facets});
  state.correctionRevision = value.revision; toast("Weak segregation correction saved separately.");
}));
el("backupButton").addEventListener("click", () => run(async () => { const value = await post("/api/backup", {}); toast(`Backup created: ${value.backup}`); }));
el("exportButton").addEventListener("click", () => run(async () => {
  if (!state.exportSelections.length) throw new Error("Select at least one submitted revision first.");
  if (!window.confirm(`Create a local encrypted export of these ${state.exportSelections.length} selected revisions?`)) return;
  const value = await post("/api/export", {
    explicit_consent: true, selected_revisions: state.exportSelections,
  });
  state.exportSelections = [];
  updateExportControls();
  toast(`Encrypted export ${value.export_id} created with ${value.label_count} labels.`);
}));
el("selectExportButton").addEventListener("click", () => run(() => {
  toggleExportRevision(state.selectedRecord.latest_annotation, state.selectedId);
}));
el("coveragePanel").addEventListener("toggle", () => {
  el("coverageToggleHint").textContent = el("coveragePanel").open ? "Hide breakdown" : "Show breakdown";
});

async function run(task) { try { await task(); } catch (error) { toast(error.message || "Operation failed.", true); } }


const focusFields = ["amount_span", "direction_span", "account_span", "counterparty_span"];
let focusedAutosaveTimer = null;
let focusedSaveInFlight = Promise.resolve();

optionList(el("focusClass"), ["", ...classes], "Choose…");
optionList(el("focusState"), ["", ...eventStates], "Choose…");
optionList(el("focusFamily"), families, "None");
optionList(el("focusRail"), rails, "None");

function populateFocused(record) {
  clearTimeout(focusedAutosaveTimer);
  state.focusedDirty = false;
  state.focusedSpans = {};
  state.activeFocusedField = null;
  const latest = record.latest_annotation;
  const payload = latest && latest.payload ? latest.payload : {};
  const contract = payload.contract || (latest && latest.canonical_label && latest.canonical_label.contract);
  const legacy = Boolean(latest) && contract !== "pocketfinancer.canonical-label/2";
  el("legacyNotice").hidden = !legacy;
  el("focusedEditor").hidden = legacy;
  const historyPanel = el("legacyHistoryPanel");
  const historyContent = el("legacyHistoryContent");
  historyContent.textContent = "";
  const oldRevisions = record.blind_locked ? [] : (record.annotation_history || []).filter((item) => {
    const value = item.canonical_label || item.payload || {};
    return value.contract !== "pocketfinancer.canonical-label/2";
  });
  historyPanel.hidden = oldRevisions.length === 0;
  oldRevisions.forEach((item) => {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Revision " + item.revision + " · " + item.status + " · canonical-label/1";
    const content = document.createElement("pre");
    content.textContent = JSON.stringify(item.canonical_label || item.payload, null, 2);
    details.append(summary, content);
    historyContent.append(details);
  });
  const event = !legacy && payload.event && typeof payload.event === "object" ? payload.event : {};
  if (!legacy) {
    focusFields.forEach((field) => {
      if (event[field]) state.focusedSpans[field] = event[field];
    });
  }
  el("focusDecision").value = legacy ? "" : payload.decision || "";
  el("focusClass").value = legacy ? "" : payload.operational_class || "";
  el("focusState").value = legacy ? "" : payload.event_state || "";
  el("focusFamily").value = legacy ? "" : payload.financial_family || "";
  el("focusRail").value = legacy ? "" : payload.payment_rail || "";
  el("focusUncertain").checked = !legacy && Boolean(payload.uncertain);
  el("focusNotes").value = legacy ? "" : payload.notes || "";
  el("focusAmount").value = event.amount_value || "";
  el("focusCurrency").value = event.currency || "";
  el("focusDirection").value = event.direction || "";
  el("focusAccount").value = event.account_reference || "";
  el("focusAccountId").value = event.existing_account_id || "";
  el("focusCounterparty").value = event.counterparty || "";
  el("focusEvent").hidden = el("focusDecision").value !== "posted";
  el("focusSaveState").textContent = legacy ? "Historical v1 revision remains read-only." : "";
  el("focusCorrection").disabled = legacy || !latest ||
    !["submitted", "adjudicated"].includes(latest.status);
  el("focusPreviewOutput").textContent = "";
  renderFocusedSpanSummary();
}

function updateFocusedPosition() {
  const node = el("focusedPosition");
  if (!node) return;
  const index = state.rowIds.indexOf(state.selectedId);
  if (index < 0) {
    node.textContent = state.total ? String(state.total) + " messages in the current queue" : "No messages in this queue";
    return;
  }
  const position = state.offset + index + 1;
  node.textContent = "Message " + position + " of " + state.total + " in the current queue · " + Math.max(0, state.total - position) + " after this";
}

function scalarSpan(value) {
  const source = Array.from(state.selectedRecord.source.body);
  const start = value.start_char;
  const end = value.end_char;
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start || end > source.length) {
    throw new Error("Select exact message text.");
  }
  return {start_scalar: start, end_scalar: end, text: source.slice(start, end).join("")};
}

function setFocusedSpan(field, value) {
  if (el("focusedEditor").hidden) throw new Error("Start a v2 revision before assigning evidence.");
  if (el("focusDecision").value !== "posted") throw new Error("Choose Posted before assigning field evidence.");
  if (!focusFields.includes(field)) throw new Error("Choose a supported field.");
  const span = scalarSpan(value);
  state.focusedSpans[field] = span;
  state.activeFocusedField = field;
  if (field === "account_span" && !el("focusAccount").value) el("focusAccount").value = span.text;
  if (field === "counterparty_span" && !el("focusCounterparty").value) el("focusCounterparty").value = span.text;
  renderFocusedSpanSummary();
  renderFocusedEvidence();
  scheduleFocusedDraft();
}

function renderFocusedSpanSummary() {
  const root = el("focusSpanSummary");
  root.textContent = "";
  focusFields.forEach((field) => {
    const span = state.focusedSpans[field];
    const item = document.createElement("div");
    const label = document.createElement("span");
    label.textContent = field.replace("_span", "") + ": " + (span ? span.text : "unassigned");
    item.append(label);
    if (span) {
      const clear = document.createElement("button");
      clear.type = "button";
      clear.className = "secondary compact";
      clear.textContent = "Clear";
      clear.addEventListener("click", () => {
        delete state.focusedSpans[field];
        state.activeFocusedField = field;
        renderFocusedSpanSummary();
        renderFocusedEvidence();
        scheduleFocusedDraft();
      });
      item.append(clear);
    }
    root.append(item);
  });
}

function renderFocusedEvidence() {
  const record = state.selectedRecord;
  if (!record) return;
  const root = el("messageText");
  const chars = Array.from(record.source.body);
  root.textContent = "";
  const boundaries = new Set([0, chars.length]);
  focusFields.forEach((field) => {
    const span = state.focusedSpans[field];
    if (span) {
      boundaries.add(span.start_scalar);
      boundaries.add(span.end_scalar);
    }
  });
  const points = [...boundaries].sort((a, b) => a - b);
  for (let index = 0; index < points.length - 1; index += 1) {
    const start = points[index];
    const end = points[index + 1];
    const text = chars.slice(start, end).join("");
    const covering = focusFields.filter((field) => {
      const span = state.focusedSpans[field];
      return span && span.start_scalar <= start && span.end_scalar >= end;
    });
    if (covering.length === 0) {
      root.append(document.createTextNode(text));
    } else {
      const mark = document.createElement("mark");
      mark.className = "field-mark " + (covering.length === 1 ? covering[0] : "overlap");
      if (covering.includes(state.activeFocusedField)) mark.classList.add("active-field");
      mark.title = covering.map((field) => field.replace("_span", "")).join(", ");
      mark.textContent = text;
      root.append(mark);
    }
  }
}

function focusedPayload() {
  const decision = el("focusDecision").value;
  const event = decision === "posted" ? {
    amount_value: el("focusAmount").value.trim(),
    currency: el("focusCurrency").value.trim().toUpperCase(),
    amount_span: state.focusedSpans.amount_span || null,
    direction: el("focusDirection").value,
    direction_span: state.focusedSpans.direction_span || null,
    account_reference: el("focusAccount").value.trim(),
    account_span: state.focusedSpans.account_span || null,
    existing_account_id: el("focusAccountId").value.trim() || null,
    counterparty: el("focusCounterparty").value.trim() || null,
    counterparty_span: state.focusedSpans.counterparty_span || null,
  } : null;
  return {
    contract: "pocketfinancer.canonical-label/2",
    decision,
    operational_class: el("focusClass").value,
    event_state: el("focusState").value,
    financial_family: el("focusFamily").value || null,
    payment_rail: el("focusRail").value || null,
    event,
    uncertain: el("focusUncertain").checked,
    notes: el("focusNotes").value,
  };
}

function scheduleFocusedDraft() {
  if (!state.selectedId || el("focusedEditor").hidden) return;
  state.focusedDirty = true;
  el("focusSaveState").textContent = "Unsaved draft";
  clearTimeout(focusedAutosaveTimer);
  focusedAutosaveTimer = setTimeout(() => run(saveFocusedDraft), 1000);
}

async function saveFocusedDraft() {
  clearTimeout(focusedAutosaveTimer);
  await focusedSaveInFlight;
  if (!state.focusedDirty || !state.selectedId || el("focusedEditor").hidden) return;
  const sourceId = state.selectedId;
  const payload = focusedPayload();
  const revision = currentRevision();
  state.focusedDirty = false;
  const save = post("/api/draft", {
    source_id: sourceId, reviewer_id: state.selectedReviewer,
    expected_revision: revision, payload,
  });
  focusedSaveInFlight = save;
  try {
    const result = await save;
    if (state.selectedId === sourceId) {
      state.selectedRecord.latest_annotation = {
        revision: result.revision, status: "draft", payload,
        canonical_label: null,
      };
      el("revisionLabel").textContent = "Revision " + result.revision;
      el("focusSaveState").textContent = "Draft saved locally";
    }
  } catch (error) {
    state.focusedDirty = true;
    throw error;
  } finally {
    focusedSaveInFlight = Promise.resolve();
  }
}

async function flushFocusedDraft() {
  clearTimeout(focusedAutosaveTimer);
  await focusedSaveInFlight;
  if (state.focusedDirty) await saveFocusedDraft();
}

async function submitFocused(adjudicated = false) {
  if (!state.selectedId || el("focusedEditor").hidden) throw new Error("Choose a v2 annotation first.");
  clearTimeout(focusedAutosaveTimer);
  await focusedSaveInFlight;
  const path = adjudicated ? "/api/adjudicate" : "/api/submit";
  await post(path, {
    source_id: state.selectedId, reviewer_id: state.selectedReviewer,
    expected_revision: currentRevision(), payload: focusedPayload(),
  });
  state.focusedDirty = false;
  const sourceId = state.selectedId;
  await selectRow(sourceId);
  await loadProgress();
  toast(adjudicated ? "Adjudication saved." : "V2 label submitted.");
}

async function navigateFocused(direction) {
  await flushFocusedDraft();
  const index = state.rowIds.indexOf(state.selectedId);
  let target = index + direction;
  if (index < 0) target = direction > 0 ? 0 : state.rowIds.length - 1;
  if (target >= 0 && target < state.rowIds.length) {
    await selectRow(state.rowIds[target]);
    return;
  }
  const nextOffset = state.offset + (direction > 0 ? state.limit : -state.limit);
  if (nextOffset < 0 || nextOffset >= state.total) {
    toast("End of the current queue.");
    return;
  }
  state.offset = nextOffset;
  await loadRows();
  await selectRow(state.rowIds[direction > 0 ? 0 : state.rowIds.length - 1]);
}

el("startV2Button").addEventListener("click", () => {
  el("legacyNotice").hidden = true;
  el("focusedEditor").hidden = false;
  el("focusSaveState").textContent = "New v2 revision ready";
  el("focusDecision").focus();
});
el("focusDecision").addEventListener("change", () => {
  const defaults = {
    posted: ["posted_candidate", "posted"],
    none: ["non_financial", "no_event"],
    abstain: ["ambiguous", "unknown"],
  };
  const selected = defaults[el("focusDecision").value];
  if (selected) [el("focusClass").value, el("focusState").value] = selected;
  el("focusEvent").hidden = el("focusDecision").value !== "posted";
  scheduleFocusedDraft();
});
document.querySelectorAll("[data-v2-span]").forEach((button) => {
  button.addEventListener("click", () => run(() => setFocusedSpan(button.dataset.v2Span, selectedSpan())));
});
el("clearV2Span").addEventListener("click", () => run(() => {
  const field = state.activeFocusedField;
  if (!field || !state.focusedSpans[field]) throw new Error("Assign a field before clearing it.");
  delete state.focusedSpans[field];
  renderFocusedSpanSummary();
  renderFocusedEvidence();
  scheduleFocusedDraft();
}));
el("focusedEditor").querySelectorAll("input, select, textarea").forEach((control) => {
  if (control.id !== "focusDecision") control.addEventListener("input", scheduleFocusedDraft);
});
el("focusSaveDraft").addEventListener("click", () => run(async () => {
  if (!state.focusedDirty) scheduleFocusedDraft();
  await saveFocusedDraft();
}));
el("focusCorrection").addEventListener("click", () => run(async () => {
  const latest = state.selectedRecord && state.selectedRecord.latest_annotation;
  if (!latest || !["submitted", "adjudicated"].includes(latest.status)) {
    throw new Error("Choose a submitted label to correct.");
  }
  scheduleFocusedDraft();
  await saveFocusedDraft();
  el("focusCorrection").disabled = true;
  toast("New draft revision saved. The submitted revision remains in history.");
}));
el("focusSubmit").addEventListener("click", () => run(() => submitFocused()));
el("focusAdjudicate").addEventListener("click", () => run(() => submitFocused(true)));
el("focusPreview").addEventListener("click", () => run(async () => {
  const value = await api("/api/preview?" + queryString({source_id: state.selectedId}));
  el("focusPreviewOutput").textContent = JSON.stringify(value, null, 2);
}));
el("focusPrevious").addEventListener("click", () => run(() => navigateFocused(-1)));
el("focusSkip").addEventListener("click", () => run(() => navigateFocused(1)));
el("focusSaveNext").addEventListener("click", () => run(async () => {
  await flushFocusedDraft();
  await navigateFocused(1);
}));
el("focusSubmitNext").addEventListener("click", () => run(async () => {
  await submitFocused();
  await navigateFocused(1);
}));
document.addEventListener("keydown", (event) => {
  if (!event.altKey || event.ctrlKey || event.metaKey || !state.selectedId || el("focusedEditor").hidden) return;
  const decisions = {"1": "posted", "2": "none", "3": "abstain"};
  if (!(event.key in decisions)) return;
  event.preventDefault();
  el("focusDecision").value = decisions[event.key];
  el("focusDecision").dispatchEvent(new Event("change"));
});
el("reviewerId").addEventListener("change", () => run(async () => {
  await flushFocusedDraft();
  state.selectedId = null;
  state.selectedRecord = null;
  state.selectedReviewer = null;
  await loadProgress();
  await resumeQueue();
}));
run(async () => {
  if (!el("reviewerId").value.trim()) return;
  await loadProgress();
  await resumeQueue();
});

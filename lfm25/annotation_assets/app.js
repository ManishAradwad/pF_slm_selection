"use strict";

const state = {
  csrf: null,
  row: null,
  position: 0,
  filter: "pending",
  saveTimer: null,
  saveInFlight: false,
  proposalInFlight: false,
  dirty: false,
  loading: false,
  mode: null,
  qcSession: false,
  spans: { amount: null, account: null, counterparty: null },
};

const byId = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const headers = { Accept: "application/json", ...(options.headers || {}) };
  if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";
    headers["X-Workbench-Token"] = state.csrf;
  }
  const response = await fetch(path, {
    ...options,
    headers,
    credentials: "same-origin",
    cache: "no-store",
  });
  const payload = await response.json().catch(() => ({ error: "Local request failed." }));
  if (!response.ok) {
    const error = new Error(payload.error || "Local request failed.");
    error.status = response.status;
    error.problems = payload.problems || [];
    throw error;
  }
  return payload;
}

function setSaveState(text, kind = "") {
  const element = byId("save-state");
  element.textContent = text;
  element.dataset.kind = kind;
}

function setGlobalError(message = "") {
  const element = byId("global-error");
  element.textContent = message;
  element.classList.toggle("hidden", !message);
}

function setFormBusy(busy) {
  const form = byId("annotation-form");
  form.inert = busy;
  form.setAttribute("aria-busy", String(busy));
}

function showProblems(problems = []) {
  const list = byId("validation-errors");
  list.replaceChildren();
  for (const problem of problems) {
    const item = document.createElement("li");
    item.textContent = problem.message || "Check this annotation field.";
    list.append(item);
  }
}

function selectedRadio(name) {
  return document.querySelector(`input[name="${name}"]:checked`)?.value || null;
}

function setRadio(name, value) {
  for (const input of document.querySelectorAll(`input[name="${name}"]`)) {
    input.checked = input.value === value;
  }
}

function annotationFromForm() {
  return {
    decision: selectedRadio("decision"),
    amount_decimal: byId("amount-decimal").value.trim() || null,
    amount_span: state.spans.amount,
    type: selectedRadio("type"),
    account_span: state.spans.account,
    counterparty_span: state.spans.counterparty,
    counterparty_absent: byId("counterparty-absent").checked,
    notes: byId("notes").value || null,
    uncertain: byId("uncertain").checked,
  };
}

function clearExtraction() {
  byId("amount-decimal").value = "";
  byId("amount-text").value = "";
  byId("account-text").value = "";
  byId("counterparty-text").value = "";
  byId("counterparty-absent").checked = false;
  setRadio("type", null);
  state.spans = { amount: null, account: null, counterparty: null };
}

function annotationToForm(annotation) {
  const value = annotation || {};
  setRadio("decision", value.decision || null);
  setRadio("type", value.type || null);
  byId("amount-decimal").value = value.amount_decimal || "";
  byId("amount-text").value = value.amount_span?.text || "";
  byId("account-text").value = value.account_span?.text || "";
  byId("counterparty-text").value = value.counterparty_span?.text || "";
  byId("counterparty-absent").checked = Boolean(value.counterparty_absent);
  byId("notes").value = value.notes || "";
  byId("uncertain").checked = Boolean(value.uncertain);
  state.spans = {
    amount: value.amount_span || null,
    account: value.account_span || null,
    counterparty: value.counterparty_span || null,
  };
  updateDecisionVisibility();
}

function updateDecisionVisibility() {
  const transaction = selectedRadio("decision") === "transaction";
  byId("transaction-fields").classList.toggle("hidden", !transaction);
  if (selectedRadio("decision") === "not_transaction") clearExtraction();
}

function selectionSpan() {
  const selection = window.getSelection();
  const smsElement = byId("sms");
  if (!selection || selection.rangeCount !== 1 || selection.isCollapsed) return null;
  const range = selection.getRangeAt(0);
  if (!smsElement.contains(range.commonAncestorContainer)) return null;
  const prefix = document.createRange();
  prefix.selectNodeContents(smsElement);
  prefix.setEnd(range.startContainer, range.startOffset);
  const selected = range.toString();
  if (!selected.trim()) return null;
  const before = prefix.toString();
  const encoder = new TextEncoder();
  return {
    text: selected,
    start: encoder.encode(before).length,
    end: encoder.encode(before + selected).length,
  };
}

function useSelection(field) {
  const span = selectionSpan();
  if (!span) {
    setGlobalError("Select exact source text inside the SMS first.");
    return;
  }
  setGlobalError();
  state.spans[field] = span;
  byId(`${field}-text`).value = span.text;
  if (field === "counterparty") byId("counterparty-absent").checked = false;
  scheduleAutosave();
}

function renderProgress(progress) {
  byId("completed").textContent = `${progress.completed_rows} / ${progress.total_rows}`;
  byId("batch-progress").textContent = `${progress.batch_completed} / ${progress.batch_size}`;
  byId("qc-progress").textContent = progress.qc_required
    ? `${progress.qc_passed} / ${progress.qc_required}`
    : "Not started";
}

function renderQueueTags(row) {
  const panel = byId("queue-tags-panel");
  const list = byId("queue-tags");
  list.replaceChildren();
  const tags = row?.status === "completed" && Array.isArray(row.queue_tags)
    ? row.queue_tags
    : [];
  for (const tag of tags) {
    if (typeof tag !== "string" || !tag) continue;
    const item = document.createElement("li");
    item.textContent = tag.replaceAll("_", " ");
    list.append(item);
  }
  panel.classList.toggle("hidden", tags.length === 0);
}

function spanSummary(span) {
  if (!span || typeof span !== "object") return "Not set";
  return '"' + String(span.text) + '" | bytes ' + String(span.start)
    + " - " + String(span.end);
}

function appendHistoryComponent(list, label, value) {
  const container = document.createElement("div");
  const term = document.createElement("dt");
  term.textContent = label;
  const description = document.createElement("dd");
  description.textContent = String(value ?? "Not set");
  container.append(term, description);
  list.append(container);
}

function renderHistoryEvent(event) {
  const item = document.createElement("li");
  item.className = "history-event";
  const heading = document.createElement("strong");
  heading.textContent = String(event.phase) + " | " + String(event.status)
    + " | " + String(event.recorded_at);
  item.append(heading);
  const annotation = event.annotation;
  if (!annotation || typeof annotation !== "object") return item;
  const components = document.createElement("dl");
  components.className = "history-components";
  const fields = [
    ["Decision", annotation.decision || "Not set"],
    ["Direction", annotation.type || "Not set"],
    ["Amount decimal", annotation.amount_decimal || "Not set"],
    ["Amount source span", spanSummary(annotation.amount_span)],
    ["Account source span", spanSummary(annotation.account_span)],
    [
      "Counterparty source span",
      annotation.counterparty_absent
        ? "Explicitly absent"
        : spanSummary(annotation.counterparty_span),
    ],
    ["Uncertain", annotation.uncertain ? "Yes" : "No"],
    ["Notes", annotation.notes || "Not set"],
  ];
  for (const [label, value] of fields) appendHistoryComponent(components, label, value);
  item.append(components);
  return item;
}

function renderRow(row) {
  state.row = row;
  state.position = row?.position || 0;
  state.dirty = false;
  renderQueueTags(row);
  const hasRow = Boolean(row);
  byId("row-card").classList.toggle("hidden", !hasRow);
  byId("empty-result").classList.toggle("hidden", hasRow);
  byId("proposal-content").textContent = "";
  byId("proposal-panel").classList.add("hidden");
  const historyAvailable = Boolean(row?.history_available);
  byId("history-panel").classList.toggle("hidden", !historyAvailable);
  byId("history").replaceChildren();
  if (!hasRow) return;
  byId("review-id").textContent = row.review_id;
  byId("row-position").textContent = `Row ${row.position + 1} of ${row.total_rows}`;
  byId("row-status").textContent = row.status;
  byId("sender").textContent = row.sender;
  byId("sms").textContent = row.sms;
  annotationToForm(row.annotation);
  showProblems();
  const revealButton = byId("reveal-proposals");
  revealButton.classList.toggle("hidden", !row.proposal_reveal_available);
  revealButton.disabled = state.proposalInFlight;
  if (historyAvailable) loadHistory(row.review_id);
  setSaveState("Saved", "saved");
}

async function loadHistory(reviewId) {
  const list = byId("history");
  list.replaceChildren();
  if (
    !state.row
    || state.row.review_id !== reviewId
    || state.row.history_available !== true
  ) return;
  try {
    const result = await api(`/api/history?review_id=${encodeURIComponent(reviewId)}`);
    if (
      !state.row
      || state.row.review_id !== reviewId
      || state.row.history_available !== true
    ) return;
    const events = Array.isArray(result.events) ? result.events : [];
    for (const event of events) list.append(renderHistoryEvent(event));
  } catch (error) {
    if (state.row?.review_id === reviewId) setGlobalError(error.message);
  }
}

async function navigate(direction = "current", filter = state.filter) {
  if (state.loading) return;
  if (state.dirty) {
    if (state.qcSession) {
      setGlobalError("Complete or correct the current QC row before navigating.");
      return;
    }
    const saved = await save(false);
    if (!saved) return;
  }
  state.loading = true;
  setGlobalError();
  try {
    const params = new URLSearchParams({
      position: String(state.position),
      direction,
      filter,
    });
    const result = await api(`/api/row?${params}`);
    state.filter = filter;
    renderRow(result.row);
    renderProgress(result.progress);
  } catch (error) {
    setGlobalError(error.message);
  } finally {
    state.loading = false;
  }
}

async function save(submit = false) {
  if (!state.row || state.loading || state.saveInFlight) return false;
  if (state.qcSession && !submit) {
    clearTimeout(state.saveTimer);
    setSaveState("QC changes save only on completion", "dirty");
    return false;
  }
  const endpoint = state.qcSession ? "/api/qc/submit" : "/api/save";
  clearTimeout(state.saveTimer);
  setSaveState("Saving…", "saving");
  state.saveInFlight = true;
  setFormBusy(true);
  showProblems();
  try {
    const result = await api(endpoint, {
      method: "POST",
      body: JSON.stringify({
        review_id: state.row.review_id,
        expected_revision: state.row.revision,
        annotation: annotationFromForm(),
        submit,
      }),
    });
    renderRow(result.row);
    renderProgress(result.progress);
    setSaveState("Saved", "saved");
    if (submit && result.next_available) await navigate("next", state.filter);
    return true;
  } catch (error) {
    showProblems(error.problems);
    if (error.status === 409) {
      state.dirty = true;
    }
    setSaveState("Save failed", "error");
    setGlobalError(error.message);
    return false;
  } finally {
    state.saveInFlight = false;
    setFormBusy(false);
  }
}

function scheduleAutosave() {
  if (!state.row) return;
  state.dirty = true;
  byId("proposal-panel").classList.add("hidden");
  byId("proposal-content").textContent = "";
  if (state.qcSession) {
    clearTimeout(state.saveTimer);
    setSaveState("QC changes save only on completion", "dirty");
    return;
  }
  clearTimeout(state.saveTimer);
  setSaveState("Unsaved", "dirty");
  state.saveTimer = setTimeout(() => save(false), 750);
}

function configurePhase(filters) {
  const allowed = new Set(filters);
  for (const option of byId("filter").options) {
    const visible = allowed.has(option.value);
    option.hidden = !visible;
    option.disabled = !visible;
  }
  byId("save-draft").disabled = state.qcSession;
  byId("start-qc").classList.toggle("hidden", state.qcSession);
}

async function startQc() {
  if (state.loading || state.saveInFlight) return;
  if (state.dirty) {
    const saved = await save(false);
    if (!saved) return;
  }
  state.loading = true;
  setFormBusy(true);
  let started = false;
  try {
    const result = await api("/api/qc/start", {
      method: "POST",
      body: JSON.stringify({}),
    });
    state.qcSession = true;
    state.filter = "qc";
    byId("filter").value = "qc";
    configurePhase(["pending", "completed", "uncertain", "qc"]);
    renderProgress(result.progress);
    started = true;
  } catch (error) {
    setGlobalError(error.message);
  } finally {
    state.loading = false;
    setFormBusy(false);
  }
  if (started) await navigate("first", "qc");
}

async function revealProposals() {
  if (
    !state.row
    || !state.row.proposal_reveal_available
    || state.loading
    || state.dirty
    || state.saveInFlight
    || state.proposalInFlight
  ) {
    if (state.dirty) setGlobalError("Save the current annotation before revealing proposals.");
    return;
  }
  const reviewId = state.row.review_id;
  const button = byId("reveal-proposals");
  state.proposalInFlight = true;
  button.disabled = true;
  setGlobalError();
  try {
    const result = await api("/api/proposals/reveal", {
      method: "POST",
      body: JSON.stringify({ review_id: reviewId }),
    });
    if (
      !state.row
      || state.row.review_id !== reviewId
      || state.loading
      || state.dirty
      || state.saveInFlight
    ) return;
    byId("proposal-content").textContent = JSON.stringify(result.proposals, null, 2);
    byId("proposal-panel").classList.remove("hidden");
  } catch (error) {
    if (state.row?.review_id === reviewId) setGlobalError(error.message);
  } finally {
    state.proposalInFlight = false;
    button.disabled = false;
  }
}

function editableTarget(target) {
  return target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement
    || target instanceof HTMLSelectElement;
}

function handleShortcut(event) {
  if (state.loading || state.saveInFlight) return;
  if (event.ctrlKey && event.key.toLowerCase() === "s") {
    event.preventDefault();
    save(true);
    return;
  }
  if (event.altKey) {
    const key = event.key.toLowerCase();
    if (["a", "c", "p", "0"].includes(key)) event.preventDefault();
    if (key === "a") useSelection("amount");
    if (key === "c") useSelection("account");
    if (key === "p") useSelection("counterparty");
    if (key === "0") {
      byId("counterparty-absent").checked = true;
      state.spans.counterparty = null;
      byId("counterparty-text").value = "";
      scheduleAutosave();
    }
    return;
  }
  if (editableTarget(event.target)) return;
  const key = event.key.toLowerCase();
  if (key === "j") navigate("next");
  if (key === "k") navigate("previous");
  if (key === "n") navigate("next", "pending");
  if (key === "t" || key === "x") {
    setRadio("decision", key === "t" ? "transaction" : "not_transaction");
    updateDecisionVisibility();
    scheduleAutosave();
  }
  if (key === "u") {
    byId("uncertain").checked = !byId("uncertain").checked;
    scheduleAutosave();
  }
}

function wireUi() {
  byId("previous").addEventListener("click", () => navigate("previous"));
  byId("next").addEventListener("click", () => navigate("next"));
  byId("next-pending").addEventListener("click", () => navigate("next", "pending"));
  byId("filter").addEventListener("change", (event) => navigate("first", event.target.value));
  byId("use-amount").addEventListener("click", () => useSelection("amount"));
  byId("use-account").addEventListener("click", () => useSelection("account"));
  byId("use-counterparty").addEventListener("click", () => useSelection("counterparty"));
  byId("save-draft").addEventListener("click", () => save(false));
  byId("start-qc").addEventListener("click", startQc);
  byId("reveal-proposals").addEventListener("click", revealProposals);
  byId("annotation-form").addEventListener("submit", (event) => {
    event.preventDefault();
    save(true);
  });
  for (const input of byId("annotation-form").querySelectorAll("input, textarea")) {
    input.addEventListener("input", scheduleAutosave);
    input.addEventListener("change", scheduleAutosave);
  }
  for (const input of document.querySelectorAll('input[name="decision"]')) {
    input.addEventListener("change", updateDecisionVisibility);
  }
  byId("counterparty-absent").addEventListener("change", () => {
    if (byId("counterparty-absent").checked) {
      state.spans.counterparty = null;
      byId("counterparty-text").value = "";
    }
  });
  document.addEventListener("keydown", handleShortcut);
}

async function bootstrap() {
  wireUi();
  try {
    const result = await api("/api/bootstrap");
    state.csrf = result.csrf_token;
    state.mode = result.mode;
    state.qcSession = result.session_phase === "qc";
    configurePhase(result.filters);
    byId("mode").textContent = result.mode_label;
    byId("reviewer").textContent = result.reviewer;
    byId("reveal-proposals").classList.toggle("hidden", result.mode !== "training_curation");
    renderProgress(result.progress);
    await navigate("first", state.qcSession ? "qc" : "pending");
  } catch (error) {
    setSaveState("Unavailable", "error");
    setGlobalError(error.message);
  }
}

bootstrap();

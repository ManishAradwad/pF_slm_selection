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
  suggestedFields: new Set(),
  bootstrapped: false,
};

const byId = (id) => document.getElementById(id);

function trustedLocalRuntime() {
  return window.location.protocol === "http:"
    && window.location.hostname === "127.0.0.1";
}

function setLaunchFailure(message) {
  byId("launch-reason").textContent = message;
  byId("launch-gate").hidden = false;
  const shell = byId("app-shell");
  shell.hidden = true;
  shell.inert = true;
  shell.setAttribute("inert", "");
}

function revealAppShell() {
  byId("launch-gate").hidden = true;
  const shell = byId("app-shell");
  shell.hidden = false;
  shell.inert = false;
  shell.removeAttribute("inert");
  state.bootstrapped = true;
}

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

const PREFILL_ELEMENTS = {
  amount_decimal: "amount-decimal",
  amount_span: "amount-text",
  type: "direction-fieldset",
  account_span: "account-text",
  counterparty_span: "counterparty-text",
};

function syncSuggestionMarkers() {
  for (const [field, elementId] of Object.entries(PREFILL_ELEMENTS)) {
    const element = byId(elementId);
    const suggested = state.suggestedFields.has(field);
    element.toggleAttribute("data-suggested", suggested);
    if (suggested) element.dataset.suggested = "true";
  }
  byId("prefill-note").classList.toggle("hidden", state.suggestedFields.size === 0);
}

function clearSuggestedField(field) {
  state.suggestedFields.delete(field);
  syncSuggestionMarkers();
}

function resetSuggestedFields() {
  state.suggestedFields.clear();
  syncSuggestionMarkers();
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
  byId("selection-status").textContent = "";
  resetSuggestedFields();
}

function annotationToForm(annotation) {
  resetSuggestedFields();
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

function validPrefillSpan(value, sms) {
  if (
    !value
    || typeof value !== "object"
    || Array.isArray(value)
    || typeof value.text !== "string"
    || !Number.isInteger(value.start)
    || !Number.isInteger(value.end)
    || value.start < 0
    || value.end <= value.start
  ) return false;
  const keys = Object.keys(value).sort().join(",");
  if (keys !== "end,start,text") return false;
  const encoded = new TextEncoder().encode(sms);
  if (value.end > encoded.length) return false;
  try {
    const decoded = new TextDecoder("utf-8", { fatal: true })
      .decode(encoded.slice(value.start, value.end));
    return decoded === value.text;
  } catch {
    return false;
  }
}

function sanitizedSourcePrefill(row) {
  const value = row?.source_prefill;
  if (
    !value
    || typeof value !== "object"
    || Array.isArray(value)
  ) return null;
  const allowed = new Set([
    "policy_version",
    "amount_decimal",
    "amount_span",
    "type",
    "account_span",
    "counterparty_span",
  ]);
  const keys = Object.keys(value);
  const hasAmount = Object.hasOwn(value, "amount_decimal");
  const hasAmountSpan = Object.hasOwn(value, "amount_span");
  if (
    value.policy_version !== 1
    || keys.length === 1
    || keys.some((key) => !allowed.has(key))
    || hasAmount !== hasAmountSpan
  ) return null;
  const result = { policy_version: 1 };
  if (Object.hasOwn(value, "amount_decimal")) {
    const amount = value.amount_decimal;
    if (
      typeof amount !== "string"
      || !/^(?:0|[1-9]\d*)(?:\.\d+)?$/.test(amount)
      || !/[1-9]/.test(amount)
    ) return null;
    result.amount_decimal = amount;
  }
  if (Object.hasOwn(value, "type")) {
    if (!["debit", "credit"].includes(value.type)) return null;
    result.type = value.type;
  }
  for (const field of ["amount_span", "account_span", "counterparty_span"]) {
    if (!Object.hasOwn(value, field)) continue;
    if (!validPrefillSpan(value[field], row.sms)) return null;
    result[field] = { ...value[field] };
  }
  return result;
}

function applySourcePrefill() {
  const row = state.row;
  if (
    !row
    || !["pending", "draft"].includes(row.status)
    || state.qcSession
    || selectedRadio("decision") !== "transaction"
  ) return;
  const value = sanitizedSourcePrefill(row);
  if (!value) return;
  const applied = [];
  if (value.amount_decimal && !byId("amount-decimal").value.trim()) {
    byId("amount-decimal").value = value.amount_decimal;
    applied.push("amount_decimal");
  }
  if (value.amount_span && state.spans.amount === null) {
    state.spans.amount = value.amount_span;
    byId("amount-text").value = value.amount_span.text;
    applied.push("amount_span");
  }
  if (value.type && selectedRadio("type") === null) {
    setRadio("type", value.type);
    applied.push("type");
  }
  if (value.account_span && state.spans.account === null) {
    state.spans.account = value.account_span;
    byId("account-text").value = value.account_span.text;
    applied.push("account_span");
  }
  if (
    value.counterparty_span
    && state.spans.counterparty === null
    && !byId("counterparty-absent").checked
  ) {
    state.spans.counterparty = value.counterparty_span;
    byId("counterparty-text").value = value.counterparty_span.text;
    applied.push("counterparty_span");
  }
  for (const field of applied) state.suggestedFields.add(field);
  syncSuggestionMarkers();
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

function canonicalAmountFromSelection(text) {
  const matches = [...text.matchAll(/[+]?\d[\d,]*(?:\.\d+)?/g)];
  if (matches.length !== 1) return null;
  const match = matches[0];
  const prefix = text.slice(0, match.index).trimEnd();
  if (prefix.endsWith("-")) return null;
  const token = match[0].replace(/^\+/, "");
  const [whole, fraction] = token.split(".");
  if (token.split(".").length > 2) return null;
  if (whole.includes(",")) {
    const western = /^\d{1,3}(?:,\d{3})+$/.test(whole);
    const indian = /^\d{1,2}(?:,\d{2})*,\d{3}$/.test(whole);
    if (!western && !indian) return null;
  }
  const digits = whole.replaceAll(",", "");
  if (!/^\d+$/.test(digits) || (fraction !== undefined && !/^\d+$/.test(fraction))) {
    return null;
  }
  if (!/[1-9]/.test(digits + (fraction || ""))) return null;
  const canonicalWhole = digits.replace(/^0+(?=\d)/, "");
  return canonicalWhole + (fraction === undefined ? "" : "." + fraction);
}

function useSelection(field) {
  const span = selectionSpan();
  if (!span) {
    setGlobalError("Select exact source text inside the SMS first.");
    byId("selection-status").textContent = "";
    return;
  }
  setGlobalError();
  clearSuggestedField(`${field}_span`);
  state.spans[field] = span;
  byId(`${field}-text`).value = span.text;
  if (field === "amount") {
    const canonical = canonicalAmountFromSelection(span.text);
    if (canonical) {
      byId("amount-decimal").value = canonical;
      clearSuggestedField("amount_decimal");
      byId("selection-status").textContent = `Amount set to ${canonical}; verify before save.`;
    } else {
      byId("selection-status").textContent =
        "Amount span assigned; enter the exact decimal manually.";
    }
  } else {
    byId("selection-status").textContent = `Selected source assigned to ${field}.`;
  }
  if (field === "counterparty") {
    byId("counterparty-absent").checked = false;
  }
  scheduleAutosave();
}

function renderProgress(progress) {
  byId("completed").textContent = `${progress.completed_rows} / ${progress.total_rows}`;
  byId("batch-progress").textContent = `${progress.batch_completed} / ${progress.batch_size}`;
  byId("qc-progress").textContent = progress.qc_required
    ? `${progress.qc_passed} / ${progress.qc_required}`
    : "Not started";
  const ready = progress.ready_for_qc === true;
  const button = byId("start-qc");
  button.disabled = !ready || state.qcSession;
  const availability = byId("qc-availability");
  if (state.qcSession) {
    availability.textContent = "Delayed QC is in progress.";
  } else if (ready) {
    availability.textContent = "All rows are complete and resolved. QC can now start.";
  } else {
    availability.textContent =
      `${progress.pending_rows} pending and ${progress.uncertain_rows} unresolved row(s) remain.`;
  }
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
  byId("selection-status").textContent = "";
  annotationToForm(row.annotation);
  if (selectedRadio("decision") === "transaction") applySourcePrefill();
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
  if (state.loading) return false;
  if (state.dirty) {
    if (state.qcSession) {
      setGlobalError("Complete or correct the current QC row before navigating.");
      return false;
    }
    const saved = await save(false);
    if (!saved) return false;
  }
  state.loading = true;
  setGlobalError();
  let loaded = false;
  try {
    const params = new URLSearchParams({
      position: String(state.position),
      direction,
      filter,
    });
    const result = await api(`/api/row?${params}`);
    state.filter = filter;
    byId("filter").value = filter;
    renderRow(result.row);
    renderProgress(result.progress);
    loaded = true;
  } catch (error) {
    byId("filter").value = state.filter;
    setGlobalError(error.message);
  } finally {
    state.loading = false;
  }
  return loaded;
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
  if (state.loading || state.saveInFlight || byId("start-qc").disabled) return;
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
      clearSuggestedField("counterparty_span");
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
    if (key === "t") applySourcePrefill();
    scheduleAutosave();
  }
  if (key === "u") {
    byId("uncertain").checked = !byId("uncertain").checked;
    scheduleAutosave();
  }
}

function recordHumanEdit(input) {
  const field = input.dataset.prefillField
    || (input.name === "type" ? "type" : null);
  if (field) clearSuggestedField(field);
  scheduleAutosave();
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
  for (const input of document.querySelectorAll('input[name="decision"]')) {
    input.addEventListener("change", () => {
      updateDecisionVisibility();
      if (input.checked && input.value === "transaction") applySourcePrefill();
    });
  }
  for (const input of byId("annotation-form").querySelectorAll("input, textarea")) {
    input.addEventListener("input", () => recordHumanEdit(input));
    input.addEventListener("change", () => recordHumanEdit(input));
  }
  byId("counterparty-absent").addEventListener("change", () => {
    if (byId("counterparty-absent").checked) {
      state.spans.counterparty = null;
      byId("counterparty-text").value = "";
    }
    clearSuggestedField("counterparty_span");
  });
  document.addEventListener("keydown", handleShortcut);
  window.addEventListener("beforeunload", (event) => {
    if (!state.dirty) return;
    event.preventDefault();
    event.returnValue = "";
  });
}

async function bootstrap() {
  if (!trustedLocalRuntime()) {
    setLaunchFailure(
      "No SMS loaded. This file is not the running app. Start the local server "
        + "with the command below, then open its exact http://127.0.0.1 URL.",
    );
    return;
  }
  wireUi();
  try {
    const result = await api("/api/bootstrap");
    state.csrf = result.csrf_token;
    state.mode = result.mode;
    state.qcSession = result.session_phase === "qc";
    const initialFilter = state.qcSession ? "qc" : "pending";
    state.filter = initialFilter;
    configurePhase(result.filters);
    byId("filter").value = initialFilter;
    byId("mode").textContent = result.mode_label;
    byId("reviewer").textContent = result.reviewer;
    byId("reveal-proposals").classList.toggle("hidden", result.mode !== "training_curation");
    renderProgress(result.progress);
    const loaded = await navigate("first", initialFilter);
    if (!loaded) throw new Error("The first local row could not be loaded.");
    revealAppShell();
  } catch (error) {
    setLaunchFailure(
      "No SMS loaded. The local workbench could not bootstrap. "
        + "Confirm the server is still running, then reopen the exact URL it printed. "
        + String(error.message || ""),
    );
  }
}

bootstrap();

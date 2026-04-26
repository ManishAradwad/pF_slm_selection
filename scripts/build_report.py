"""
Build a single self-contained HTML report from RESULTS/llamacpp/.

Walks every results_<ts>.json + matching samples_<ts>.jsonl, groups by
(HF model id, quant), and emits RESULTS/report.html with:
  1. Aggregate metrics table (one row per model+quant; best per column highlighted)
  2. Failure-mode badge (over-rejects / ghost-prone / leaks-few-shot / clean)
  3. Side-by-side viewer: same SMS shown across all models (Q4_K_M as default)
  4. Per-model quirks blurbs
  5. Run metadata footer

Run after the smoke and/or full passes; pass --include-archives to merge older
result directories. Default: only RESULTS/llamacpp/.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Per-model curated quirks blurbs ────────────────────────────────────────────
MODEL_QUIRKS = {
    "google/gemma-3-270m-it": (
        "Gemma's smallest 270M instruct. No thinking. Behaves more like a "
        "completion model than a reasoner — at this size it tends to copy "
        "whichever few-shot answer most closely matches the input shape."
    ),
    "google/gemma-4-E2B-it": (
        "2B-class instruct model. No thinking. Strong general-purpose "
        "extractor; baseline for the slate. Stable across quants down to Q3."
    ),
    "Qwen/Qwen3-0.6B": (
        "Hybrid thinking model. Two-phase generation: phase 1 emits "
        "&lt;think&gt;…&lt;/think&gt; reasoning, phase 2 emits grammar-constrained "
        "JSON. At 0.6B, thinking is shallow and frequently inconclusive."
    ),
    "Qwen/Qwen3.5-0.8B": (
        "Newer Qwen3.5 generation; same thinking architecture as Qwen3 with "
        "larger thinking traces. Two-phase generation applies."
    ),
    "Qwen/Qwen3-1.7B": (
        "1.7B thinking model. Without two-phase generation the GBNF grammar "
        "suppresses &lt;think&gt; tokens and the model collapses into copying "
        "few-shot answers (23.6% leakage observed). With two-phase + a strong "
        "task delimiter, it recovers fully."
    ),
    "LiquidAI/LFM2.5-1.2B-Instruct": (
        "Liquid's hybrid Mamba/Attention 1.2B. Uses the &lt;|im_start|&gt; chat "
        "template with a &lt;|startoftext|&gt; BOS. The chat template references "
        "&lt;/think&gt; only as a no-op past-message stripper, so thinking is off."
    ),
    "arcee-ai/arcee-lite": (
        "Arcee's 2B chat-tuned model. No thinking. Tends to be enthusiastic "
        "about extracting fields even from non-transactional SMS — the "
        "extract_json_nonnull filter is the main lever for it."
    ),
}

# ── Headline metric set we surface ─────────────────────────────────────────────
HEADLINE_METRICS = [
    ("full_match_accuracy",       "Full match",        True),   # higher better
    ("ghost_transaction_rate",    "Ghost rate",        False),  # lower better
    ("missed_transaction_rate",   "Missed rate",       False),
    ("few_shot_leakage_rate",     "FS leakage",        False),
    ("amount_accuracy",           "Amount",            True),
    ("type_accuracy",             "Type",              True),
    ("account_accuracy",          "Account",           True),
    ("merchant_accuracy",         "Merchant",          True),
    ("date_accuracy",             "Date",              True),
    ("json_validity",             "JSON valid",        True),
]


@dataclass
class RunRecord:
    """One eval run = one (model, quant, timestamp) tuple."""
    timestamp: str
    model_id: str       # HF id, e.g. "Qwen/Qwen3-1.7B"
    model_slug: str     # e.g. "Qwen__Qwen3-1.7B"
    gguf_path: str
    quant: str          # e.g. "Q4_K_M"
    n_ctx: int
    n_samples: int
    metrics: dict       # nonnull-filtered headline metrics
    raw_metrics: dict   # extract_json (raw filter) headline metrics
    samples_path: str   # full path to samples jsonl


def _parse_quant_from_path(p: str) -> str:
    """Extract the quant tag (Q4_K_M, IQ4_XS, …) from a gguf filename."""
    m = re.search(r"-((?:I?Q[0-9]_(?:K_)?[A-Z]+)|Q[0-9]_[0-9]|F16|BF16)\.gguf$", p)
    return m.group(1) if m else "?"


def _model_id_from_slug(slug: str) -> str:
    return slug.replace("__", "/")


def _walk_results(roots: list[Path]) -> list[RunRecord]:
    runs: list[RunRecord] = []
    for root in roots:
        if not root.exists():
            continue
        for slug_dir in sorted(root.iterdir()):
            if not slug_dir.is_dir():
                continue
            for f in sorted(slug_dir.iterdir()):
                m = re.match(r"results_(.+)\.json$", f.name)
                if not m:
                    continue
                ts = m.group(1)
                samples_path = slug_dir / f"samples_sms_extraction_{ts}.jsonl"
                if not samples_path.exists():
                    continue
                try:
                    data = json.loads(f.read_text())
                except Exception:
                    continue
                cfg = data.get("config", {})
                model_id = cfg.get("model_args", {}).get("tokenizer") or _model_id_from_slug(slug_dir.name)
                gguf = cfg.get("model_args", {}).get("path", "")
                quant = _parse_quant_from_path(gguf)
                results = data.get("results", {}).get("sms_extraction", {})
                if not results:
                    continue
                metrics = {k: results.get(f"{k},extract_json_nonnull", 0.0)
                           for k, *_ in [m for m in HEADLINE_METRICS]}
                raw_metrics = {k: results.get(f"{k},extract_json", 0.0)
                               for k, *_ in [m for m in HEADLINE_METRICS]}
                runs.append(RunRecord(
                    timestamp=ts,
                    model_id=model_id,
                    model_slug=slug_dir.name,
                    gguf_path=gguf,
                    quant=quant,
                    n_ctx=int(cfg.get("model_args", {}).get("n_ctx") or 0),
                    n_samples=int(results.get("sample_len", 0)),
                    metrics=metrics,
                    raw_metrics=raw_metrics,
                    samples_path=str(samples_path),
                ))
    return runs


def _latest_per_pair(runs: list[RunRecord]) -> list[RunRecord]:
    """For each (model_id, quant) keep only the most recent run."""
    by_pair: dict[tuple[str, str], RunRecord] = {}
    for r in sorted(runs, key=lambda r: r.timestamp):
        by_pair[(r.model_id, r.quant)] = r
    return list(by_pair.values())


def _classify(metrics: dict) -> list[str]:
    """Auto-classify failure modes from the nonnull-filtered metrics."""
    badges = []
    if metrics.get("missed_transaction_rate", 0) > 0.30:
        badges.append("over-rejects")
    if metrics.get("ghost_transaction_rate", 0) > 0.20:
        badges.append("ghost-prone")
    if metrics.get("few_shot_leakage_rate", 0) > 0.05:
        badges.append("leaks-few-shot")
    if not badges and metrics.get("full_match_accuracy", 0) > 0.7:
        badges.append("clean")
    if not badges:
        badges.append("mid")
    return badges


def _file_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / 1024 / 1024
    except OSError:
        return 0.0


def _select_doc_ids(runs: list[RunRecord], k: int = 8) -> list[int]:
    """Pick representative doc_ids: a mix of real txn and null-gold cases.
    Uses the doc IDs available in the first run's samples file."""
    if not runs:
        return []
    samples = []
    with open(runs[0].samples_path) as f:
        for line in f:
            samples.append(json.loads(line))
    real = [s for s in samples if (s.get("target") or "").strip() != "null"]
    nulls = [s for s in samples if (s.get("target") or "").strip() == "null"]
    pick = []
    # interleave real + null so the viewer alternates
    for i in range(max(len(real), len(nulls))):
        if i < len(real): pick.append(real[i]["doc_id"])
        if i < len(nulls): pick.append(nulls[i]["doc_id"])
        if len(pick) >= k: break
    return pick[:k]


def _load_samples_subset(path: str, doc_ids: set[int]) -> dict[int, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            if s["doc_id"] in doc_ids:
                out[s["doc_id"]] = s
    return out


# ── HTML rendering ─────────────────────────────────────────────────────────────

CSS = """
body { font: 14px/1.4 system-ui, -apple-system, sans-serif; margin: 0; background: #fafafa; color: #222; }
header { background: #1f2937; color: #fff; padding: 18px 24px; }
header h1 { margin: 0 0 4px 0; font-size: 22px; }
header p { margin: 0; opacity: 0.8; font-size: 13px; }
main { padding: 18px 24px; max-width: 1500px; margin: 0 auto; }
section { margin-bottom: 32px; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px 20px; }
section h2 { margin: 0 0 14px 0; font-size: 18px; border-bottom: 1px solid #e5e7eb; padding-bottom: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 6px 9px; text-align: right; border-bottom: 1px solid #f0f0f0; }
th { background: #f3f4f6; font-weight: 600; text-align: right; }
th:first-child, td:first-child, th.left, td.left { text-align: left; }
tr:hover { background: #fafbfc; }
td.best { background: #ecfdf5; font-weight: 600; color: #047857; }
td.worst { background: #fef2f2; color: #b91c1c; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-right: 4px; font-weight: 600; }
.badge-clean { background: #d1fae5; color: #065f46; }
.badge-mid { background: #e0e7ff; color: #3730a3; }
.badge-over-rejects { background: #fee2e2; color: #991b1b; }
.badge-ghost-prone { background: #fef3c7; color: #92400e; }
.badge-leaks-few-shot { background: #fbcfe8; color: #9d174d; }
.viewer { display: grid; grid-template-columns: 320px repeat(var(--cols), 1fr); gap: 12px; align-items: start; font-size: 12px; }
.viewer .col-head { font-weight: 600; padding: 8px; background: #f3f4f6; border-radius: 4px; text-align: center; }
.viewer .sms { background: #fff; padding: 10px 12px; border: 1px solid #e5e7eb; border-radius: 4px; }
.viewer .sms .sender { color: #6b7280; font-size: 11px; }
.viewer .sms .body { white-space: pre-wrap; }
.viewer .sms .target { background: #ecfdf5; padding: 6px; border-radius: 3px; margin-top: 6px; font-family: ui-monospace, monospace; font-size: 11px; word-break: break-all; }
.viewer .resp { padding: 8px; background: #fafbfc; border: 1px solid #f0f0f0; border-radius: 4px; font-family: ui-monospace, monospace; font-size: 11px; white-space: pre-wrap; word-break: break-all; max-height: 300px; overflow-y: auto; }
.viewer .resp.match { border-left: 3px solid #10b981; }
.viewer .resp.miss { border-left: 3px solid #ef4444; }
.thinking { color: #6b7280; font-style: italic; font-size: 10px; max-height: 80px; overflow-y: auto; padding: 4px; background: #f9fafb; border-radius: 3px; margin-bottom: 4px; }
.thinking::before { content: "<think> "; opacity: 0.5; }
.quirks { background: #fef9c3; padding: 10px 14px; border-radius: 6px; margin: 8px 0; border-left: 4px solid #ca8a04; font-size: 13px; }
.quirks strong { color: #92400e; display: block; margin-bottom: 2px; }
.size-hint { color: #6b7280; font-size: 11px; }
"""


def _fmt_metric(v: float, name: str, higher_is_better: bool, best: float, worst: float) -> str:
    cls = ""
    if abs(v - best) < 1e-9:
        cls = "best"
    elif abs(v - worst) < 1e-9:
        cls = "worst"
    return f'<td class="{cls}">{v:.3f}</td>'


def _render_metrics_table(runs: list[RunRecord]) -> str:
    if not runs:
        return "<p>(no runs)</p>"
    # compute best/worst per metric
    best_per_metric = {}
    worst_per_metric = {}
    for name, _, higher in HEADLINE_METRICS:
        vals = [r.metrics.get(name, 0.0) for r in runs]
        if higher:
            best_per_metric[name] = max(vals)
            worst_per_metric[name] = min(vals)
        else:
            best_per_metric[name] = min(vals)
            worst_per_metric[name] = max(vals)

    rows = []
    sorted_runs = sorted(runs, key=lambda r: -r.metrics.get("full_match_accuracy", 0.0))
    for r in sorted_runs:
        size_mb = _file_size_mb(r.gguf_path)
        badges = " ".join(f'<span class="badge badge-{b}">{b}</span>' for b in _classify(r.metrics))
        cells = []
        for name, _, higher in HEADLINE_METRICS:
            v = r.metrics.get(name, 0.0)
            cells.append(_fmt_metric(v, name, higher, best_per_metric[name], worst_per_metric[name]))
        rows.append(
            f'<tr><td class="left">{html.escape(r.model_id)}</td>'
            f'<td class="left">{html.escape(r.quant)}</td>'
            f'<td class="left"><span class="size-hint">{size_mb:,.0f} MB</span></td>'
            f'<td class="left">{badges}</td>'
            + "".join(cells)
            + f'<td class="left"><span class="size-hint">{r.n_samples} samples</span></td></tr>'
        )

    headers = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in HEADLINE_METRICS)
    return (
        '<table><thead><tr>'
        '<th class="left">Model</th><th class="left">Quant</th>'
        '<th class="left">Size</th><th class="left">Mode</th>'
        + headers +
        '<th class="left">N</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )


def _strip_thinking(resp: str) -> tuple[str, str]:
    """Return (thinking, after) — empty thinking string if absent."""
    m = re.match(r"\s*<think>(.*?)</think>\s*(.*)", resp, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", resp.strip()


def _render_viewer(runs: list[RunRecord], doc_ids: list[int]) -> str:
    if not runs or not doc_ids:
        return "<p>(no runs)</p>"

    # Pick one quant per HF model — Q4_K_M preferred, else first available.
    by_model: dict[str, RunRecord] = {}
    for r in sorted(runs, key=lambda r: (r.model_id, r.quant != "Q4_K_M", r.quant)):
        by_model.setdefault(r.model_id, r)

    samples_by_model: dict[str, dict[int, dict]] = {}
    for model_id, r in by_model.items():
        samples_by_model[model_id] = _load_samples_subset(r.samples_path, set(doc_ids))

    # Common prompt (all share the same dataset) — pull from any first run.
    any_run_samples = next(iter(samples_by_model.values()))

    cols = list(by_model.keys())
    head = (
        '<div class="col-head">SMS</div>' +
        "".join(f'<div class="col-head">{html.escape(m.split("/")[-1])}<br><span class="size-hint">{html.escape(by_model[m].quant)}</span></div>' for m in cols)
    )

    rows_html = []
    for did in doc_ids:
        sample = any_run_samples.get(did)
        if sample is None:
            continue
        doc = sample.get("doc", {})
        target = sample.get("target", "")
        sms_cell = (
            '<div class="sms">'
            f'<div class="sender">{html.escape(str(doc.get("sender","")))}</div>'
            f'<div class="body">{html.escape(str(doc.get("sms","")))}</div>'
            f'<div class="target"><strong>target:</strong> {html.escape(target)}</div>'
            "</div>"
        )

        cells = [sms_cell]
        for model_id in cols:
            s = samples_by_model.get(model_id, {}).get(did)
            if s is None:
                cells.append('<div class="resp">(no sample)</div>')
                continue
            raw = s.get("resps", [[""]])[0][0]
            think, after = _strip_thinking(raw)
            filtered = (s.get("filtered_resps") or [""])[0]
            full_match = bool(s.get("full_match_accuracy", 0.0) >= 0.999)
            cls = "match" if full_match else "miss"
            think_html = f'<div class="thinking">{html.escape(think[:600])}{"…" if len(think) > 600 else ""}</div>' if think else ""
            cells.append(
                f'<div class="resp {cls}">{think_html}'
                f'<strong>{html.escape((filtered or "").strip()[:250])}</strong></div>'
            )

        rows_html.append("".join(cells))

    body = "".join(rows_html)
    n_cols = len(cols) + 1
    return f'<div class="viewer" style="--cols: {len(cols)}">{head}{body}</div>'


def _render_quirks() -> str:
    blocks = []
    for model_id, blurb in MODEL_QUIRKS.items():
        blocks.append(
            f'<div class="quirks"><strong>{html.escape(model_id)}</strong>{blurb}</div>'
        )
    return "".join(blocks)


def build_html(runs: list[RunRecord]) -> str:
    runs_latest = _latest_per_pair(runs)
    doc_ids = _select_doc_ids(runs_latest, k=8)

    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_runs = len(runs_latest)
    n_models = len({r.model_id for r in runs_latest})

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SLM slate report — pocket-financer</title>
<style>{CSS}</style></head><body>
<header>
  <h1>SLM slate report — pocket-financer SMS extraction</h1>
  <p>Generated {now} · {n_runs} runs across {n_models} models · llama.cpp + GBNF grammar · 203-sample test set</p>
</header>
<main>
  <section>
    <h2>Aggregate metrics — best per column highlighted</h2>
    {_render_metrics_table(runs_latest)}
    <p class="size-hint" style="margin-top:8px">All metrics are computed against the <em>extract_json_nonnull</em> filter
    (the production rule: any output with null in amount/type/account is rejected as null).
    Higher is better for accuracy metrics; lower is better for ghost / missed / leakage.</p>
  </section>

  <section>
    <h2>Side-by-side response viewer (Q4_K_M per model where available)</h2>
    <p class="size-hint" style="margin:0 0 14px 0">Same SMS shown across all models. Green border = full match, red = miss.
    Thinking traces from Qwen3 are shown in collapsed grey boxes.</p>
    {_render_viewer(runs_latest, doc_ids)}
  </section>

  <section>
    <h2>Per-model quirks</h2>
    {_render_quirks()}
  </section>
</main>
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default=str(REPO_ROOT / "RESULTS" / "llamacpp"),
        help="Directory containing <model_slug>/results_*.json",
    )
    parser.add_argument(
        "--include-archives",
        action="store_true",
        help="Also include RESULTS/llamacpp_pre_thinking_*/ directories.",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "RESULTS" / "report.html"),
        help="Output HTML path.",
    )
    args = parser.parse_args()

    roots = [Path(args.results_root)]
    if args.include_archives:
        for d in sorted((REPO_ROOT / "RESULTS").iterdir()):
            if d.is_dir() and d.name.startswith("llamacpp_pre"):
                roots.append(d)

    runs = _walk_results(roots)
    if not runs:
        print("[build_report] no runs found under:", roots)
        return 1
    html_text = build_html(runs)
    Path(args.out).write_text(html_text)
    print(f"[build_report] wrote {args.out} ({len(runs)} runs latest-per-pair)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

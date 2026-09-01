from pathlib import Path


ASSET_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "pocketfinancer_sms"
    / "workbench"
    / "assets"
)


def test_sms_workbench_keeps_columns_in_viewport_and_reveals_selected_detail() -> None:
    markup = (ASSET_ROOT / "index.html").read_text(encoding="utf-8")
    styles = (ASSET_ROOT / "styles.css").read_text(encoding="utf-8")
    script = (ASSET_ROOT / "app.js").read_text(encoding="utf-8")

    assert "html, body { height: 100%; }" in styles
    assert "[hidden] { display: none !important; }" in styles
    assert "grid-template-rows: auto auto auto minmax(0, 1fr)" in styles
    assert ".workspace { min-height: 0;" in styles
    assert 'el("detailContent").closest(".detail").scrollTop = 0;' in script
    assert 'id="coverageToggleHint"' in markup
    assert 'coverage.textContent = error.message || "Coverage data could not be loaded.";' in script
    assert 'el("coveragePanel").addEventListener("toggle"' in script

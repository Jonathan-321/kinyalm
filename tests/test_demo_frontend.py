from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "kinyalm-chat"


def test_demo_frontend_has_local_assets_and_core_controls():
    html = (APP / "index.html").read_text(encoding="utf-8")
    javascript = (APP / "app.js").read_text(encoding="utf-8")

    assert (APP / "styles.css").is_file()
    assert (APP / "assets" / "kinyalm-mark.png").is_file()
    assert 'data-mode="converse"' in html
    assert 'data-mode="translate"' in html
    assert 'data-mode="learn"' in html
    assert "/api/chat" in javascript
    assert "/api/feedback" in javascript
    assert "http://" not in html
    assert "https://" not in html

from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_dashboard_renders_and_switches_sample(monkeypatch):
    monkeypatch.setenv("PAIRS_DEMO", "1")
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert any("SYNTHETIC" in w.value for w in app.warning)
    assert len(app.metric) >= 8
    app.radio(key="evaluation").set_value("In sample · 60%").run()
    assert not app.exception
    assert any("IN SAMPLE" in w.value for w in app.warning)


def test_real_snapshot_controls_submit_and_exploratory_warning(monkeypatch):
    monkeypatch.delenv("PAIRS_DEMO", raising=False)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=60)
    app.query_params["snapshot"] = "1"
    app.run()
    assert not app.exception
    assert not any("SYNTHETIC DEMONSTRATION" in w.value for w in app.warning)
    app.checkbox(key="qualification_gate").uncheck()
    app.button(key="run_research").click().run()
    assert not app.exception
    assert any("Exploratory trading" in w.value for w in app.warning)

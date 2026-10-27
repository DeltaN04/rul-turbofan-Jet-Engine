"""Smoke test for Streamlit app: verify each page renders without exceptions."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_smoke():
    """Run the app and check each page renders without errors."""
    at = AppTest.from_file(str(APP)).run(timeout=60)

    # Check no exceptions on initial load (Overview page)
    assert not at.exception, f"Exception on load: {at.exception}"

    # Check Overview page elements
    assert at.get("title")[0].value == "Turbofan Engine RUL Estimation & Maintenance Alerting"
    assert len(at.get("subheader")) >= 1

    # Navigate to each page via sidebar radio
    pages = [
        "1. Overview",
        "2. Engine Monitor",
        "3. Fleet View",
        "4. Model Performance",
        "5. Cost & Alert Policy",
        "6. About / Limitations",
    ]

    for page_name in pages:
        # Select the page in sidebar radio (by key)
        at.sidebar.radio(key="navigation").set_value(page_name).run(timeout=90)
        assert not at.exception, f"Exception on page '{page_name}': {at.exception}"
        # Verify page-specific content exists
        if page_name == "1. Overview":
            assert any("Headline Metrics" in str(el.value) for el in at.get("subheader"))
        elif page_name == "2. Engine Monitor":
            assert any("Engine Monitor" in str(el.value) for el in at.get("title"))
        elif page_name == "3. Fleet View":
            assert any("Fleet View" in str(el.value) for el in at.get("title"))
        elif page_name == "4. Model Performance":
            assert any("Model Performance" in str(el.value) for el in at.get("title"))
        elif page_name == "5. Cost & Alert Policy":
            assert any("Cost & Alert Policy" in str(el.value) for el in at.get("title"))
        elif page_name == "6. About / Limitations":
            assert any("About / Limitations" in str(el.value) for el in at.get("title"))

    print("✅ All pages rendered successfully without exceptions")


if __name__ == "__main__":
    test_app_smoke()
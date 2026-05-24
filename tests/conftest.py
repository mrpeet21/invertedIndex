import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "id": 10,
            "final_url": "https://example.test/1",
            "title": "machine learning",
            "text_content": "model model training",
        },
        {
            "id": 20,
            "final_url": "https://example.test/2",
            "title": "search index",
            "text_content": "index stores documents",
        },
        {
            "id": 30,
            "final_url": "https://example.test/3",
            "title": "model search",
            "text_content": "search model",
        },
        {
            "id": 40,
            "final_url": "https://example.test/4",
            "title": "learning notes",
            "text_content": "machine methods and learning",
        },
    ])


@pytest.fixture
def simple_preprocess():
    def _simple_preprocess(text):
        if not isinstance(text, str):
            return []
        return [token for token in text.lower().split() if len(token) > 2]

    return _simple_preprocess


@pytest.fixture
def patched_preprocess(monkeypatch, simple_preprocess):
    import indexer
    import search_engine

    monkeypatch.setattr(indexer, "preprocess_text", simple_preprocess)
    monkeypatch.setattr(search_engine, "preprocess_text", simple_preprocess)

    return simple_preprocess


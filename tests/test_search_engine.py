from indexer import (
    build_compressed_delta_index,
    build_compressed_gamma_index,
    build_doc_id_maps,
    build_inverted_index,
)
from search_engine import (
    benchmark_phrase_search,
    benchmark_search,
    phrase_exists_in_doc,
    search_phrase_normal,
    search_query_compressed_delta,
    search_query_compressed_gamma,
    search_query_normal,
    show_results,
)


def _build_indexes(sample_df):
    page_to_doc, doc_to_page = build_doc_id_maps(sample_df)
    inverted_index, positional_index, _indexing_time = build_inverted_index(sample_df, page_to_doc)
    gamma_index, _gamma_time = build_compressed_gamma_index(inverted_index)
    delta_index, _delta_time = build_compressed_delta_index(inverted_index)
    return inverted_index, positional_index, gamma_index, delta_index, doc_to_page


def test_normal_search_uses_and_semantics(sample_df, patched_preprocess):
    inverted_index, _positional_index, _gamma_index, _delta_index, _doc_to_page = _build_indexes(
        sample_df
    )

    assert search_query_normal(inverted_index, "model search") == [3]
    assert search_query_normal(inverted_index, "missing") == []


def test_phrase_exists_in_doc_requires_adjacent_positions():
    assert phrase_exists_in_doc([[2, 10], [3, 30]]) is True
    assert phrase_exists_in_doc([[2, 10], [4, 30]]) is False


def test_phrase_search_requires_adjacent_words(sample_df, patched_preprocess):
    _inverted_index, positional_index, _gamma_index, _delta_index, _doc_to_page = _build_indexes(
        sample_df
    )

    assert search_phrase_normal(positional_index, "machine learning") == [1]
    assert search_phrase_normal(positional_index, "learning machine") == []


def test_compressed_search_matches_normal_for_several_queries(sample_df, patched_preprocess):
    inverted_index, _positional_index, gamma_index, delta_index, _doc_to_page = _build_indexes(
        sample_df
    )

    queries = [
        "machine learning",
        "model search",
        "search index",
        "missing query",
    ]

    for query in queries:
        normal = search_query_normal(inverted_index, query)
        gamma = search_query_compressed_gamma(gamma_index, query)
        delta = search_query_compressed_delta(delta_index, query)

        assert gamma == normal
        assert delta == normal


def test_benchmark_search_returns_stats_for_all_indexes(sample_df, patched_preprocess):
    inverted_index, _positional_index, gamma_index, delta_index, _doc_to_page = _build_indexes(
        sample_df
    )

    stats, normal, gamma, delta = benchmark_search(
        query_text="model search",
        inverted_index=inverted_index,
        compressed_gamma_index=gamma_index,
        compressed_delta_index=delta_index,
        repeats=2,
    )

    assert normal == gamma == delta == [3]
    assert list(stats["Индекс"]) == ["Обычный", "Gap + Elias Gamma", "Gap + Elias Delta"]
    assert (stats["Среднее время поиска, сек"] >= 0).all()


def test_benchmark_phrase_search_returns_stats(sample_df, patched_preprocess):
    _inverted_index, positional_index, _gamma_index, _delta_index, _doc_to_page = _build_indexes(
        sample_df
    )

    stats, result = benchmark_phrase_search(
        query_text="machine learning",
        positional_index=positional_index,
        repeats=2,
    )

    assert result == [1]
    assert list(stats["Тип поиска"]) == ["Фразовый поиск по позиционному индексу"]
    assert (stats["Среднее время поиска, сек"] >= 0).all()


def test_show_results_maps_doc_ids_to_pages(sample_df, patched_preprocess):
    _inverted_index, _positional_index, _gamma_index, _delta_index, doc_to_page = _build_indexes(
        sample_df
    )

    results = show_results([1, 3], sample_df, doc_to_page, limit=2)

    assert list(results["page_id"]) == [10, 30]
    assert list(results["doc_id"]) == [1, 3]


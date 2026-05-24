from indexer import (
    build_compressed_delta_index,
    build_compressed_gamma_index,
    build_doc_id_maps,
    build_inverted_index,
    delta_compressed_size_bits,
    gamma_compressed_size_bits,
    uncompressed_size_bits,
)


def test_build_doc_id_maps(sample_df):
    page_to_doc, doc_to_page = build_doc_id_maps(sample_df)

    assert page_to_doc == {10: 1, 20: 2, 30: 3, 40: 4}
    assert doc_to_page == {1: 10, 2: 20, 3: 30, 4: 40}


def test_build_inverted_index_includes_title_and_text(sample_df, patched_preprocess):
    page_to_doc, _doc_to_page = build_doc_id_maps(sample_df)
    inverted_index, positional_index, indexing_time = build_inverted_index(sample_df, page_to_doc)

    assert indexing_time >= 0
    assert inverted_index["machine"] == [1, 4]
    assert inverted_index["model"] == [1, 3]
    assert positional_index["model"][1] == [2, 3]


def test_postings_are_sorted_and_unique(sample_df, patched_preprocess):
    page_to_doc, _doc_to_page = build_doc_id_maps(sample_df)
    inverted_index, _positional_index, _indexing_time = build_inverted_index(sample_df, page_to_doc)

    for postings in inverted_index.values():
        assert postings == sorted(postings)
        assert len(postings) == len(set(postings))


def test_compressed_index_sizes_are_positive(sample_df, patched_preprocess):
    page_to_doc, _doc_to_page = build_doc_id_maps(sample_df)
    inverted_index, _positional_index, _indexing_time = build_inverted_index(sample_df, page_to_doc)

    normal_bits = uncompressed_size_bits(inverted_index)
    gamma_bits = gamma_compressed_size_bits(inverted_index)
    delta_bits = delta_compressed_size_bits(inverted_index)

    assert normal_bits > 0
    assert gamma_bits > 0
    assert delta_bits > 0


def test_build_compressed_indexes(sample_df, patched_preprocess):
    page_to_doc, _doc_to_page = build_doc_id_maps(sample_df)
    inverted_index, _positional_index, _indexing_time = build_inverted_index(sample_df, page_to_doc)

    gamma_index, gamma_time = build_compressed_gamma_index(inverted_index)
    delta_index, delta_time = build_compressed_delta_index(inverted_index)

    assert set(gamma_index) == set(inverted_index)
    assert set(delta_index) == set(inverted_index)
    assert gamma_time >= 0
    assert delta_time >= 0


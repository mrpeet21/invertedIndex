import timeit

import pandas as pd

from elias import decompress_postings_delta, decompress_postings_gamma
from preprocessing import preprocess_text


def intersect_sorted_lists(lists: list[list[int]]) -> list[int]:
    if not lists:
        return []

    result = set(lists[0])

    for lst in lists[1:]:
        result &= set(lst)

    return sorted(result)


def search_query_normal(index: dict[str, list[int]], query_text: str) -> list[int]:
    query_terms = preprocess_text(query_text)

    if not query_terms:
        return []

    postings_lists = [
        index.get(term, [])
        for term in query_terms
    ]

    return intersect_sorted_lists(postings_lists)


def search_query_compressed_gamma(
    compressed_index: dict[str, str],
    query_text: str,
) -> list[int]:
    query_terms = preprocess_text(query_text)

    if not query_terms:
        return []

    postings_lists = []

    for term in query_terms:
        bitstring = compressed_index.get(term)

        if bitstring is None:
            return []

        postings = decompress_postings_gamma(bitstring)
        postings_lists.append(postings)

    return intersect_sorted_lists(postings_lists)


def search_query_compressed_delta(
    compressed_index: dict[str, str],
    query_text: str,
) -> list[int]:
    query_terms = preprocess_text(query_text)

    if not query_terms:
        return []

    postings_lists = []

    for term in query_terms:
        bitstring = compressed_index.get(term)

        if bitstring is None:
            return []

        postings = decompress_postings_delta(bitstring)
        postings_lists.append(postings)

    return intersect_sorted_lists(postings_lists)


def show_results(
    doc_ids: list[int],
    df: pd.DataFrame,
    doc_id_to_page_id: dict[int, int],
    limit: int = 10,
) -> pd.DataFrame:
    """
    Возвращает таблицу с найденными документами.
    """
    rows = []

    for doc_id in doc_ids[:limit]:
        real_page_id = doc_id_to_page_id[doc_id]
        page = df[df["id"] == real_page_id].iloc[0]

        rows.append({
            "doc_id": doc_id,
            "page_id": real_page_id,
            "title": page["title"],
            "url": page["final_url"],
        })

    return pd.DataFrame(rows)


def benchmark_search(
    query_text: str,
    inverted_index: dict[str, list[int]],
    compressed_gamma_index: dict[str, str],
    compressed_delta_index: dict[str, str],
    repeats: int = 1000,
) -> tuple[pd.DataFrame, list[int], list[int], list[int]]:
    """
    Сравнивает поиск:
    - по обычному индексу;
    - по Gamma-сжатому индексу;
    - по Delta-сжатому индексу.
    """
    normal_result = search_query_normal(inverted_index, query_text)
    gamma_result = search_query_compressed_gamma(compressed_gamma_index, query_text)
    delta_result = search_query_compressed_delta(compressed_delta_index, query_text)

    normal_time = timeit.timeit(
        stmt="search_query_normal(inverted_index, query_text)",
        globals={
            "search_query_normal": search_query_normal,
            "inverted_index": inverted_index,
            "query_text": query_text,
        },
        number=repeats,
    )

    gamma_time = timeit.timeit(
        stmt="search_query_compressed_gamma(compressed_gamma_index, query_text)",
        globals={
            "search_query_compressed_gamma": search_query_compressed_gamma,
            "compressed_gamma_index": compressed_gamma_index,
            "query_text": query_text,
        },
        number=repeats,
    )

    delta_time = timeit.timeit(
        stmt="search_query_compressed_delta(compressed_delta_index, query_text)",
        globals={
            "search_query_compressed_delta": search_query_compressed_delta,
            "compressed_delta_index": compressed_delta_index,
            "query_text": query_text,
        },
        number=repeats,
    )

    search_stats = pd.DataFrame([
        {
            "Индекс": "Обычный",
            "Среднее время поиска, сек": normal_time / repeats,
            "Найдено документов": len(normal_result),
            "Совпадает с обычным индексом": True,
        },
        {
            "Индекс": "Gap + Elias Gamma",
            "Среднее время поиска, сек": gamma_time / repeats,
            "Найдено документов": len(gamma_result),
            "Совпадает с обычным индексом": normal_result == gamma_result,
        },
        {
            "Индекс": "Gap + Elias Delta",
            "Среднее время поиска, сек": delta_time / repeats,
            "Найдено документов": len(delta_result),
            "Совпадает с обычным индексом": normal_result == delta_result,
        },
    ])

    return search_stats, normal_result, gamma_result, delta_result

def phrase_exists_in_doc(position_lists: list[list[int]]) -> bool:
    """
    Проверяет, есть ли в одном документе слова запроса подряд.

    Пример для запроса из двух слов:
    первое слово стоит на позициях [5, 20]
    второе слово стоит на позициях [6, 30]

    Тогда фраза найдена, потому что 5 + 1 = 6.

    Для трёх слов проверяется цепочка:
    pos, pos + 1, pos + 2.
    """
    if not position_lists:
        return False

    first_positions = position_lists[0]
    other_position_sets = [set(positions) for positions in position_lists[1:]]

    for start_pos in first_positions:
        found = True

        for offset, positions in enumerate(other_position_sets, start=1):
            if start_pos + offset not in positions:
                found = False
                break

        if found:
            return True

    return False


def search_phrase_normal(
    positional_index: dict[str, dict[int, list[int]]],
    query_text: str,
) -> list[int]:
    """
    Фразовый поиск: ищет документы, где слова запроса стоят рядом
    и в том же порядке.

    Например, запрос "ректор спбгу" найдёт документ с фрагментом:

        "... ректор спбгу ..."

    Но НЕ найдёт документ, где написано:

        "ректор университета спбгу"
    """
    query_terms = preprocess_text(query_text)

    if not query_terms:
        return []

    # Если хотя бы одного слова нет в позиционном индексе,
    # фраза точно не найдена.
    for term in query_terms:
        if term not in positional_index:
            return []

    # Сначала берём только документы, где есть все слова запроса.
    candidate_docs = set(positional_index[query_terms[0]].keys())

    for term in query_terms[1:]:
        candidate_docs &= set(positional_index[term].keys())

    result = []

    # Потом для каждого документа-кандидата проверяем позиции слов.
    for doc_id in candidate_docs:
        position_lists = [
            positional_index[term][doc_id]
            for term in query_terms
        ]

        if phrase_exists_in_doc(position_lists):
            result.append(doc_id)

    return sorted(result)


def benchmark_phrase_search(
    query_text: str,
    positional_index: dict[str, dict[int, list[int]]],
    repeats: int = 1000,
) -> tuple[pd.DataFrame, list[int]]:
    """
    Измеряет скорость фразового поиска по позиционному индексу.
    """
    phrase_result = search_phrase_normal(positional_index, query_text)

    phrase_time = timeit.timeit(
        stmt="search_phrase_normal(positional_index, query_text)",
        globals={
            "search_phrase_normal": search_phrase_normal,
            "positional_index": positional_index,
            "query_text": query_text,
        },
        number=repeats,
    )

    phrase_stats = pd.DataFrame([
        {
            "Тип поиска": "Фразовый поиск по позиционному индексу",
            "Среднее время поиска, сек": phrase_time / repeats,
            "Найдено документов": len(phrase_result),
        }
    ])

    return phrase_stats, phrase_result

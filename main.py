from pathlib import Path

from indexer import (
    build_compressed_delta_index,
    build_compressed_gamma_index,
    build_doc_id_maps,
    build_inverted_index,
    load_pages,
    make_summary,
)
from preprocessing import ensure_nltk_data, preprocess_text
from search_engine import benchmark_phrase_search, benchmark_search, show_results


DATA_PATH = "data/pages.csv.gz"

# Для быстрой проверки можно поставить, например, 1000. Для полного запуска – None.
MAX_DOCS = None

QUERY_TEXT = "Машинное обучение"

# Количество повторов для измерения среднего времени поиска.
SEARCH_REPEATS = 1000

def main() -> None:
    ensure_nltk_data()

    data_path = Path(DATA_PATH)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {data_path}\n"
            f"Положи pages.csv.gz в папку data/ рядом с main.py."
        )

    print("Загрузка данных...")
    df = load_pages(str(data_path), max_docs=MAX_DOCS)

    print("Количество текстовых документов:", len(df))
    print("Колонки:", list(df.columns))
    print()

    page_id_to_doc_id, doc_id_to_page_id = build_doc_id_maps(df)

    print("Количество внутренних doc_id:", len(page_id_to_doc_id))

    inverted_index, positional_index, indexing_time = build_inverted_index(
    df = df,
    page_id_to_doc_id = page_id_to_doc_id,
    )

    total_postings = sum(len(postings) for postings in inverted_index.values())
    docs_per_second = len(df) / indexing_time if indexing_time > 0 else 0

    print()
    print("Индекс построен")
    print("Количество документов:", len(df))
    print("Количество уникальных терминов:", len(inverted_index))
    print("Общее количество postings:", total_postings)
    print(f"Время индексирования: {indexing_time:.4f} сек.")
    print(f"Скорость индексирования: {docs_per_second:.2f} документов/сек.")
    print()

    compressed_gamma_index, gamma_build_time = build_compressed_gamma_index(inverted_index)
    compressed_delta_index, delta_build_time = build_compressed_delta_index(inverted_index)

    print()
    print(f"Время построения Gamma-сжатого индекса: {gamma_build_time:.4f} сек.")
    print(f"Время построения Delta-сжатого индекса: {delta_build_time:.4f} сек.")
    print()

    summary = make_summary(
        df=df,
        index=inverted_index,
        indexing_time=indexing_time,
        gamma_build_time=gamma_build_time,
        delta_build_time=delta_build_time,
    )

    print("Сводная таблица:")
    print(summary.to_string(index=False))
    print()

    search_stats, normal_result, gamma_result, delta_result = benchmark_search(
        query_text=QUERY_TEXT,
        inverted_index=inverted_index,
        compressed_gamma_index=compressed_gamma_index,
        compressed_delta_index=compressed_delta_index,
        repeats=SEARCH_REPEATS,
    )

    print("Запрос:", QUERY_TEXT)
    print("Леммы запроса:", preprocess_text(QUERY_TEXT))
    print()
    print("Статистика поиска:")
    print(search_stats.to_string(index=False))
    print()

    results = show_results(
        doc_ids=normal_result,
        df=df,
        doc_id_to_page_id=doc_id_to_page_id,
        limit=10,
    )

    print("Первые найденные документы:")
    if len(results) == 0:
        print("Документы не найдены.")
    else:
        print(results.to_string(index=False))


    print()

    query_terms = preprocess_text(QUERY_TEXT)
    if len(query_terms) < 2:
        return

    print("Фразовый поиск, то есть слова должны стоять рядом:")

    phrase_stats, phrase_result = benchmark_phrase_search(
        query_text=QUERY_TEXT,
        positional_index=positional_index,
        repeats=SEARCH_REPEATS,
    )

    print(phrase_stats.to_string(index=False))

    phrase_results = show_results(
        doc_ids=phrase_result,
        df=df,
        doc_id_to_page_id=doc_id_to_page_id,
        limit=10,
    )

    print()
    print("Первые найденные документы для фразового поиска:")

    if len(phrase_results) == 0:
        print("Документы не найдены.")
    else:
        print(phrase_results.to_string(index=False))

if __name__ == "__main__":
    main()


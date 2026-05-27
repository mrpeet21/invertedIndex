import time
from collections import Counter, defaultdict

import pandas as pd
from tqdm import tqdm

from elias import (
    compress_postings_delta,
    compress_postings_gamma,
    elias_delta_length,
    elias_gamma_length,
    gap_encode,
)
from preprocessing import preprocess_text


REQUIRED_COLUMNS = ["id", "final_url", "title", "text_content"]


def load_pages(csv_path: str, max_docs: int | None = None) -> pd.DataFrame:
    """
    Загружает pages.csv.gz.

    """
    df = pd.read_csv(
        csv_path,
        compression="gzip",
        usecols=REQUIRED_COLUMNS,
        encoding="utf-8",
        low_memory=False,
    )

    df = df.dropna(subset=["text_content"]).copy()
    df = df[df["text_content"].astype(str).str.len() > 0].copy()

    df["title"] = df["title"].fillna("")
    df["final_url"] = df["final_url"].fillna("")

    if max_docs is not None:
        df = df.head(max_docs).copy()

    return df


def build_doc_id_maps(df: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    """
    В CSV id страниц могут быть не подряд.
    """
    real_page_ids = sorted(df["id"].unique())

    page_id_to_doc_id = {
        int(page_id): i + 1
        for i, page_id in enumerate(real_page_ids)
    }

    doc_id_to_page_id = {
        doc_id: page_id
        for page_id, doc_id in page_id_to_doc_id.items()
    }

    return page_id_to_doc_id, doc_id_to_page_id


def build_inverted_index(    df: pd.DataFrame,
    page_id_to_doc_id: dict[int, int],
) -> tuple[
    dict[str, list[int]],
    dict[str, dict[int, list[int]]],
    float,
]:
    """
    Строит две структуры:

    1. inverted_index:
       term -> [doc_id1, doc_id2, ...]
       Нужен для обычного AND-поиска:
       документ содержит все слова запроса.

    2. positional_index:
       term -> {doc_id: [pos1, pos2, ...]}
       Нужен для фразового поиска:
       слова должны стоять рядом и в правильном порядке.

    """
    inverted_index = defaultdict(set)
    positional_index = defaultdict(lambda: defaultdict(list))

    start = time.perf_counter()

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Индексирование"):
        real_page_id = int(row.id)
        doc_id = page_id_to_doc_id[real_page_id]

        title = "" if pd.isna(row.title) else str(row.title)
        text_content = "" if pd.isna(row.text_content) else str(row.text_content)

        full_text = title + " " + text_content

        tokens = preprocess_text(full_text)

        
        for term in set(tokens):
            inverted_index[term].add(doc_id)

        indexing_time = time.perf_counter() - start

       
        for position, term in enumerate(tokens):
            positional_index[term][doc_id].append(position)

    inverted_index = {
        term: sorted(doc_ids)
        for term, doc_ids in inverted_index.items()
    }

    positional_index = {
        term: dict(doc_positions)
        for term, doc_positions in positional_index.items()
    }

    return inverted_index, positional_index, indexing_time


def uncompressed_size_bits(index: dict[str, list[int]], bits_per_doc_id: int = 32) -> int:
    """
    Оцениваем размер классического индекса:
    каждый doc_id хранится как 32-битное целое число.
    """
    return sum(len(postings) * bits_per_doc_id for postings in index.values())


def gamma_compressed_size_bits(index: dict[str, list[int]]) -> int:
    """
    Оцениваем размер Gap + Elias Gamma.
    Строки битов не строим, считаем только длину.
    Так быстрее и экономнее по памяти.
    """
    total = 0

    for postings in index.values():
        gaps = gap_encode(postings)

        for gap in gaps:
            total += elias_gamma_length(gap)

    return total


def delta_compressed_size_bits(index: dict[str, list[int]]) -> int:
    """
    Оцениваем размер Gap + Elias Delta.
    """
    total = 0

    for postings in index.values():
        gaps = gap_encode(postings)

        for gap in gaps:
            total += elias_delta_length(gap)

    return total


def build_compressed_gamma_index(index: dict[str, list[int]]) -> tuple[dict[str, str], float]:
    start = time.perf_counter()

    compressed_index = {
        term: compress_postings_gamma(postings)
        for term, postings in tqdm(index.items(), desc="Gamma-сжатие")
    }

    elapsed = time.perf_counter() - start

    return compressed_index, elapsed


def build_compressed_delta_index(index: dict[str, list[int]]) -> tuple[dict[str, str], float]:
    start = time.perf_counter()

    compressed_index = {
        term: compress_postings_delta(postings)
        for term, postings in tqdm(index.items(), desc="Delta-сжатие")
    }

    elapsed = time.perf_counter() - start

    return compressed_index, elapsed


def make_summary(
    df: pd.DataFrame,
    index: dict[str, list[int]],
    indexing_time: float,
    gamma_build_time: float,
    delta_build_time: float,
) -> pd.DataFrame:
    total_postings = sum(len(postings) for postings in index.values())
    docs_per_second = len(df) / indexing_time if indexing_time > 0 else 0

    normal_bits_32 = uncompressed_size_bits(index, bits_per_doc_id=32)
    gamma_bits = gamma_compressed_size_bits(index)
    delta_bits = delta_compressed_size_bits(index)

    normal_mb = normal_bits_32 / 8 / 1024 / 1024
    gamma_mb = gamma_bits / 8 / 1024 / 1024
    delta_mb = delta_bits / 8 / 1024 / 1024

    gamma_ratio = normal_bits_32 / gamma_bits if gamma_bits > 0 else 0
    delta_ratio = normal_bits_32 / delta_bits if delta_bits > 0 else 0

    return pd.DataFrame([
        {
            "Метрика": "Количество документов",
            "Значение": len(df),
        },
        {
            "Метрика": "Количество уникальных терминов",
            "Значение": len(index),
        },
        {
            "Метрика": "Общее количество postings",
            "Значение": total_postings,
        },
        {
            "Метрика": "Время индексирования, сек",
            "Значение": round(indexing_time, 4),
        },
        {
            "Метрика": "Скорость индексирования, документов/сек",
            "Значение": round(docs_per_second, 2),
        },
        {
            "Метрика": "Обычный индекс, МБ",
            "Значение": round(normal_mb, 4),
        },
        {
            "Метрика": "Gap + Elias Gamma, МБ",
            "Значение": round(gamma_mb, 4),
        },
        {
            "Метрика": "Gap + Elias Delta, МБ",
            "Значение": round(delta_mb, 4),
        },
        {
            "Метрика": "Коэффициент сжатия Gamma",
            "Значение": round(gamma_ratio, 2),
        },
        {
            "Метрика": "Коэффициент сжатия Delta",
            "Значение": round(delta_ratio, 2),
        },
        {
            "Метрика": "Время построения Gamma-сжатого индекса, сек",
            "Значение": round(gamma_build_time, 4),
        },
        {
            "Метрика": "Время построения Delta-сжатого индекса, сек",
            "Значение": round(delta_build_time, 4),
        },
    ])

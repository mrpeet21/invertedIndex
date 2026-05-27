
Кусок кода из файла `indexer.py`, для понимания размеров индексов:

``` python
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

```

Кусок кода из файла `elias.py` (функции, которые нужны для подсчета размера):

``` python
def gap_encode(numbers: list[int]) -> list[int]:
    """
    Преобразует posting list в список разностей.

    """
    if not numbers:
        return []

    gaps = [numbers[0]]

    for i in range(1, len(numbers)):
        gaps.append(numbers[i] - numbers[i - 1])

    return gaps

def elias_gamma_length(n: int) -> int:
    """
    Длина gamma-кода без фактического построения строки.
    Это быстрее для оценки размера индекса.
    """
    if n <= 0:
        raise ValueError("Elias Gamma works only for positive integers")

    return 2 * n.bit_length() - 1

def elias_delta_length(n: int) -> int:
    """
    Длина delta-кода без фактического построения строки.
    """
    if n <= 0:
        raise ValueError("Elias Delta works only for positive integers")

    length = n.bit_length()

    return elias_gamma_length(length) + (length - 1)

```

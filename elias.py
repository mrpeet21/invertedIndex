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


def gap_decode(gaps: list[int]) -> list[int]:
    """
    Восстанавливает posting list из списка разностей.

    """
    numbers = []
    current = 0

    for gap in gaps:
        current += gap
        numbers.append(current)

    return numbers


def elias_gamma_encode_number(n: int) -> str:
    """
    Gamma-кодирование Элиаса для одного положительного числа.

    """
    if n <= 0:
        raise ValueError("Elias Gamma works only for positive integers")

    binary = bin(n)[2:]
    prefix = "0" * (len(binary) - 1)

    return prefix + binary


def elias_gamma_decode_one(bitstring: str, start: int = 0) -> tuple[int, int]:
    i = start
    zeros = 0

    while i < len(bitstring) and bitstring[i] == "0":
        zeros += 1
        i += 1

    if i >= len(bitstring):
        raise ValueError("Invalid Elias Gamma code")

    length = zeros + 1
    binary = bitstring[i:i + length]

    if len(binary) < length:
        raise ValueError("Invalid Elias Gamma code")

    value = int(binary, 2)

    return value, i + length


def elias_gamma_decode_stream(bitstring: str) -> list[int]:
    numbers = []
    i = 0

    while i < len(bitstring):
        value, i = elias_gamma_decode_one(bitstring, i)
        numbers.append(value)

    return numbers


def elias_gamma_length(n: int) -> int:
    """
    Длина gamma-кода без фактического построения строки.
    Это быстрее для оценки размера индекса.
    """
    if n <= 0:
        raise ValueError("Elias Gamma works only for positive integers")

    return 2 * n.bit_length() - 1


def elias_delta_encode_number(n: int) -> str:
    """
    Delta-кодирование Элиаса для одного положительного числа.
    """
    if n <= 0:
        raise ValueError("Elias Delta works only for positive integers")

    binary = bin(n)[2:]
    length = len(binary)

    length_code = elias_gamma_encode_number(length)
    offset = binary[1:]

    return length_code + offset


def elias_delta_decode_one(bitstring: str, start: int = 0) -> tuple[int, int]:
    length, i = elias_gamma_decode_one(bitstring, start)

    offset_length = length - 1
    offset = bitstring[i:i + offset_length]

    if len(offset) < offset_length:
        raise ValueError("Invalid Elias Delta code")

    binary = "1" + offset
    value = int(binary, 2)

    return value, i + offset_length


def elias_delta_decode_stream(bitstring: str) -> list[int]:
    numbers = []
    i = 0

    while i < len(bitstring):
        value, i = elias_delta_decode_one(bitstring, i)
        numbers.append(value)

    return numbers


def elias_delta_length(n: int) -> int:
    """
    Длина delta-кода без фактического построения строки.
    """
    if n <= 0:
        raise ValueError("Elias Delta works only for positive integers")

    length = n.bit_length()

    return elias_gamma_length(length) + (length - 1)


def compress_postings_gamma(postings: list[int]) -> str:
    gaps = gap_encode(postings)
    return "".join(elias_gamma_encode_number(gap) for gap in gaps)


def decompress_postings_gamma(bitstring: str) -> list[int]:
    gaps = elias_gamma_decode_stream(bitstring)
    return gap_decode(gaps)


def compress_postings_delta(postings: list[int]) -> str:
    gaps = gap_encode(postings)
    return "".join(elias_delta_encode_number(gap) for gap in gaps)


def decompress_postings_delta(bitstring: str) -> list[int]:
    gaps = elias_delta_decode_stream(bitstring)
    return gap_decode(gaps)

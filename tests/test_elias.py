import pytest

from elias import (
    compress_postings_delta,
    compress_postings_gamma,
    decompress_postings_delta,
    decompress_postings_gamma,
    elias_delta_decode_stream,
    elias_delta_encode_number,
    elias_gamma_decode_stream,
    elias_gamma_encode_number,
    gap_decode,
    gap_encode,
)


def test_gap_encode_known_list():
    assert gap_encode([12, 57, 104, 205]) == [12, 45, 47, 101]


def test_gap_round_trip():
    postings = [1, 3, 10, 100]
    assert gap_decode(gap_encode(postings)) == postings


@pytest.mark.parametrize("number", [1, 2, 3, 5, 20, 100])
def test_elias_gamma_round_trip(number):
    assert elias_gamma_decode_stream(elias_gamma_encode_number(number)) == [number]


@pytest.mark.parametrize("number", [1, 2, 3, 5, 20, 100])
def test_elias_delta_round_trip(number):
    assert elias_delta_decode_stream(elias_delta_encode_number(number)) == [number]


@pytest.mark.parametrize("number", [0, -1])
def test_elias_rejects_non_positive_numbers(number):
    with pytest.raises(ValueError):
        elias_gamma_encode_number(number)

    with pytest.raises(ValueError):
        elias_delta_encode_number(number)


def test_compressed_postings_round_trip():
    postings = [12, 57, 104, 205]
    assert decompress_postings_gamma(compress_postings_gamma(postings)) == postings
    assert decompress_postings_delta(compress_postings_delta(postings)) == postings


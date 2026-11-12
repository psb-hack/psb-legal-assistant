from __future__ import annotations

import pytest

from lex_search_tokens import WORD_NGRAM_LENGTH, search_tokens, word_ngrams


def test_word_becomes_ngrams_with_boundary_marks() -> None:
    assert search_tokens("Срок") == ("_сро", "срок", "рок_")


def test_word_shorter_than_ngram_is_one_token() -> None:
    assert search_tokens("по") == ("_по_",)
    assert all(len(gram) == WORD_NGRAM_LENGTH for gram in word_ngrams("договора"))


def test_case_and_yo_do_not_change_tokens() -> None:
    assert search_tokens("ОТЧЁТ") == search_tokens("отчет")


def test_word_forms_share_most_ngrams() -> None:
    shared = set(search_tokens("поставки")) & set(search_tokens("поставщику"))
    assert shared == {"_пос", "пост", "оста", "став"}


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("П-17/2024", ("п-17/2024", "_п_", "17", "2024")),
        ("152-ФЗ", ("152-фз", "152", "_фз_")),
        ("13.11", ("13.11", "13", "11")),
        ("3.1.", ("3.1", "3", "1")),
        ("0,1%", ("0,1", "0", "1")),
        ("2024", ("2024",)),
    ],
)
def test_number_stays_whole_and_gives_its_parts(
    text: str, tokens: tuple[str, ...]
) -> None:
    assert search_tokens(text) == tokens


def test_number_written_with_another_dash_is_the_same_token() -> None:
    assert search_tokens("№ П‑17/2024") == search_tokens("п-17/2024")


def test_number_written_without_dash_shares_parts_with_full_form() -> None:
    shared = set(search_tokens("п17/2024")) & set(search_tokens("П-17/2024"))
    assert shared == {"_п_", "17", "2024"}


def test_numbers_differing_in_one_digit_have_different_whole_tokens() -> None:
    assert search_tokens("П-17/2024")[0] != search_tokens("П-17/2023")[0]


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("450 000 рублей", "450000"),
        ("450 000 рублей", "450000"),
        ("10 000 000 (десять миллионов)", "10000000"),
        ("450000", "450000"),
    ],
)
def test_digit_groups_of_a_sum_are_joined(text: str, number: str) -> None:
    assert search_tokens(text)[0] == number


def test_clause_number_before_a_sum_is_not_glued_to_it() -> None:
    tokens = search_tokens("п. 3.2 450 000")
    assert "3.2" in tokens
    assert "450000" in tokens


def test_year_before_a_number_is_not_glued_to_it() -> None:
    assert search_tokens("2024 100") == ("2024", "100")


def test_listed_clauses_are_separate_tokens() -> None:
    tokens = search_tokens("п. 3.1, 3.2")
    assert "3.1" in tokens
    assert "3.2" in tokens


def test_hyphenated_name_is_split_into_words() -> None:
    assert search_tokens("Альфа-Трейд") == search_tokens("альфа трейд")


def test_text_without_letters_and_digits_has_no_tokens() -> None:
    assert search_tokens(" «—» №?! ") == ()

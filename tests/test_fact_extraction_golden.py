"""Извлечение величин на настоящей базе: пары «цифры (прописью)» и числительные.

Архив базы в репозиторий не входит: без него тесты пропускаются.
"""

from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import yaml

from demo_answers import ARCHIVE, base
from lex_fact_extraction import Quantity, find_facts
from lex_numerals import read_numeral, read_ordinal

GOLDEN: dict[str, Any] = yaml.safe_load(
    (Path(__file__).resolve().parent / "golden" / "fact_extraction.yaml").read_text(
        encoding="utf-8"
    )
)
# «450 000 (четыреста пятьдесят тысяч)», «7% (семь процентов)»
SPELLED_PAIR = re.compile(r"(\d[\d\s]*(?:,\d+)?)\s*%?\s*\(([А-Яа-яёЁ\s-]+)\)")
WORD = re.compile(r"[а-яё]+")
# Слова, начало которых совпадает с началом числительного. Список намеренно
# широкий: в знаменатель попадает и «одновременно», доля от этого только ниже.
NUMERAL_LIKE = re.compile(
    r"(?:дв[аеу]|трех|трем|четыр|пят|шест|сем[иь]|седьм|вос[еь]м|девят|десят|сорок"
    r"|сто$|ста$|[а-я]+дцат|[а-я]+сот$|[а-я]+стам|тысяч|миллион|миллиард|полтор"
    r"|полутор|один|одн[аоуи]|перв|втор|трет)"
)

pytestmark = pytest.mark.skipif(
    not ARCHIVE.is_file(), reason="нет data/knowledge-base.zip"
)


def _texts() -> list[str]:
    return [
        provision.text
        for document in base().documents
        for provision in document.provisions
    ]


def test_every_spelled_duplicate_in_base_agrees_with_digits() -> None:
    pairs = [pair for text in _texts() for pair in SPELLED_PAIR.finditer(text)]
    agreed = 0
    for pair in pairs:
        digits = Fraction(re.sub(r"\s", "", pair.group(1)).replace(",", "."))
        spelled = read_numeral(WORD.findall(pair.group(2).lower()))
        facts = [
            fact for fact in find_facts(pair.group()) if isinstance(fact, Quantity)
        ]
        # Одна величина на пару: пропись в скобках вторым фактом не становится.
        if spelled is not None and spelled.value == digits and len(facts) == 1:
            agreed += facts[0].value == digits
    assert (agreed, len(pairs)) == (
        GOLDEN["spelled_pairs"]["agree"],
        GOLDEN["spelled_pairs"]["total"],
    )


def test_share_of_numeral_like_words_read_as_numerals() -> None:
    words = {
        word.replace("ё", "е")
        for text in _texts()
        for word in WORD.findall(text.lower())
    }
    alike = sorted(word for word in words if NUMERAL_LIKE.match(word))
    unread = [
        word
        for word in alike
        if read_numeral([word]) is None and read_ordinal([word]) is None
    ]
    expected = GOLDEN["numeral_like_words"]
    assert (len(alike) - len(unread), len(alike)) == (
        expected["read"],
        expected["total"],
    )
    assert unread == sorted(expected["unread"])

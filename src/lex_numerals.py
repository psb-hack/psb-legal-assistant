"""Русские числительные прописью: таблица форм и разбор по разрядам.

Числительные — закрытый класс слов, поэтому падежные формы перечислены явно,
без морфологии. Слово, которого нет в таблице, числом не считается: «стороны»,
«семьи», «одновременно» не дают ни «сто», ни «семь», ни «один».
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction


class _Place(Enum):
    """Место слова в тройке разрядов: «триста (сотни) сорок (десятки) пять»."""

    ONES = 1
    # 10–19 занимают сразу десятки и единицы: после них в тройке ничего нет.
    TEENS = 2
    TENS = 3
    HUNDREDS = 4


@dataclass(frozen=True, slots=True)
class NumeralReading:
    value: Fraction
    # Сколько слов с начала заняло числительное.
    words: int


def _forms(table: Sequence[tuple[int, str]]) -> dict[str, int]:
    return {form: value for value, forms in table for form in forms.split()}


def _regular(value: int, stem: str) -> tuple[int, str]:
    # «пять, пяти, пятью»: так склоняются 5–7, 9–20 и 30.
    return value, f"{stem}ь {stem}и {stem}ью"


def _tens(value: int, stem: str, oblique: str, instrumental: str) -> tuple[int, str]:
    # «пятьдесят, пятидесяти, пятьюдесятью»
    return value, f"{stem}десят {oblique}десяти {instrumental}десятью"


def _hundreds(
    value: int, stem: str, oblique: str, instrumental: str
) -> tuple[int, str]:
    # «пятьсот, пятисот, пятистам, пятьюстами, пятистах»
    return (
        value,
        f"{stem}сот {oblique}сот {oblique}стам {instrumental}стами {oblique}стах",
    )


_ONES = _forms(
    (
        (0, "ноль нуль ноля нуля нолю нулю нолем нулем"),
        (1, "один одна одно одного одной одному одним одном одну"),
        (2, "два две двух двум двумя"),
        (3, "три трех трем тремя"),
        (4, "четыре четырех четырем четырьмя"),
        _regular(5, "пят"),
        _regular(6, "шест"),
        _regular(7, "сем"),
        (8, "восемь восьми восемью восьмью"),
        _regular(9, "девят"),
    )
)
_TEENS = _forms(
    (
        _regular(10, "десят"),
        _regular(11, "одиннадцат"),
        _regular(12, "двенадцат"),
        _regular(13, "тринадцат"),
        _regular(14, "четырнадцат"),
        _regular(15, "пятнадцат"),
        _regular(16, "шестнадцат"),
        _regular(17, "семнадцат"),
        _regular(18, "восемнадцат"),
        _regular(19, "девятнадцат"),
    )
)
_TENS = _forms(
    (
        _regular(20, "двадцат"),
        _regular(30, "тридцат"),
        (40, "сорок сорока"),
        _tens(50, "пять", "пяти", "пятью"),
        _tens(60, "шесть", "шести", "шестью"),
        _tens(70, "семь", "семи", "семью"),
        (80, "восемьдесят восьмидесяти восемьюдесятью восьмьюдесятью"),
        (90, "девяносто девяноста"),
    )
)
_HUNDREDS = _forms(
    (
        (100, "сто ста"),
        (200, "двести двухсот двумстам двумястами двухстах"),
        (300, "триста трехсот тремстам тремястами трехстах"),
        (400, "четыреста четырехсот четыремстам четырьмястами четырехстах"),
        _hundreds(500, "пять", "пяти", "пятью"),
        _hundreds(600, "шесть", "шести", "шестью"),
        _hundreds(700, "семь", "семи", "семью"),
        (800, "восемьсот восьмисот восьмистам восемьюстами восьмьюстами восьмистах"),
        _hundreds(900, "девять", "девяти", "девятью"),
    )
)
_PLACES: dict[str, tuple[int, _Place]] = {
    form: (value, place)
    for table, place in (
        (_ONES, _Place.ONES),
        (_TEENS, _Place.TEENS),
        (_TENS, _Place.TENS),
        (_HUNDREDS, _Place.HUNDREDS),
    )
    for form, value in table.items()
}
_MASCULINE_ENDINGS = ("", "а", "у", "ом", "е", "ы", "ов", "ам", "ами", "ах")
_MULTIPLIERS: dict[str, int] = {
    **_forms(
        (
            (
                1000,
                "тысяча тысячи тысяче тысячу тысячей тысячью тысяч "
                "тысячам тысячами тысячах",
            ),
        )
    ),
    **{
        f"{stem}{ending}": value
        for value, stem in ((10**6, "миллион"), (10**9, "миллиард"))
        for ending in _MASCULINE_ENDINGS
    },
}
# Не раскладываются по разрядам и стоят только в начале: «полутора миллионов».
_STANDALONE: dict[str, Fraction] = {
    "полтора": Fraction(3, 2),
    "полторы": Fraction(3, 2),
    "полутора": Fraction(3, 2),
    "полтораста": Fraction(150),
    "полутораста": Fraction(150),
}
# «ноль целых одна десятая», «две целых пять сотых»
_WHOLE_WORDS = frozenset({"целая", "целой", "целую", "целых", "целые"})
_DENOMINATORS: dict[str, int] = {
    f"{stem}{ending}": value
    for value, stem in ((10, "десят"), (100, "сот"), (1000, "тысячн"))
    for ending in ("ая", "ой", "ую", "ых", "ые", "ыми", "ым")
}

_ORDINAL_STEMS = (
    (1, "перв"),
    (2, "втор"),
    (4, "четверт"),
    (5, "пят"),
    (6, "шест"),
    (7, "седьм"),
    (8, "восьм"),
    (9, "девят"),
    (10, "десят"),
    (11, "одиннадцат"),
    (12, "двенадцат"),
    (13, "тринадцат"),
    (14, "четырнадцат"),
    (15, "пятнадцат"),
    (16, "шестнадцат"),
    (17, "семнадцат"),
    (18, "восемнадцат"),
    (19, "девятнадцат"),
    (20, "двадцат"),
    (30, "тридцат"),
)
# Формы, в которых порядковое стоит в датах и при слове «число»: «первое
# августа», «до пятого числа», «к десятому», «пятым числом», «в первом».
_ORDINALS: dict[str, int] = {
    **{
        f"{stem}{ending}": value
        for value, stem in _ORDINAL_STEMS
        for ending in ("ое", "ого", "ому", "ым", "ом")
    },
    **_forms(((3, "третье третьего третьему третьим третьем"),)),
}
_LAST_DAY_OF_MONTH = 31


def read_numeral(words: Sequence[str]) -> NumeralReading | None:
    """Количественное числительное в начале последовательности слов.

    Берётся самая длинная грамматически верная запись: «сорока пяти дней» — 45,
    «трёх пяти» — 3, «ноль целых одна десятая» — 1/10. None — первое слово
    не числительное.
    """
    folded = [_folded(word) for word in words]
    whole = _read_integer(folded)
    if whole is None:
        return None
    return _read_fraction(folded, whole) or whole


def read_ordinal(words: Sequence[str]) -> NumeralReading | None:
    """Порядковое от 1 до 31 в начале слов: «пятого», «двадцать первого»."""
    folded = [_folded(word) for word in words[:2]]
    if not folded:
        return None
    if folded[0] in _ORDINALS:
        return NumeralReading(Fraction(_ORDINALS[folded[0]]), 1)
    if len(folded) < 2 or folded[0] not in _TENS or folded[1] not in _ORDINALS:
        return None
    value = _TENS[folded[0]] + _ORDINALS[folded[1]]
    if _ORDINALS[folded[1]] >= 10 or value > _LAST_DAY_OF_MONTH:
        return None
    return NumeralReading(Fraction(value), 2)


def multiplier_value(word: str) -> int | None:
    """«тысяч» — 1000, «миллионов» — 1 000 000; для записи «12 миллионов»."""
    return _MULTIPLIERS.get(_folded(word))


def _folded(word: str) -> str:
    return word.lower().replace("ё", "е")


def _read_integer(words: Sequence[str]) -> NumeralReading | None:
    if words and words[0] in _STANDALONE:
        return _with_multiplier(_STANDALONE[words[0]], words)
    total = 0
    # Тройка разрядов, которая сейчас набирается; None — ещё не начата.
    group: int | None = None
    last_place: _Place | None = None
    last_multiplier: int | None = None
    used = 0
    for word in words:
        if word in _MULTIPLIERS:
            multiplier = _MULTIPLIERS[word]
            # Множители строго убывают: «тысяча миллионов» — два числа.
            if last_multiplier is not None and multiplier >= last_multiplier:
                break
            total += (1 if group is None else group) * multiplier
            group, last_place, last_multiplier = None, None, multiplier
        elif word in _PLACES:
            value, place = _PLACES[word]
            if last_place is not None and not _may_follow(place, last_place):
                break
            group = (group or 0) + value
            last_place = place
        else:
            break
        used += 1
    if used == 0:
        return None
    return NumeralReading(Fraction(total + (group or 0)), used)


def _may_follow(place: _Place, previous: _Place) -> bool:
    """Внутри тройки разряды только убывают: «сорок сорок» — два числа."""
    if previous is _Place.HUNDREDS:
        return place is not _Place.HUNDREDS
    return previous is _Place.TENS and place is _Place.ONES


def _with_multiplier(value: Fraction, words: Sequence[str]) -> NumeralReading:
    if len(words) > 1 and words[1] in _MULTIPLIERS:
        return NumeralReading(value * _MULTIPLIERS[words[1]], 2)
    return NumeralReading(value, 1)


def _read_fraction(
    words: Sequence[str], whole: NumeralReading
) -> NumeralReading | None:
    """«… целых одна десятая» после целой части; None — дробной части нет."""
    rest = words[whole.words :]
    if not rest or rest[0] not in _WHOLE_WORDS:
        return None
    numerator = _read_integer(rest[1:])
    if numerator is None:
        return None
    used = 1 + numerator.words
    if len(rest) <= used or rest[used] not in _DENOMINATORS:
        return None
    value = whole.value + numerator.value / _DENOMINATORS[rest[used]]
    return NumeralReading(value, whole.words + used + 1)

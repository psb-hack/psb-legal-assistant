"""Величины в тексте: суммы, проценты, сроки, дни месяца и даты.

Факт — это значение вместе с единицей и оператором: «не более 5 рабочих дней»
и «5 дней» — разные факты. Номера пунктов, статей и документов, ИНН и адреса
величинами не считаются: это реквизиты, их проверяет белый список ссылок.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from lex_dates import MONTH_WORD_PATTERN, PartialDate, calendar_period, find_dates
from lex_dates import month_number as month_by_word
from lex_numerals import multiplier_value, read_numeral, read_ordinal


class Unit(Enum):
    RUBLE = "ruble"
    PERCENT = "percent"
    # Четыре разные единицы: друг в друга дни не пересчитываются, а «дни» без
    # уточнения не равны ни календарным, ни рабочим.
    CALENDAR_DAY = "calendar_day"
    WORKING_DAY = "working_day"
    BANKING_DAY = "banking_day"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"
    HOUR = "hour"
    # «5-го числа»: не дата и не срок.
    DAY_OF_MONTH = "day_of_month"
    # Число без известной единицы: «500 000», «в 3 экземплярах».
    NONE = "none"


class QuantityKind(Enum):
    MONEY = "money"
    PERCENT = "percent"
    DURATION = "duration"
    DAY_OF_MONTH = "day_of_month"
    NUMBER = "number"


class Bound(Enum):
    """Оператор при величине. «до 7%» и «7%» — разные утверждения."""

    EXACT = "exact"
    # «не более», «не позднее», «не может превышать»
    AT_MOST = "at_most"
    # «не менее», «не ранее»
    AT_LEAST = "at_least"
    # «свыше», «более», «превышает»
    ABOVE = "above"
    # «менее», «ранее»
    BELOW = "below"
    # «до»: входит ли сама граница, из слова не видно.
    UP_TO = "up_to"
    # «от 3 до 5», «3–5»
    RANGE = "range"
    # «порог — 15 000 000»: граница названа, направление — нет.
    THRESHOLD = "threshold"


class Vat(Enum):
    INCLUDED = "included"
    EXCLUDED = "excluded"


@dataclass(frozen=True, slots=True)
class Quantity:
    value: Fraction
    # Верхняя граница диапазона; у остальных величин None.
    upper: Fraction | None
    unit: Unit
    bound: Bound
    # None — про НДС рядом с суммой ничего не сказано.
    vat: Vat | None
    # Место в тексте: число вместе с единицей, без оператора.
    start: int
    end: int

    @property
    def kind(self) -> QuantityKind:
        return _KINDS.get(self.unit, QuantityKind.DURATION)


@dataclass(frozen=True, slots=True)
class DateFact:
    period: PartialDate
    bound: Bound
    start: int
    end: int


Fact = Quantity | DateFact

_KINDS = {
    Unit.RUBLE: QuantityKind.MONEY,
    Unit.PERCENT: QuantityKind.PERCENT,
    Unit.DAY_OF_MONTH: QuantityKind.DAY_OF_MONTH,
    Unit.NONE: QuantityKind.NUMBER,
}

# Виды пробела между разрядами и тире приводятся к одному знаку, длина текста
# не меняется: места величин относятся к исходной строке.
_FOLDED_CHARACTERS = str.maketrans(
    {
        "ё": "е",
        " ": " ",
        " ": " ",
        " ": " ",
        " ": " ",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
    }
)
# Знак на месте уже разобранного или вырезанного: не буква, не цифра, не пробел,
# поэтому соседние числа через него не склеиваются.
_MASK = "\x1f"

_WORD = re.compile(r"[а-я]+")
# Столько слов хватает самому длинному числительному с дробной частью.
_LONGEST_NUMERAL = 16
# Числительное прописью без единицы — факт только начиная с этого значения:
# «один из», «две стороны», «раз в три года» величин не называют.
_SMALLEST_BARE_NUMERAL = 10

# «первого августа 2024 года»: день прописью перед месяцем и годом
_MONTH_AND_YEAR = re.compile(
    rf"\s+({MONTH_WORD_PATTERN})\s+((?:19|20)\d{{2}})(?!\d)"
    r"(?:\s*(?:г\.|года?(?![а-я])))?"
)
# «5-го числа», «25 числа»
_DAY_OF_MONTH = re.compile(
    r"(?<![\w.,/-])(\d{1,2})(?:-(?:го|ого|е|ое|му|м))?\s+числ[аоу](?![а-я])"
)
# «пятого числа»: слово после порядкового
_DAY_WORD_AFTER = re.compile(r"\s+числ[аоу](?![а-я])")

# Реквизиты и адреса. Число в них — не величина.
_NOT_QUANTITIES = (
    # «№ П-17/2024», «склад № 3», «N 152-ФЗ»
    re.compile(r"(?:№|(?<![a-zа-я])n(?=\s*\d))\s*[\w./-]*\w"),
    # «п. 3.1», «пп. 3.1, 3.2», «ст. 13.11», «ч. 5», «статьи 70», «пунктом 3.4»
    re.compile(
        r"(?<![а-я])(?:(?:п|пп|ст|ч|подп|абз|разд|гл|прил|прим)\."
        r"|(?:пункт|подпункт|стать|част|абзац|раздел|глав|приложени|примечани)"
        r"[а-я]*)\s*\d+(?:\.\d+)*(?:\s*(?:,|и|-)\s*\d+(?:\.\d+)+)*"
    ),
    # «152-ФЗ», «П-17/2024», «17/2024», «14-П», «5-го» без слова «число»
    re.compile(
        r"(?<![\w-])(?:[a-zа-я]{1,4}-\d[\w/-]*|\d+(?:/\d+)+"
        r"|\d+-[a-zа-я]{1,3}(?![a-zа-я]))"
    ),
    # «ИНН 7701234567», «ОГРН 1027700132195», «р/с 40702810…»
    re.compile(r"(?<![а-я])(?:инн|огрн|огрнип|кпп|бик|окпо|р/с|к/с)[\s:]*\d+"),
    # «каб. 412», «офис 12», «д. 5», «этаж 2», но не «помещение 1 200 кв. м»
    re.compile(
        r"(?<![а-я])(?:(?:каб|оф|кв|корп|стр|д|эт|пом|ком)\."
        r"|(?:кабинет|офис|квартир|корпус|строени|дом|этаж|помещени|комнат|склад"
        r"|секци|бокс)[а-я]*)\s*\d+[а-я]?(?![а-я\d]| \d{3})"
    ),
)
# «1)», «2)» — пункты перечисления, если скобка не закрывает «(не более 30)»
_LIST_MARKER = re.compile(r"(?<![\w.,/])\d{1,2}\)")

# «450 000», «10 000 000», «0,1», «12». После числа не идёт продолжение
# идентификатора: «30-дневный», «17/2024».
_NUMBER = re.compile(
    r"(?<![\w.,/])(?<![a-zа-я]-)(?P<whole>\d{1,3}(?: \d{3})+(?!\d)|\d+)"
    r"(?:(?P<mark>[.,])(?P<fraction>\d+))?(?![\w/]|-[a-zа-я]|[.,]\d)"
)
# «450 тыс. руб.», «12 млн», «1,5 млрд»
_SHORT_MULTIPLIER = re.compile(r"\s*(тыс|млн|млрд)(?:\.|(?![а-я]))")
_SHORT_MULTIPLIERS = {"тыс": 10**3, "млн": 10**6, "млрд": 10**9}
# «12 миллионов»: слово сразу после числа
_NEXT_WORD = re.compile(r"\s*([а-я]+)")
# «(четыреста пятьдесят тысяч)» после числа цифрами: та же величина прописью
_IN_BRACKETS = re.compile(r"\s*\(([^()]*)\)")

_DAY_WORD = r"(?:день|дня|дней|дню|днем|дням|днями|днях|дни)(?![а-я])"
_NOUN_ENDINGS = r"(?:а|ов|ев|у|ом|ем|е|ы|ам|ами|ах)?(?![а-я])"
# Порядок важен: «рабочих дней» раньше, чем просто «дней».
_UNIT_WORDS = (
    (Unit.PERCENT, rf"%|процент{_NOUN_ENDINGS}"),
    (Unit.RUBLE, r"руб(?:\.|л[а-я]+|(?![а-я]))|₽"),
    (Unit.CALENDAR_DAY, rf"календарн[а-я]+\s+{_DAY_WORD}"),
    (Unit.WORKING_DAY, rf"рабоч[а-я]+\s+{_DAY_WORD}"),
    (Unit.BANKING_DAY, rf"банковск[а-я]+\s+{_DAY_WORD}"),
    (Unit.DAY, _DAY_WORD),
    (Unit.WEEK, r"недел[а-я]+"),
    (Unit.MONTH, rf"месяц{_NOUN_ENDINGS}|мес\."),
    (Unit.YEAR, r"(?:год(?:а|у|ом|е|ы)?|лет)(?![а-я])"),
    (Unit.HOUR, rf"час{_NOUN_ENDINGS}"),
)
# «рублей», «%», «рабочих дней», «месяцев» сразу после числа
_UNIT = re.compile(
    r"\s*(?:"
    + "|".join(f"(?P<{unit.value}>{words})" for unit, words in _UNIT_WORDS)
    + ")"
)

# Порядок важен: «не более» раньше, чем «более».
_BOUND_WORDS = (
    (
        Bound.AT_MOST,
        r"не (?:более|больше|свыше|выше|дольше|позднее|позже)|не превыша[а-я]+"
        r"|не (?:может|могут|должен|должна|должно|должны) превышать|максимум",
    ),
    (Bound.AT_LEAST, r"не (?:менее|меньше|ниже|короче|ранее|раньше)|минимум"),
    (Bound.ABOVE, r"свыше|более|больше|выше|дольше|позднее|позже|превыша[а-я]+"),
    (Bound.BELOW, r"менее|меньше|ниже|ранее|раньше"),
    (Bound.UP_TO, r"до"),
    (
        Bound.THRESHOLD,
        r"(?:порог|предел|лимит)[а-я]*(?:\s*[-:]| (?:в|составляет|равен))?",
    ),
)
# «не более чем на 7%», «свыше 10 000 000», «порог — 15 000 000»: слова прямо
# перед числом
_BOUND = re.compile(
    (
        r"(?<![а-я])(?:"
        + "|".join(f"(?P<{bound.value}>{words})" for bound, words in _BOUND_WORDS)
        + r")(?: чем)?(?: (?:на|в|за|через))?"
    ).replace(" ", r"\s+")
    + r"\s*$"
)
# Сколько знаков перед числом читается в поиске оператора.
_BOUND_WINDOW = 48
# «от 3 до 5 дней»: предлог перед нижней границей диапазона
_RANGE_OPENING = re.compile(r"(?<![а-я])от\s*$")
_RANGE_DASH = "-"
_RANGE_WORD = "до"

# «…рублей в месяц, включая НДС», «без НДС», «кроме того НДС 20%»
_VAT = re.compile(
    r"[^.;\d]{0,40}?(?:(?P<included>включая ндс|в том числе ндс|в т\. ?ч\. ндс"
    r"|с учетом ндс|с ндс)|(?P<excluded>без учета ндс|без ндс|не включая ндс"
    r"|кроме того,? ндс|ндс не облагается))"
)


def find_facts(text: str) -> tuple[Fact, ...]:
    """Величины и даты текста в порядке появления."""
    folded = _folded(text)
    spelled_dates = _spelled_dates(folded)
    remaining = _masked(folded, spelled_dates)
    dates = [
        DateFact(
            found.period,
            _date_bound(remaining, found.start),
            found.start,
            found.end,
        )
        for found in find_dates(remaining)
    ]
    remaining = _masked(remaining, dates)
    days = _digit_days_of_month(remaining)
    remaining = _masked(remaining, days)
    for pattern in _NOT_QUANTITIES:
        remaining = pattern.sub(lambda match: _MASK * len(match.group()), remaining)
    remaining = _without_list_markers(remaining)
    numbers, remaining = _digit_quantities(remaining)
    quantities = sorted([*days, *numbers, *_spelled_quantities(remaining)], key=_start)
    facts: list[Fact] = [*spelled_dates, *dates, *_with_ranges(quantities, folded)]
    return tuple(sorted(facts, key=_start))


def same_fact(first: Fact, second: Fact) -> bool:
    """Одна ли это величина: значение, единица, оператор, НДС; место не важно."""
    if isinstance(first, DateFact) or isinstance(second, DateFact):
        return (
            isinstance(first, DateFact)
            and isinstance(second, DateFact)
            and (first.period, first.bound) == (second.period, second.bound)
        )
    return (first.value, first.upper, first.unit, first.bound, first.vat) == (
        second.value,
        second.upper,
        second.unit,
        second.bound,
        second.vat,
    )


def _folded(text: str) -> str:
    # У отдельных знаков («İ») строчная форма длиннее: такие остаются как есть.
    lowered = "".join(ch.lower() if len(ch.lower()) == 1 else ch for ch in text)
    return lowered.translate(_FOLDED_CHARACTERS)


def _start(fact: Fact) -> int:
    return fact.start


def _masked(text: str, facts: Iterable[Fact]) -> str:
    for fact in facts:
        text = _masked_span(text, fact.start, fact.end)
    return text


def _masked_span(text: str, start: int, end: int) -> str:
    return f"{text[:start]}{_MASK * (end - start)}{text[end:]}"


def _without_list_markers(text: str) -> str:
    for match in _LIST_MARKER.finditer(text):
        before = text[: match.start()]
        if before.rfind("(") <= before.rfind(")"):
            text = _masked_span(text, *match.span())
    return text


def _spelled_dates(text: str) -> list[DateFact]:
    words = list(_WORD.finditer(text))
    dates: list[DateFact] = []
    for index, word in enumerate(words):
        if dates and word.start() < dates[-1].end:
            # «двадцать первого марта»: «первого» уже вошло в дату.
            continue
        run = _adjacent_words(words, index, text, 2)
        day = read_ordinal([one.group() for one in run])
        if day is None:
            continue
        tail = _MONTH_AND_YEAR.match(text, run[day.words - 1].end())
        if tail is None:
            continue
        period = calendar_period(
            int(tail.group(2)), month_by_word(tail.group(1)), str(day.value)
        )
        if period is not None:
            bound = _date_bound(text, word.start())
            dates.append(DateFact(period, bound, word.start(), tail.end()))
    return dates


def _date_bound(text: str, start: int) -> Bound:
    bound = _bound_before(text, start)
    # «Порог» относится к суммам и срокам, не к датам.
    return Bound.EXACT if bound is Bound.THRESHOLD else bound


def _bound_before(text: str, start: int) -> Bound:
    found = _BOUND.search(text[max(0, start - _BOUND_WINDOW) : start])
    if found is None or found.lastgroup is None:
        return Bound.EXACT
    return Bound(found.lastgroup)


def _digit_days_of_month(text: str) -> list[Quantity]:
    return [
        Quantity(
            value=Fraction(match.group(1)),
            upper=None,
            unit=Unit.DAY_OF_MONTH,
            bound=_bound_before(text, match.start()),
            vat=None,
            start=match.start(),
            end=match.end(),
        )
        for match in _DAY_OF_MONTH.finditer(text)
    ]


def _digit_quantities(text: str) -> tuple[list[Quantity], str]:
    """Числа цифрами с множителем, единицей и дублем прописью в скобках.

    Вторым значением — текст, в котором разобранное закрыто: числительные
    из скобок второй раз фактом не станут.
    """
    quantities: list[Quantity] = []
    consumed_until = 0
    for match in _NUMBER.finditer(text):
        if match.start() < consumed_until:
            continue
        value = Fraction(
            f"{match['whole'].replace(' ', '')}.{match['fraction'] or '0'}"
        )
        multiplier, position = _multiplier_at(text, match.end())
        unit, position = _unit_at(text, position)
        position = _after_spelled_duplicate(text, position)
        if unit is None:
            unit, position = _unit_at(text, position)
        if match["mark"] == "." and unit is None and multiplier == 1:
            # «3.1» без единицы — номер пункта: десятичный знак в русском
            # тексте — запятая.
            continue
        unit = unit or Unit.NONE
        quantities.append(
            Quantity(
                value=value * multiplier,
                upper=None,
                unit=unit,
                bound=_bound_before(text, match.start()),
                vat=_vat_at(text, position, unit),
                start=match.start(),
                end=position,
            )
        )
        consumed_until = position
    return quantities, _masked(text, quantities)


def _multiplier_at(text: str, position: int) -> tuple[int, int]:
    short = _SHORT_MULTIPLIER.match(text, position)
    if short is not None:
        return _SHORT_MULTIPLIERS[short.group(1)], short.end()
    word = _NEXT_WORD.match(text, position)
    value = None if word is None else multiplier_value(word.group(1))
    if word is None or value is None:
        return 1, position
    return value, word.end()


def _unit_at(text: str, position: int) -> tuple[Unit | None, int]:
    found = _UNIT.match(text, position)
    if found is None or found.lastgroup is None:
        return None, position
    return Unit(found.lastgroup), found.end()


def _after_spelled_duplicate(text: str, position: int) -> int:
    brackets = _IN_BRACKETS.match(text, position)
    if brackets is None or read_numeral(_WORD.findall(brackets.group(1))) is None:
        return position
    return brackets.end()


def _vat_at(text: str, position: int, unit: Unit) -> Vat | None:
    if unit not in (Unit.RUBLE, Unit.NONE):
        return None
    found = _VAT.match(text, position)
    if found is None or found.lastgroup is None:
        return None
    return Vat(found.lastgroup)


def _spelled_quantities(text: str) -> list[Quantity]:
    words = list(_WORD.finditer(text))
    quantities: list[Quantity] = []
    index = 0
    while index < len(words):
        run = _adjacent_words(words, index, text, _LONGEST_NUMERAL)
        spelled = [word.group() for word in run]
        start = words[index].start()
        ordinal = read_ordinal(spelled)
        numeral = read_numeral(spelled)
        if ordinal is not None:
            day_word = _DAY_WORD_AFTER.match(text, run[ordinal.words - 1].end())
            if day_word is not None:
                quantities.append(
                    Quantity(
                        value=ordinal.value,
                        upper=None,
                        unit=Unit.DAY_OF_MONTH,
                        bound=_bound_before(text, start),
                        vat=None,
                        start=start,
                        end=day_word.end(),
                    )
                )
            index += ordinal.words
        elif numeral is not None:
            unit, end = _unit_at(text, run[numeral.words - 1].end())
            if unit is not None or numeral.value >= _SMALLEST_BARE_NUMERAL:
                unit = unit or Unit.NONE
                quantities.append(
                    Quantity(
                        value=numeral.value,
                        upper=None,
                        unit=unit,
                        bound=_bound_before(text, start),
                        vat=_vat_at(text, end, unit),
                        start=start,
                        end=end,
                    )
                )
            index += numeral.words
        else:
            index += 1
    return quantities


def _adjacent_words(
    words: Sequence[re.Match[str]], index: int, text: str, limit: int
) -> list[re.Match[str]]:
    """Слова подряд, между которыми только пробелы: запятая обрывает число."""
    run = [words[index]]
    for word in words[index + 1 : index + limit]:
        if text[run[-1].end() : word.start()].strip():
            break
        run.append(word)
    return run


def _with_ranges(quantities: Sequence[Quantity], text: str) -> list[Quantity]:
    """«от 3 до 5 дней» и «3–5 дней» — одна величина с двумя границами."""
    merged: list[Quantity] = []
    for quantity in quantities:
        lower = merged[-1] if merged else None
        if lower is not None and _is_range(lower, quantity, text):
            merged[-1] = Quantity(
                value=lower.value,
                upper=quantity.value,
                unit=quantity.unit,
                bound=Bound.RANGE,
                vat=quantity.vat,
                start=lower.start,
                end=quantity.end,
            )
        else:
            merged.append(quantity)
    return merged


def _is_range(lower: Quantity, upper: Quantity, text: str) -> bool:
    if lower.upper is not None or lower.unit not in (Unit.NONE, upper.unit):
        return False
    between = text[lower.end : upper.start].strip()
    if between == _RANGE_DASH:
        return True
    opened = _RANGE_OPENING.search(text[: lower.start]) is not None
    return between == _RANGE_WORD and opened

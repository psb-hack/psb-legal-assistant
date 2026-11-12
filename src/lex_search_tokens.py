"""Токены лексического поиска: символьные n-граммы слов и целые номера.

Морфологического разбора в разрешённых пакетах нет, поэтому словоформы сближают
общие n-граммы: у «поставки» и «поставщику» совпадают «_пос», «пост», «оста»,
«став». Номера на n-граммы не режутся: «П-17/2024» и «П-17/2023» отличаются
одним знаком и называют разные документы.
"""

from __future__ import annotations

import re

from lex_normalization import comparison_key

WORD_NGRAM_LENGTH = 4
# Отличает начало и конец слова от середины: «_дог» и «ора_» в слове «договора».
# В тексте токена этот знак встретиться не может: шаблон токена его не захватывает.
WORD_BOUNDARY = "_"

# «450 000», «10 000 000»: разряды числа через пробел (неразрывный пробел ключ
# сравнения уже заменил обычным). Число начинается с группы до трёх цифр, которая
# не продолжает номер: в «п. 3.2 450 000» это «450», а не «2».
_SPACED_NUMBER = re.compile(r"(?<![\d.,/-])\d{1,3}(?: \d{3})+(?!\d)")
# «п-17/2024», «13.11», «152-фз», «альфа-трейд», «договора»: буквы и цифры,
# соединённые дефисом, косой чертой или точкой. Запятая соединяет только цифры:
# «0,1», но не «п. 3.1, 3.2».
_TOKEN = re.compile(r"[^\W_]+(?:(?:[-/.]|(?<=\d),(?=\d))[^\W_]+)*")
# «п17/2024» → «п», «17», «2024»: подряд идущие цифры или подряд идущие буквы.
_RUN = re.compile(r"\d+|[^\W\d_]+")
_DIGIT = re.compile(r"\d")


def search_tokens(text: str) -> tuple[str, ...]:
    """Токены текста или запроса; у индекса и запроса разбор один и тот же."""
    key = _SPACED_NUMBER.sub(_without_spaces, comparison_key(text))
    tokens: list[str] = []
    for match in _TOKEN.finditer(key):
        tokens.extend(_token_forms(match.group()))
    return tuple(tokens)


def word_ngrams(word: str) -> list[str]:
    """«срок» → «_сро», «срок», «рок_»; слово короче n-граммы даёт один токен."""
    marked = f"{WORD_BOUNDARY}{word}{WORD_BOUNDARY}"
    if len(marked) <= WORD_NGRAM_LENGTH:
        return [marked]
    return [
        marked[start : start + WORD_NGRAM_LENGTH]
        for start in range(len(marked) - WORD_NGRAM_LENGTH + 1)
    ]


def _without_spaces(match: re.Match[str]) -> str:
    return match.group().replace(" ", "")


def _token_forms(token: str) -> list[str]:
    runs: list[str] = _RUN.findall(token)
    if _DIGIT.search(token) is None:
        # «альфа-трейд» — два слова: дефис и пробел в наименовании равнозначны.
        return [gram for word in runs for gram in word_ngrams(word)]
    # Номер остаётся целым токеном: точное совпадение весит больше совпадения
    # частей. Части нужны для другой записи того же номера: «п17/2024», «ФЗ 152».
    forms = [token]
    if len(runs) > 1:
        for run in runs:
            forms.extend([run] if _DIGIT.match(run) else word_ngrams(run))
    return forms

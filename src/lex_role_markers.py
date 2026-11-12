"""Маркеры ролей и токены чат-шаблонов в данных: поиск и замена видимым знаком.

Вопрос, история диалога и текст базы могут содержать «<|im_start|>», «[INST]»
или «Ассистент:» в начале строки. Прокси может превратить такую запись
в управляющий токен, а модель — принять её за смену роли. Маркер не удаляется:
его края заменяются знаками ‹ и ›, текст остаётся читаемым.

Маркеры ищутся по ключу детекции, поэтому полноширинные знаки, другой регистр
и смешение алфавитов их не прячут; правится сам текст.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from lex_normalization import AlignedKey, detection_alignment

OPENING_SIGN = "‹"
CLOSING_SIGN = "›"
_EDGE_SIGNS = (("open", OPENING_SIGN), ("close", CLOSING_SIGN))
_SAME_LINE_SPACE = r"[^\S\n]"

# Роли чата и названия участников диалога. Разбор истории (lex_dialogue) берёт
# свой, более узкий список меток: реплику начинают только пользователь и
# ассистент. Сторон договора («Покупатель», «Арендодатель») здесь нет намеренно.
_ROLE = (
    r"(?:system|assistant|user|human|developer|tool|ai|bot"
    r"|система|ассистент|пользователь|юзер|бот|ии)"
)
_ROLE_PHRASE = (
    rf"(?:system{_SAME_LINE_SPACE}+(?:message|prompt|instruction)"
    rf"|системн\w+{_SAME_LINE_SPACE}+(?:сообщение|промпт|инструкци\w+)|{_ROLE})"
)


class RoleMarkerKind(Enum):
    CHAT_TEMPLATE = "токен чат-шаблона"
    TAG = "тег"
    BRACKETED_ROLE = "роль в квадратных скобках"
    ROLE_HEADING = "заголовок с ролью"
    ROLE_LABEL = "метка роли"


@dataclass(frozen=True, slots=True)
class RoleMarker:
    kind: RoleMarkerKind
    # Как маркер записан в тексте: с этим он попадает в журнал.
    text: str


@dataclass(frozen=True, slots=True)
class _Edit:
    start: int
    end: int
    sign: str


# Группа open — отрезок, на место которого встаёт ‹, группа close — ›. Пустая
# группа — вставка знака: так слово роли берётся в ‹…› и остаётся в тексте.
_PATTERNS = (
    (
        RoleMarkerKind.CHAT_TEMPLATE,
        # «<|im_start|>», «<｜User｜>», «</｜DSML｜tool_calls>» — края токена
        # по отдельности: незакрытый «<|im_start» тоже теряет вид токена.
        re.compile(r"(?P<open><\||<(?=/\|))|(?P<close>\|>)"),
    ),
    (
        RoleMarkerKind.TAG,
        # «<think>», «</s>», «<tool_result>», «<<SYS>>», «<system role="x">»
        # и теги наших блоков данных. Имя латиницей: «<…>», «<1>», «<адрес>»
        # в юридическом тексте остаются как есть.
        re.compile(
            rf"(?P<open><{{1,2}})/?[a-z][\w.:-]{{0,40}}"
            rf"(?:{_SAME_LINE_SPACE}[^<>\n]{{0,200}})?/?(?P<close>>{{1,2}})"
        ),
    ),
    (
        RoleMarkerKind.BRACKETED_ROLE,
        # «[INST]», «[/INST]», «[SYSTEM]», «[system_prompt]», «[Ассистент]»;
        # «[1]» и «[наименование]» — нет.
        re.compile(
            rf"(?P<open>\[){_SAME_LINE_SPACE}*/?{_SAME_LINE_SPACE}*(?:{_ROLE}|inst|sys)"
            rf"(?:(?![\]\n])[\W_][^\]\n]{{0,40}})?(?P<close>\])"
        ),
    ),
    (
        RoleMarkerKind.ROLE_HEADING,
        # «### Instruction:», «## Response», «### Ассистент: хорошо»; заголовок
        # документа «### Инструкция по охране труда» — нет.
        re.compile(
            rf"^{_SAME_LINE_SPACE}*(?P<open>)\#{{1,6}}(?P<close>){_SAME_LINE_SPACE}*"
            rf"(?:{_ROLE}|instructions?|input|response|инструкци\w*|ответ)"
            rf"{_SAME_LINE_SPACE}*(?::|$)",
            re.MULTILINE,
        ),
    ),
    (
        RoleMarkerKind.ROLE_LABEL,
        # «system: …», «Ассистент: хорошо», «**User:** …», «> Бот: …»,
        # «…оплаты. Пользователь: а срок?» — в начале строки или предложения.
        re.compile(
            rf"(?:^{_SAME_LINE_SPACE}*[>*_-]{{0,3}}{_SAME_LINE_SPACE}*"
            rf"|[.!?\"]{_SAME_LINE_SPACE}+)"
            rf"(?P<open>){_ROLE_PHRASE}(?P<close>)[*_]{{0,2}}{_SAME_LINE_SPACE}*:",
            re.MULTILINE,
        ),
    ),
)


def find_role_markers(text: str) -> tuple[RoleMarker, ...]:
    aligned = detection_alignment(text)
    found: list[RoleMarker] = []
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(aligned.key):
            start, end = aligned.source_span(*match.span())
            found.append(RoleMarker(kind, text[start:end].strip()))
    return tuple(found)


def neutralize_role_markers(text: str) -> str:
    """Текст, в котором края каждого маркера заменены знаками ‹ и ›."""
    aligned = detection_alignment(text)
    for _, pattern in _PATTERNS:
        edits = _edge_edits(pattern, aligned)
        if edits:
            text = _with_edits(text, edits)
            # Позиции сдвинулись: следующий класс ищется по новому ключу.
            aligned = detection_alignment(text)
    return text


def _edge_edits(pattern: re.Pattern[str], aligned: AlignedKey) -> list[_Edit]:
    edits: list[_Edit] = []
    for match in pattern.finditer(aligned.key):
        for group, sign in _EDGE_SIGNS:
            if match.group(group) is not None:
                start, end = aligned.source_span(*match.span(group))
                edits.append(_Edit(start, end, sign))
    return edits


def _with_edits(text: str, edits: list[_Edit]) -> str:
    # С конца: правка не сдвигает позиции тех, что стоят левее.
    for edit in sorted(edits, key=lambda item: item.start, reverse=True):
        text = text[: edit.start] + edit.sign + text[edit.end :]
    return text

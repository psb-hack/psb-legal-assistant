"""Детекторы, которые не читают смысл: разметка, место фрагмента, маскировка."""

from __future__ import annotations

import re
from dataclasses import dataclass

from lex_detection import Fragment, Signal, SignalStrength
from lex_documents import HiddenContentKind
from lex_normalization import TextFindingKind

STRUCTURE_DETECTOR_NAME = "структура"
CONCEALMENT_DETECTOR_NAME = "маскировка"

FEATURE_CHAT_MARKUP = "разметка чат-шаблона"
FEATURE_ROLE_LABEL = "метка роли"
FEATURE_FAKE_DIALOGUE = "поддельный диалог"
FEATURE_HTML_COMMENT = "HTML-комментарий"
FEATURE_HIDDEN_TEXT = "скрытый текст DOCX"
FEATURE_AFTER_SIGNATURE = "текст после подписи"

# «<|im_start|>system», «[SYSTEM]», «[/INST]», «<<SYS>>», «### Instruction», «<system>»
_CHAT_MARKUP = re.compile(
    r"""<\|[a-z_]+\|>
      | \[/?(?:system|inst|assistant|user)\]
      | <<\s*sys\s*>>
      | </?(?:system|assistant|user)>
      | \#{2,}\s*(?:instruction|system|response|инструкци\w*|система)\b""",
    re.VERBOSE,
)
# «SYSTEM: режим отключён», «…оплаты. Ассистент: хорошо» — метка в начале предложения.
_MACHINE_ROLE_LABEL = re.compile(
    r"""(?:^|[.!?"]\s+)(?:system|assistant|система|ассистент)\s*:""", re.VERBOSE
)
# «Пользователь: какой срок?» — реплика человека; сигнал только рядом с меткой машины.
_HUMAN_ROLE_LABEL = re.compile(r"""(?:^|[.!?"]\s+)(?:user|пользователь)\s*:""")
_HTML_COMMENT_OPENING = "<!--"
# «Начальник юридического отдела П. С. Лебедев», «…директор А. С. Ковалёва.»,
# «Лебедев П. С.» — инициалы с фамилией в конце абзаца.
_SIGNATURE = re.compile(
    r"(?:[А-ЯЁ]\.\s?[А-ЯЁ]\.\s?[А-ЯЁ][а-яё-]+|[А-ЯЁ][а-яё-]+\s[А-ЯЁ]\.\s?[А-ЯЁ]\.?)\.?$"
)


def ends_with_signature(text: str) -> bool:
    return _SIGNATURE.search(text.strip()) is not None


class StructureDetector:
    def inspect(self, fragment: Fragment) -> Signal:
        key = fragment.text.detection_key
        strong: list[str] = []
        weak: list[str] = []
        if _CHAT_MARKUP.search(key):
            strong.append(FEATURE_CHAT_MARKUP)
        if _MACHINE_ROLE_LABEL.search(key):
            strong.append(FEATURE_ROLE_LABEL)
            if _HUMAN_ROLE_LABEL.search(key):
                strong.append(FEATURE_FAKE_DIALOGUE)
        if fragment.hidden is HiddenContentKind.HIDDEN_TEXT:
            strong.append(FEATURE_HIDDEN_TEXT)
        if _HTML_COMMENT_OPENING in fragment.text.clean:
            weak.append(FEATURE_HTML_COMMENT)
        if fragment.follows_signature:
            weak.append(FEATURE_AFTER_SIGNATURE)
        return _signal(STRUCTURE_DETECTOR_NAME, strong, weak)


@dataclass(frozen=True, slots=True)
class ConcealmentDetector:
    """Читает находки нормализации: что в тексте было спрятано от читателя."""

    # Одно слово со смешением алфавитов — опечатка, несколько в абзаце — маскировка.
    mixed_script_words_for_strong: int

    def inspect(self, fragment: Fragment) -> Signal:
        strong: list[str] = []
        weak: list[str] = []
        for finding in fragment.text.findings:
            is_strong = finding.kind in (
                TextFindingKind.TAG_CHARACTERS,
                TextFindingKind.BIDI_OVERRIDE,
            ) or (
                finding.kind is TextFindingKind.MIXED_SCRIPT_WORDS
                and finding.count >= self.mixed_script_words_for_strong
            )
            (strong if is_strong else weak).append(finding.kind.value)
        return _signal(CONCEALMENT_DETECTOR_NAME, strong, weak)


def _signal(detector: str, strong: list[str], weak: list[str]) -> Signal:
    if strong:
        strength = SignalStrength.STRONG
    elif weak:
        strength = SignalStrength.WEAK
    else:
        strength = SignalStrength.NONE
    return Signal(detector, strength, (*strong, *weak))

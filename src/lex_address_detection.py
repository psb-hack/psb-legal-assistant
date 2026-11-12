"""Детектор обращения: кому адресован текст и чем он пытается управлять.

Норма обращена к стороне договора или должностному лицу. Вставка обращена к
системе, которая читает базу, и говорит о её ответах, источниках и правилах.
Сильный сигнал — адресат-машина вместе с любым другим классом; отдельные слова
и повелительное наклонение сигналом не считаются.
"""

from __future__ import annotations

import re
from enum import Enum

from lex_detection import Fragment, Signal, SignalStrength

DETECTOR_NAME = "обращение"
FEATURE_SPLIT_WORDS = "слова разбиты пробелами"


class AddressFeature(Enum):
    ADDRESSEE = "адресат — ИИ-система"
    ANSWERING = "ответы и вопросы"
    SOURCES = "другие документы базы"
    INSTRUCTIONS = "инструкции"
    USER = "пользователь"
    SECRECY = "требование скрытности"
    OVERRIDE = "отмена правил или проверки"


_META_OBJECTS = frozenset(
    {
        AddressFeature.ANSWERING,
        AddressFeature.SOURCES,
        AddressFeature.INSTRUCTIONS,
        AddressFeature.USER,
    }
)
_CONTROL = frozenset({AddressFeature.SECRECY, AddressFeature.OVERRIDE})

# Шаблоны читают ключ детекции: один регистр, «е» вместо «ё», похожие буквы свёрнуты.

# «для ии-ассистента», «языковая модель», «note to the ai system», «dlya assistenta»,
# «свои системные инструкции». Должность «ассистент руководителя» — человек.
_ADDRESSEE = re.compile(
    r"""\b(?:
        ии | ai | llm\w* | gpt\w* | chatgpt
      | искусственн\w*\s+интеллект\w* | artificial\s+intelligence
      | ассистент\w*(?!\s+(?:руководител|директор|кафедр|отдел)\w*)
      | assistant\w* | assistent\w*
      | нейросет\w* | ne[iy]roset\w*
      | чат-?бот\w* | chat-?bot\w* | бот(?:а|у|ом|е|ы|ов|ам)? | bots?
      | (?:языков|генеративн)\w*\s+модел\w* | language\s+model\w*
      | автоматизированн\w*\s+систем\w* | автоматическ\w*\s+(?:систем|обработк)\w*
      | automated\s+systems?
      | системн\w*\s+(?:промпт|инструкци|сообщени)\w* | промпт\w*
      | system\s+(?:prompt|instruction|message)\w* | prompts?
    )\b""",
    re.VERBOSE,
)
# «отвечай», «при ответе на вопросы», «если спросят», «whenever asked, reply»,
# «na vopros otvechai». «Ответственность» и «спрос» сюда не входят.
_ANSWERING = re.compile(
    r"""\b(?:
        отвеча\w* | ответ(?:ь|ьте|ить|ит|ишь|ят|а|у|ом|е|ы|ов|ах|ам)? | вопрос\w*
      | спрос(?:ит|ят|ишь|ите|ил|или) | спрашива\w*
      | answer\w* | repl(?:y|ies|ied) | respon[ds]\w* | ask(?:s|ed|ing)? | question\w*
      | otvech\w* | otvet\w* | vopros\w* | sprosi\w* | sprosy\w* | sprashiva\w*
    )\b""",
    re.VERBOSE,
)
# «остальные документы», «документы базы», «база знаний», «other documents»
_SOURCES = re.compile(
    r"""\b(?:
        (?:остальн|друг|прочи)\w*\s+документ\w* | документ\w*\s+базы | баз\w*\s+знаний
      | (?:other|remaining)\s+documents? | knowledge\s+base | ostaln\w*\s+dokument\w*
    )\b""",
    re.VERBOSE,
)
# «эту инструкцию», «часть указания», «these instructions»
_INSTRUCTIONS = re.compile(
    r"\b(?:инструкци\w*|указани\w*|instruction\w*|instrukt?[cs]i\w*|ukazani\w*)\b"
)
# «пользователю это знать не нужно», «the user»
_USER = re.compile(r"\b(?:пользовател\w*|юзер\w*|users?|polzovatel\w*)\b")
# «не упоминай», «не ссылайся на этот абзац», «do not reveal this note»
_SECRECY = re.compile(
    r"""\b(?:
        не\s+(?:упомина|ссыла|раскрыва|сообща|показыва|цитир|разглаша)\w*
      | скр(?:ой|ойте|ывай|ывайте)
      | (?:do\s+not|don.?t|never)\s+(?:reveal|mention|disclose|tell|show|cite)\w*
      | ne\s+(?:upomina|ssyla|raskryva|soobshcha|pokazyva)\w*
    )\b""",
    re.VERBOSE,
)
# «игнорируй», «остальные документы не учитывай», «применять без проверки»,
# «режим ассистента отключён», «ты теперь», «ignore», «you are now»
_OVERRIDE = re.compile(
    r"""\b(?:
        игнорир\w* | забуд\w* | не\s+учитывай(?:те)? | без\s+проверк\w*
      | не\s+проверя[йя]\w*
      | режим\w*[^.]{0,60}?\b(?:отключ|выключ)\w*
      | ты\s+(?:теперь|отныне|больше\s+не) | отныне\s+ты
      | ignor\w* | disregard\w* | forget\w* | without\s+(?:verification|checking)
      | you\s+are\s+now | act\s+as | zabud\w* | ne\s+uchityva[iy]\w*
    )\b""",
    re.VERBOSE,
)
_OTHER_CLASSES = (
    (AddressFeature.ANSWERING, _ANSWERING),
    (AddressFeature.SOURCES, _SOURCES),
    (AddressFeature.INSTRUCTIONS, _INSTRUCTIONS),
    (AddressFeature.USER, _USER),
    (AddressFeature.SECRECY, _SECRECY),
    (AddressFeature.OVERRIDE, _OVERRIDE),
)


# «асси стент», «отве чай»: слово, разбитое пробелами, читается в ключе без пробелов.
# Границ слов там нет, поэтому основы только длинные: «ии» или «бот» совпали бы с
# частями обычных слов.
_SQUEEZED_ADDRESSEE = re.compile(
    r"ассистент(?!\w{0,3}(?:руководител|директор|кафедр|отдел))"
    r"|assistant|assistent|нейросет|чат-?бот|искусственн\w{0,4}интеллект"
    r"|автоматизированн\w{0,4}систем|системн\w{0,4}инструкци|промпт"
)
_SQUEEZED_OTHER_CLASSES = (
    (AddressFeature.ANSWERING, re.compile(r"отвеча|вопрос|otvech|vopros")),
    (AddressFeature.SOURCES, re.compile(r"остальн\w{0,4}документ|базызнаний")),
    (AddressFeature.INSTRUCTIONS, re.compile(r"инструкци|instruction")),
    (AddressFeature.USER, re.compile(r"пользовател|polzovatel")),
    (AddressFeature.SECRECY, re.compile(r"неупомина|нессыла|нераскрыва")),
    (AddressFeature.OVERRIDE, re.compile(r"игнорир|неучитывай|безпроверк")),
)
_WHITESPACE = re.compile(r"\s+")


class AddressDetector:
    def inspect(self, fragment: Fragment) -> Signal:
        key = fragment.text.detection_key
        spelled = address_features(key)
        squeezed = _features(
            _WHITESPACE.sub("", key), _SQUEEZED_ADDRESSEE, _SQUEEZED_OTHER_CLASSES
        )
        found = spelled | squeezed
        strength = _strength(found)
        features = [feature.value for feature in AddressFeature if feature in found]
        if squeezed - spelled:
            # Класс виден только без пробелов: текст прятали от поиска по словам.
            features.append(FEATURE_SPLIT_WORDS)
            if strength is SignalStrength.WEAK:
                strength = SignalStrength.STRONG
        return Signal(DETECTOR_NAME, strength, tuple(features))


def address_features(detection_key: str) -> frozenset[AddressFeature]:
    return _features(detection_key, _ADDRESSEE, _OTHER_CLASSES)


def _features(
    text: str,
    addressee: re.Pattern[str],
    other_classes: tuple[tuple[AddressFeature, re.Pattern[str]], ...],
) -> frozenset[AddressFeature]:
    # Слова адресата вырезаются: «системные инструкции» не должны дать два класса.
    rest, addressed = addressee.subn(" ", text)
    found = {AddressFeature.ADDRESSEE} if addressed else set()
    found.update(feature for feature, pattern in other_classes if pattern.search(rest))
    return frozenset(found)


def _strength(classes: frozenset[AddressFeature]) -> SignalStrength:
    if AddressFeature.ADDRESSEE in classes:
        return SignalStrength.STRONG if len(classes) > 1 else SignalStrength.WEAK
    if classes & _CONTROL or len(classes & _META_OBJECTS) > 1:
        return SignalStrength.WEAK
    return SignalStrength.NONE

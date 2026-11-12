"""Санитайзер базы: решение о карантине для каждого абзаца положения.

Документы не меняются. Результат — журнал решений с причинами; текст положения
без абзацев в карантине выдаёт admitted_text.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from lex_address_detection import AddressDetector
from lex_detection import Detector, Fragment, Signal, SignalStrength
from lex_documents import (
    Document,
    DocumentId,
    HiddenContentKind,
    Provision,
    ProvisionId,
    context_heading,
)
from lex_knowledge_base import KnowledgeBase
from lex_normalization import NormalizedText, clean_text, normalize
from lex_settings import SanitizerThresholds
from lex_structure_detection import (
    ConcealmentDetector,
    StructureDetector,
    ends_with_signature,
)

WINDOW_DETECTOR_NAME = "окно соседних абзацев"
_LINE_BREAK = "\n"
_SMALLEST_WINDOW = 2


class Verdict(Enum):
    QUARANTINED = "quarantined"
    # Слабый сигнал: в индекс идёт, окончательно решает классификатор.
    SUSPECT = "suspect"
    # Слабый сигнал был, классификатор указаний ИИ-системе не нашёл.
    CLEARED = "cleared"


@dataclass(frozen=True, slots=True)
class Decision:
    """Запись журнала: что решено о фрагменте и по каким сигналам."""

    document_id: DocumentId
    provision_id: ProvisionId | None
    line: int
    hidden: HiddenContentKind | None
    verdict: Verdict
    signals: tuple[Signal, ...]
    original: str


@dataclass(frozen=True, slots=True)
class Screening:
    # Только фрагменты с сигналами; о чистых записей нет.
    decisions: tuple[Decision, ...]

    @property
    def quarantined(self) -> tuple[Decision, ...]:
        return tuple(
            item for item in self.decisions if item.verdict is Verdict.QUARANTINED
        )

    def quarantined_lines(self, provision_id: ProvisionId) -> frozenset[int]:
        return frozenset(
            item.line for item in self.quarantined if item.provision_id == provision_id
        )


@dataclass(frozen=True, slots=True)
class AdmittedProvision:
    """Положение в том виде, в каком санитайзер пропускает его к модели.

    Сборка контекста принимает только этот тип, а создаёт его admit: передать
    модели текст положения в обход карантина можно лишь собрав значение вручную.
    """

    id: ProvisionId
    heading: str
    text: str


@dataclass(frozen=True, slots=True)
class Detectors:
    per_fragment: tuple[Detector, ...]
    # Читает несколько соседних абзацев как один: вставку могут разбить на части.
    across_neighbours: Detector


def default_detectors(thresholds: SanitizerThresholds) -> Detectors:
    address = AddressDetector()
    return Detectors(
        per_fragment=(
            address,
            StructureDetector(),
            ConcealmentDetector(thresholds.mixed_script_words_for_strong),
        ),
        across_neighbours=address,
    )


def screen(
    base: KnowledgeBase, detectors: Detectors, thresholds: SanitizerThresholds
) -> Screening:
    decisions: list[Decision] = []
    for document in base.documents:
        visible = visible_fragments(document)
        window_signals = _window_signals(
            visible, detectors.across_neighbours, thresholds
        )
        for index, fragment in enumerate(visible):
            signals = _signals(fragment, detectors.per_fragment)
            if index in window_signals:
                signals.append(window_signals[index])
            decisions.extend(_decision(fragment, signals, thresholds))
        for fragment in hidden_fragments(document):
            signals = _signals(fragment, detectors.per_fragment)
            decisions.extend(_decision(fragment, signals, thresholds))
    return Screening(tuple(decisions))


def decide(
    signals: Sequence[Signal], thresholds: SanitizerThresholds
) -> Verdict | None:
    """Сильный сигнал любого детектора или слабые от нескольких — карантин."""
    strengths = [signal.strength for signal in signals]
    weak = strengths.count(SignalStrength.WEAK)
    if SignalStrength.STRONG in strengths:
        return Verdict.QUARANTINED
    if weak >= thresholds.weak_detectors_for_quarantine:
        return Verdict.QUARANTINED
    return Verdict.SUSPECT if weak else None


def admitted_text(
    document: Document, provision: Provision, screening: Screening
) -> str:
    """Чистый текст положения без абзацев в карантине: его читают индекс и модель.

    Пустая строка — положение в карантине целиком.
    """
    rejected = screening.quarantined_lines(document.provision_id(provision))
    lines = provision.text.split(_LINE_BREAK)
    kept = (line for index, line in enumerate(lines) if index not in rejected)
    return clean_text(_LINE_BREAK.join(kept))


def admit(
    document: Document, provision: Provision, screening: Screening
) -> AdmittedProvision | None:
    """None — положение в карантине целиком: в контекст ему нельзя."""
    text = admitted_text(document, provision, screening)
    if not text.strip():
        return None
    return AdmittedProvision(
        id=document.provision_id(provision),
        heading=clean_text(context_heading(document, provision)),
        text=text,
    )


def admitted_provisions(
    base: KnowledgeBase, screening: Screening
) -> tuple[AdmittedProvision, ...]:
    """Всё, что из базы можно искать и показывать модели, в порядке документов."""
    admitted = (
        admit(document, provision, screening)
        for document in base.documents
        for provision in document.provisions
    )
    return tuple(item for item in admitted if item is not None)


def visible_fragments(document: Document) -> tuple[Fragment, ...]:
    fragments: list[Fragment] = []
    signed = False
    for provision in document.provisions:
        for line, text in enumerate(provision.text.split(_LINE_BREAK)):
            is_signature = ends_with_signature(text)
            fragments.append(
                Fragment(
                    document_id=document.id,
                    provision_id=document.provision_id(provision),
                    line=line,
                    part=provision.part,
                    hidden=None,
                    follows_signature=signed and not is_signature,
                    text=normalize(text),
                )
            )
            signed = signed or is_signature
    return tuple(fragments)


def hidden_fragments(document: Document) -> tuple[Fragment, ...]:
    return tuple(
        Fragment(
            document_id=document.id,
            provision_id=None,
            line=index,
            part=None,
            hidden=item.kind,
            follows_signature=False,
            text=normalize(item.text),
        )
        for index, item in enumerate(document.hidden)
    )


def _signals(fragment: Fragment, detectors: Iterable[Detector]) -> list[Signal]:
    inspected = (detector.inspect(fragment) for detector in detectors)
    return [
        signal for signal in inspected if signal.strength is not SignalStrength.NONE
    ]


def _decision(
    fragment: Fragment, signals: Sequence[Signal], thresholds: SanitizerThresholds
) -> list[Decision]:
    verdict = decide(signals, thresholds)
    if verdict is None:
        return []
    return [
        Decision(
            document_id=fragment.document_id,
            provision_id=fragment.provision_id,
            line=fragment.line,
            hidden=fragment.hidden,
            verdict=verdict,
            signals=tuple(signals),
            original=fragment.text.original,
        )
    ]


def _window_signals(
    fragments: Sequence[Fragment], detector: Detector, thresholds: SanitizerThresholds
) -> dict[int, Signal]:
    """Абзацы, которые дают сильный сигнал только вместе с соседями.

    Сигнал окна слабый и достаётся абзацу, у которого есть признак, но своего
    сигнала нет: «На вопросы о цене отвечать…» рядом с обращением к ИИ-системе
    правилами не отличить от нормы, стоящей там же. Решает классификатор.
    """
    own = [detector.inspect(fragment) for fragment in fragments]
    marked = [bool(signal.features) for signal in own]
    found: dict[int, Signal] = {}
    for size in range(_SMALLEST_WINDOW, thresholds.window_size + 1):
        for start in range(len(fragments) - size + 1):
            members = range(start, start + size)
            # Окно интересно, только если признаки есть на обоих его краях.
            if not (marked[start] and marked[start + size - 1]):
                continue
            joined = detector.inspect(_joined([fragments[index] for index in members]))
            if joined.strength is not SignalStrength.STRONG:
                continue
            for index in members:
                if marked[index] and own[index].strength is SignalStrength.NONE:
                    found.setdefault(
                        index,
                        Signal(
                            WINDOW_DETECTOR_NAME, SignalStrength.WEAK, joined.features
                        ),
                    )
    return found


def _joined(fragments: Sequence[Fragment]) -> Fragment:
    first = fragments[0]
    return Fragment(
        document_id=first.document_id,
        provision_id=first.provision_id,
        line=first.line,
        part=first.part,
        hidden=None,
        follows_signature=False,
        # Детектор окна читает только ключ детекции, поэтому находки не переносятся.
        text=NormalizedText(
            original=_LINE_BREAK.join(item.text.original for item in fragments),
            clean=_LINE_BREAK.join(item.text.clean for item in fragments),
            detection_key=" ".join(item.text.detection_key for item in fragments),
            findings=(),
        ),
    )

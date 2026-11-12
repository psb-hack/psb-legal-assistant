from datetime import date

import pytest

from lex_dates import PartialDate
from lex_documents import (
    AmendmentNote,
    LawAddress,
    LawArticleHeader,
    LawUnit,
    LawUnitKind,
)
from lex_law_validity import (
    LawTextInForce,
    LawTextNewer,
    LawTextUncertain,
    Uncertainty,
    law_text_as_retrieved,
    law_text_on,
)
from synthetic_base import LAW_ARTICLE
from synthetic_links import document

RETRIEVED = date(2031, 6, 5)
CLAUSE_1 = LawUnit(LawUnitKind.CLAUSE, "1")
CLAUSE_2 = LawUnit(LawUnitKind.CLAUSE, "2")
SUBCLAUSE_5 = LawUnit(LawUnitKind.SUBCLAUSE, "5")
RESTATED_IN_2025 = (
    "Федеральным законом от 3 марта 2025 г. N 7-ФЗ статья 300 настоящего Кодекса "
    "изложена в новой редакции, вступающей в силу с 1 июля 2025 г."
)


def _note(effective_from: date, *units: LawUnit) -> AmendmentNote:
    target = LawAddress(units, None) if units else None
    return AmendmentNote(target, effective_from, "Федеральный закон № 5-ФЗ", "пометка")


def _header(
    *amendments: AmendmentNote,
    retrieved: date | None = RETRIEVED,
    comments: tuple[str, ...] = (),
) -> LawArticleHeader:
    return LawArticleHeader(
        act="Трудовой кодекс Российской Федерации от 30.12.2001 N 197-ФЗ (ТК РФ)",
        act_short="ТК РФ",
        article="300",
        title="Учебный отпуск",
        source_url="https://example.org/labour/300",
        retrieved=retrieved,
        amendments=amendments,
        comments=comments,
        repealed=(),
    )


def _day(year: int, month: int, day: int) -> PartialDate:
    return PartialDate.of_day(date(year, month, day))


ARTICLE_CHANGED = _note(date(2031, 3, 1))
ARTICLE_WILL_CHANGE = _note(date(2032, 1, 1))


def test_text_changed_before_the_event_is_in_force_on_its_date() -> None:
    header = _header(ARTICLE_CHANGED)
    in_force = LawTextInForce(ARTICLE_CHANGED, date_stated=True)
    assert law_text_on(header, None, _day(2031, 3, 1)) == in_force
    assert law_text_on(header, None, PartialDate.of_month(2031, 4)) == in_force


def test_text_changed_after_the_event_is_newer_than_the_event() -> None:
    header = _header(_note(date(2030, 1, 1)), ARTICLE_CHANGED)
    newer = LawTextNewer(ARTICLE_CHANGED)
    assert law_text_on(header, None, _day(2031, 2, 28)) == newer
    assert law_text_on(header, None, PartialDate.of_year(2030)) == newer


def test_incomplete_date_across_the_change_is_uncertain() -> None:
    across = law_text_on(_header(ARTICLE_CHANGED), None, PartialDate.of_year(2031))
    assert across == LawTextUncertain(Uncertainty.SPANS_AMENDMENT, ARTICLE_CHANGED)


def test_text_without_amendment_notes_is_in_force() -> None:
    assert law_text_on(_header(), None, _day(2030, 5, 5)) == LawTextInForce(None, True)


def test_change_of_another_unit_does_not_concern_the_norm() -> None:
    changed = _note(date(2031, 3, 1), CLAUSE_2)
    address = LawAddress((CLAUSE_1,), None)
    outcome = law_text_on(_header(changed), address, _day(2030, 5, 5))
    assert outcome == LawTextInForce(None, True)


def test_change_of_the_parent_unit_concerns_its_subclause() -> None:
    changed = _note(date(2031, 3, 1), CLAUSE_2)
    address = LawAddress((CLAUSE_2, SUBCLAUSE_5), None)
    assert law_text_on(_header(changed), address, _day(2030, 5, 5)) == LawTextNewer(
        changed
    )


def test_change_of_a_subclause_does_not_concern_its_parent_or_sibling_clause() -> None:
    changed = _note(date(2031, 3, 1), CLAUSE_2, SUBCLAUSE_5)
    header = _header(changed)
    for units in [(CLAUSE_2,), (CLAUSE_1, SUBCLAUSE_5)]:
        outcome = law_text_on(header, LawAddress(units, None), _day(2030, 5, 5))
        assert outcome == LawTextInForce(None, True)


def test_added_unit_named_without_its_path_is_matched_by_the_unit_alone() -> None:
    added = _note(date(2031, 3, 1), SUBCLAUSE_5)
    address = LawAddress((CLAUSE_1, SUBCLAUSE_5), None)
    assert law_text_on(_header(added), address, _day(2030, 5, 5)) == LawTextNewer(added)


def test_any_change_concerns_the_article_as_a_whole() -> None:
    changed = _note(date(2031, 3, 1), CLAUSE_2)
    assert law_text_on(_header(changed), None, _day(2030, 5, 5)) == LawTextNewer(
        changed
    )


@pytest.mark.parametrize("period", [_day(2031, 4, 1), _day(2032, 2, 1)])
def test_amendment_not_in_force_at_retrieval_leaves_the_text_uncertain(
    period: PartialDate,
) -> None:
    header = _header(ARTICLE_WILL_CHANGE, ARTICLE_CHANGED)
    assert law_text_on(header, None, period) == LawTextUncertain(
        Uncertainty.PENDING_AMENDMENT, ARTICLE_WILL_CHANGE
    )


def test_event_before_the_last_change_in_force_is_certain_despite_a_pending_one() -> (
    None
):
    header = _header(ARTICLE_WILL_CHANGE, ARTICLE_CHANGED)
    assert law_text_on(header, None, _day(2031, 2, 1)) == LawTextNewer(ARTICLE_CHANGED)


def test_event_after_retrieval_is_uncertain() -> None:
    header = _header(ARTICLE_CHANGED)
    after = LawTextUncertain(Uncertainty.AFTER_RETRIEVAL, None)
    assert law_text_on(header, None, _day(2031, 6, 6)) == after
    assert law_text_on(header, None, PartialDate.of_month(2031, 6)) == after
    assert law_text_on(header, None, _day(2031, 6, 5)) == LawTextInForce(
        ARTICLE_CHANGED, True
    )


def test_article_without_retrieval_date_is_uncertain() -> None:
    header = _header(ARTICLE_CHANGED, retrieved=None)
    unknown = LawTextUncertain(Uncertainty.NO_RETRIEVAL_DATE, None)
    assert law_text_on(header, None, _day(2031, 4, 1)) == unknown
    assert law_text_as_retrieved(header, None) == unknown


def test_question_without_a_date_gets_the_text_as_retrieved() -> None:
    outcome = law_text_as_retrieved(_header(ARTICLE_CHANGED), None)
    assert outcome == LawTextInForce(ARTICLE_CHANGED, date_stated=False)


def test_question_without_a_date_is_told_about_a_pending_amendment() -> None:
    header = _header(ARTICLE_WILL_CHANGE, ARTICLE_CHANGED)
    assert law_text_as_retrieved(header, None) == LawTextUncertain(
        Uncertainty.PENDING_AMENDMENT, ARTICLE_WILL_CHANGE
    )


def test_earlier_change_recorded_as_a_comment_makes_the_text_newer() -> None:
    header = _header(comments=(RESTATED_IN_2025,))
    restated = AmendmentNote(None, date(2025, 7, 1), None, RESTATED_IN_2025)
    assert law_text_on(header, None, _day(2025, 6, 30)) == LawTextNewer(restated)
    assert law_text_on(header, None, _day(2025, 7, 1)) == LawTextInForce(restated, True)


@pytest.mark.parametrize(
    "comment",
    [
        "Федеральным законом от 3 марта 2025 г. N 7-ФЗ в статью 300 внесены изменения, "
        "вступающие в силу по истечении одного месяца со дня опубликования",
        "О конституционно-правовом смысле статьи 300 см. постановление от 19 мая 2020 г.",
        "Положения статьи 300 применяются к отпускам, начавшимся с 1 января 2031 г.",
    ],
)
def test_comment_without_a_dated_change_is_not_a_change(comment: str) -> None:
    outcome = law_text_on(_header(comments=(comment,)), None, _day(2020, 5, 5))
    assert outcome == LawTextInForce(None, True)


def test_notes_of_a_parsed_article_decide_the_outcome() -> None:
    notes = (
        "Статья 300 изменена с 1 января 2032 г. - Федеральный закон от 1 мая 2031 г. "
        "№ 9-ФЗ\n"
        "Статья 300 изменена с 1 марта 2031 г. - Федеральный закон от 1 февраля 2031 г. "
        "№ 5-ФЗ\n"
    )
    text = LAW_ARTICLE.replace("Статья 300. Учебный", f"{notes}Статья 300. Учебный")
    parsed = document(text)
    header = parsed.header
    assert isinstance(header, LawArticleHeader)
    address = parsed.provisions[0].number
    assert isinstance(address, LawAddress)
    newer = law_text_on(header, address, _day(2031, 2, 1))
    assert isinstance(newer, LawTextNewer)
    assert newer.amended.effective_from == date(2031, 3, 1)
    pending = law_text_on(header, address, _day(2031, 5, 1))
    assert isinstance(pending, LawTextUncertain)
    assert pending.reason is Uncertainty.PENDING_AMENDMENT
    assert pending.note is not None
    assert pending.note.effective_from == date(2032, 1, 1)

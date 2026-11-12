from datetime import date

import pytest

from lex_dates import PartialDate
from lex_document_graph import Edition, EditionFamily, FamilyKey, build_graph
from lex_documents import DocumentId
from lex_edition_choice import (
    EditionInForce,
    EditionsAcrossBoundary,
    MissingEdition,
    NoEditionOnDate,
    RepealDateUnknown,
    current_edition,
    edition_on,
)
from synthetic_links import NEW_RULES, OLD_RULES, knowledge_base

BASE = knowledge_base(OLD_RULES, NEW_RULES)
(FAMILY,) = build_graph(BASE).families
# Положение от 02.02.2029 действует до 15.07.2031, с этого дня — редакция от 01.07.2031.
OLD, NEW = FAMILY.editions
CHANGE = date(2031, 7, 15)

ONLY_OLD = knowledge_base(OLD_RULES)
(ONLY_OLD_FAMILY,) = build_graph(ONLY_OLD).families
REPEALED_SOMETIME = knowledge_base(OLD_RULES.replace(" с 15.07.2031", ""))
(REPEALED_SOMETIME_FAMILY,) = build_graph(REPEALED_SOMETIME).families


def _day(year: int, month: int, day: int) -> PartialDate:
    return PartialDate.of_day(date(year, month, day))


@pytest.mark.parametrize(
    ("period", "edition"),
    [
        (_day(2029, 2, 2), OLD),
        (_day(2031, 7, 14), OLD),
        (_day(2031, 7, 15), NEW),
        (PartialDate.of_month(2031, 6), OLD),
        (PartialDate.of_month(2031, 8), NEW),
        (PartialDate.of_year(2030), OLD),
        (PartialDate.of_year(2032), NEW),
    ],
)
def test_period_inside_one_edition_gives_that_edition(
    period: PartialDate, edition: Edition
) -> None:
    assert edition_on(FAMILY, period, BASE) == EditionInForce(edition, True)


@pytest.mark.parametrize(
    "period", [PartialDate.of_month(2031, 7), PartialDate.of_year(2031)]
)
def test_incomplete_date_across_the_change_gives_both_editions_and_the_boundary(
    period: PartialDate,
) -> None:
    assert edition_on(FAMILY, period, BASE) == EditionsAcrossBoundary(
        editions=(OLD, NEW), boundaries=(CHANGE,)
    )


def test_date_before_the_first_edition_has_no_edition() -> None:
    choice = edition_on(FAMILY, _day(2029, 2, 1), BASE)
    assert choice == NoEditionOnDate(MissingEdition.BEFORE_FIRST_EDITION)
    assert edition_on(FAMILY, PartialDate.of_year(2028), BASE) == choice


def test_period_that_starts_before_the_first_edition_is_not_a_single_edition() -> None:
    choice = edition_on(FAMILY, PartialDate.of_year(2029), BASE)
    assert choice == EditionsAcrossBoundary((OLD,), (date(2029, 2, 2),))


def test_date_after_the_repeal_of_the_last_edition_has_no_edition() -> None:
    choice = edition_on(ONLY_OLD_FAMILY, _day(2031, 7, 15), ONLY_OLD)
    assert choice == NoEditionOnDate(MissingEdition.NOT_IN_FORCE)


def test_period_across_the_repeal_of_the_last_edition_names_the_boundary() -> None:
    choice = edition_on(ONLY_OLD_FAMILY, PartialDate.of_month(2031, 7), ONLY_OLD)
    assert choice == EditionsAcrossBoundary(ONLY_OLD_FAMILY.editions, (CHANGE,))


def test_gap_between_editions_has_no_edition() -> None:
    first = Edition(
        DocumentId("d01"), date(2029, 2, 2), date(2029, 2, 2), date(2030, 1, 1)
    )
    second = Edition(DocumentId("d02"), date(2031, 7, 1), date(2031, 7, 1), None)
    family = EditionFamily(FamilyKey("regulation:правила"), "Правила", (first, second))
    choice = edition_on(family, PartialDate.of_month(2030, 6), BASE)
    assert choice == NoEditionOnDate(MissingEdition.NOT_IN_FORCE)
    assert edition_on(family, PartialDate.of_year(2030), BASE) == NoEditionOnDate(
        MissingEdition.NOT_IN_FORCE
    )
    assert edition_on(family, PartialDate.of_year(2031), BASE) == (
        EditionsAcrossBoundary((second,), (date(2031, 7, 1),))
    )


def test_edition_repealed_without_a_date_is_not_claimed_to_be_in_force() -> None:
    (edition,) = REPEALED_SOMETIME_FAMILY.editions
    assert edition.effective_to is None
    choice = edition_on(REPEALED_SOMETIME_FAMILY, _day(2030, 5, 5), REPEALED_SOMETIME)
    assert choice == RepealDateUnknown(edition)


def test_question_without_a_date_gets_the_current_edition_marked_as_such() -> None:
    assert current_edition(FAMILY, date(2031, 9, 1), BASE) == EditionInForce(NEW, False)
    assert current_edition(FAMILY, date(2031, 7, 14), BASE) == EditionInForce(
        OLD, False
    )


def test_question_without_a_date_has_no_current_edition_after_the_repeal() -> None:
    not_in_force = NoEditionOnDate(MissingEdition.NOT_IN_FORCE)
    assert current_edition(ONLY_OLD_FAMILY, date(2031, 9, 1), ONLY_OLD) == not_in_force
    assert (
        current_edition(REPEALED_SOMETIME_FAMILY, date(2031, 9, 1), REPEALED_SOMETIME)
        == not_in_force
    )


def test_question_asked_before_the_first_edition_has_no_current_edition() -> None:
    choice = current_edition(FAMILY, date(2029, 1, 1), BASE)
    assert choice == NoEditionOnDate(MissingEdition.BEFORE_FIRST_EDITION)

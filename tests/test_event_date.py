from datetime import date

import pytest

from lex_dates import PartialDate
from lex_event_date import (
    DateRole,
    EventDate,
    QuestionDate,
    UnclearEventDate,
    read_event_date,
)

TODAY = date(2031, 3, 14)
AUGUST_2030 = PartialDate.of_month(2030, 8)


def _period(question: str) -> PartialDate:
    event = read_event_date(question, TODAY)
    assert isinstance(event, EventDate)
    return event.period


@pytest.mark.parametrize(
    ("question", "period"),
    [
        ("Подряд заключили в августе 2030 года. Кто согласует?", AUGUST_2030),
        ("Подряд заключили в августе 2030. Кто согласует?", AUGUST_2030),
        ("Закупка 05.08.2030 — кто согласует?", PartialDate.of_day(date(2030, 8, 5))),
        ("Закупка 5 августа 2030 г.", PartialDate.of_day(date(2030, 8, 5))),
        ("Какой порог действовал в 2029 году?", PartialDate.of_year(2029)),
        ("Порог закупок в мае 2031", PartialDate.of_month(2031, 5)),
    ],
)
def test_date_named_with_its_year_is_the_event_date(
    question: str, period: PartialDate
) -> None:
    assert _period(question) == period


@pytest.mark.parametrize(
    ("question", "period"),
    [
        ("Кто согласует закупку сегодня?", PartialDate.of_day(date(2031, 3, 14))),
        ("Закупку провели вчера.", PartialDate.of_day(date(2031, 3, 13))),
        ("Закупку провели позавчера.", PartialDate.of_day(date(2031, 3, 12))),
        ("Закупка была в этом месяце.", PartialDate.of_month(2031, 3)),
        ("Закупка была в прошлом месяце.", PartialDate.of_month(2031, 2)),
        ("Какой порог в текущем году?", PartialDate.of_year(2031)),
        ("Закупку провели в прошлом году.", PartialDate.of_year(2030)),
        ("Закупку провели в позапрошлом году.", PartialDate.of_year(2029)),
        ("Закупку проведём в следующем году.", PartialDate.of_year(2032)),
        ("Подряд заключили в августе прошлого года.", AUGUST_2030),
        (
            "Подряд от нас ушёл 5 августа прошлого года.",
            PartialDate.of_day(date(2030, 8, 5)),
        ),
    ],
)
def test_relative_date_is_counted_from_the_day_of_the_question(
    question: str, period: PartialDate
) -> None:
    assert _period(question) == period


def test_previous_month_in_january_is_december_of_the_previous_year() -> None:
    event = read_event_date("Закупка была в прошлом месяце.", date(2031, 1, 20))
    assert event == EventDate(PartialDate.of_month(2030, 12), "прошлом месяце")


def test_event_date_keeps_the_wording_of_the_question() -> None:
    event = read_event_date("Подряд заключили в августе 2030 года.", TODAY)
    assert event == EventDate(AUGUST_2030, "августе 2030")


def test_question_without_dates_has_no_event_date() -> None:
    question = "Кто согласует договор № С-21/2030 на 2030 рублей по п. 2.1?"
    assert read_event_date(question, TODAY) is None


def test_the_same_date_named_twice_is_one_event_date() -> None:
    question = "Подряд заключили в августе 2030 года. Кто согласовывал в августе 2030?"
    assert _period(question) == AUGUST_2030


def test_several_different_dates_are_returned_without_a_choice() -> None:
    question = "Подряд заключили 05.08.2030, а оплатили в прошлом месяце."
    assert read_event_date(question, TODAY) == UnclearEventDate(
        candidates=(
            QuestionDate(
                PartialDate.of_day(date(2030, 8, 5)), "05.08.2030", DateRole.STATED
            ),
            QuestionDate(
                PartialDate.of_month(2031, 2), "прошлом месяце", DateRole.STATED
            ),
        ),
        incomplete=(),
    )


def test_date_of_a_document_alone_is_not_taken_for_the_event_date() -> None:
    question = "Какой порог по Положению о закупках в редакции от 02.02.2029?"
    assert read_event_date(question, TODAY) == UnclearEventDate(
        candidates=(
            QuestionDate(
                PartialDate.of_day(date(2029, 2, 2)),
                "02.02.2029",
                DateRole.DOCUMENT_DATE,
            ),
        ),
        incomplete=(),
    )


def test_document_date_next_to_a_stated_date_makes_the_event_date_unclear() -> None:
    question = "Договор от 5 мая 2030 г. оплатили в августе 2030 года."
    event = read_event_date(question, TODAY)
    assert isinstance(event, UnclearEventDate)
    assert [found.role for found in event.candidates] == [
        DateRole.DOCUMENT_DATE,
        DateRole.STATED,
    ]


@pytest.mark.parametrize(
    ("question", "wording"),
    [
        ("Подряд заключили в августе. Кто согласует?", "августе"),
        ("Закупку провели 5 мая.", "5 мая"),
        ("Закупку провели два года назад.", "два года назад"),
        ("Подряд заключили полгода назад.", "полгода назад"),
        ("Закупку провели 31 февраля прошлого года.", "31 февраля прошлого года"),
    ],
)
def test_time_that_cannot_be_computed_is_reported_as_incomplete(
    question: str, wording: str
) -> None:
    assert read_event_date(question, TODAY) == UnclearEventDate((), (wording,))


def test_impossible_calendar_date_is_not_read_as_its_year() -> None:
    assert read_event_date("Закупку провели 31.02.2030 года.", TODAY) is None

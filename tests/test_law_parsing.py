from datetime import date

import pytest

import lex_law_parsing
from lex_documents import (
    DocumentPart,
    Heading,
    LawAddress,
    LawUnit,
    LawUnitKind,
    ParagraphRange,
)
from lex_errors import KnowledgeBaseError
from lex_law_lines import LawLineKind, classify_lines, split_marker
from lex_law_parsing import ParsedArticle, parse_article
from lex_paragraphs import split_paragraphs

TAX_CODE = """\
Источник: Статья 501. НК РФ Ставки сбора. Налоговый кодекс Российской Федерации (НК РФ) (с изменениями и дополнениями)

URL: https://example.org/tax/501

Дата выгрузки: 05.06.2031

Федеральным законом от 1 марта 2030 г. N 11-ФЗ в статью 501 настоящего Кодекса внесены изменения, вступающие в силу с 1 января 2031 г.

Статья 501. Ставки сбора

Пункт 1 изменен с 1 июля 2031 г. - Федеральный закон от 2 марта 2031 г. № 77-ФЗ

1. Сбор взимается по ставке 0 процентов при реализации:

1) товаров, вывезенных за пределы территории;

Пункт 1 дополнен подпунктом 1.1 с 1 августа 2031 г. - Федеральный закон от 3 марта 2031 г. N 78-ФЗ

Подпункт 1.1 применяется к услугам, оказанным начиная с 1 августа 2031 г.

1.1) услуг по перевозке товаров.

Положения настоящего подпункта применяются при наличии договора.

2) утратил силу с 1 января 2029 г.;

2. Ставка 10 процентов применяется при реализации:

книг;

абзац третий утратил силу;

журналов.

О применении пункта 2 см. письмо ведомства

3. Утратил силу с 1 января 2030 г.

Статья 501 дополнена пунктом 3.1-1 с 1 сентября 2031 г. - Федеральный закон от 4 марта 2031 г. № 79-ФЗ

3.1-1. В остальных случаях применяется ставка 20 процентов.

Статья 501 дополнена пунктом 4 с 1 октября 2031 г. - Федеральный закон от 5 марта 2031 г. № 80-ФЗ
"""

OFFENCES_CODE = """\
Источник: Статья 77.7. КоАП РФ Нарушение порядка учета. Кодекс Российской Федерации об административных правонарушениях от 30.12.2001 N 195-ФЗ (КоАП РФ) (с изменениями и дополнениями)
URL: https://example.org/offences/77-7
Дата выгрузки: 05.06.2031
Статья 77.7. Нарушение порядка учета
Cрок давности привлечения к ответственности по ст. 77.7
Часть 1 изменена с 1 мая 2031 г. - Федеральный закон от 2 марта 2031 г. N 77-ФЗ
1. Нарушение порядка учета, если оно не содержит признаков преступления, -
влечет наложение административного штрафа в размере одной тысячи рублей.
Статья 77.7 дополнена частью 1.1 с 1 мая 2031 г. - Федеральный закон от 2 марта 2031 г. N 77-ФЗ
Обратите внимание, что часть 1.1 применяется к организациям
1.1. Повторное нарушение порядка учета -
влечет наложение административного штрафа в размере двух тысяч рублей.
Примечания изменены с 1 мая 2031 г. - Федеральный закон от 2 марта 2031 г. N 77-ФЗ
Примечания:
1. Под учетом понимается ведение журнала.
2. Штраф не назначается за первое нарушение.
"""

LABOUR_CODE = """\
Источник: Отпуск. Статья 300. ТК РФ Учебный отпуск. Трудовой кодекс Российской Федерации от 30.12.2001 N 197-ФЗ (ТК РФ) (с изменениями и дополнениями)
URL: https://example.org/labour/300
Дата выгрузки: 05.06.2031
Статья 300 изменена с 1 марта 2032 г. - Федеральный закон от 5 мая 2031 г. № 80-ФЗ
Статья 300 изменена с 1 сентября 2031 г. - Федеральный закон от 6 мая 2031 г. № 81-ФЗ
Статья 300. Учебный отпуск
Учебный отпуск предоставляется работнику по его заявлению.
Часть вторая утратила силу.
Отпуск не предоставляется:
лицам, принятым на срок до двух месяцев;
О конституционно-правовом смысле абзаца третьего части третьей статьи 300 см. постановление от 1 июня 2030 г. N 5-П
лицам, не достигшим возраста восемнадцати лет.
Отпуск продлевается в следующих случаях:
1) при болезни работника;
Пункт 2 изменен с 1 июля 2031 г. - Федеральный закон от 7 мая 2031 г. N 82-ФЗ
2) при исполнении государственных обязанностей.
В других случаях отпуск не продлевается.
"""

FEDERAL_LAW_WITHOUT_TITLE = """\
Источник: Статья 2. Федеральный закон от 01.02.2030 N 500-ФЗ "О базовой ставке" (с изменениями и дополнениями)
URL: https://example.org/law/500
Дата выгрузки: 05.06.2031
Статья 2. Установить базовую ставку с 1 января 2031 года в сумме 100 рублей.
Базовая ставка пересматривается ежегодно.
"""

FEDERAL_LAW = """\
Источник: Статья 3. Согласие на учет. Федеральный закон от 01.02.2030 N 501-ФЗ "Об учете" (с изменениями и дополнениями)
URL: https://example.org/law/501
Дата выгрузки: 05.06.2031
Статья 3. Согласие на учет
1. Согласие дается в письменной форме и включает:
1) фамилию;
2) подпись.
2. Согласие может быть отозвано.
"""


def _labels(article: ParsedArticle) -> list[str]:
    return [provision.label for provision in article.provisions]


def _texts(article: ParsedArticle) -> dict[str, str]:
    return {provision.label: provision.text for provision in article.provisions}


def _targets(article: ParsedArticle) -> list[str | None]:
    return [
        None if note.target is None else note.target.label
        for note in article.header.amendments
    ]


def _assert_nothing_is_lost(text: str, article: ParsedArticle) -> None:
    """Шапка, пометки, заглушки и положения вместе дают исходный текст."""
    paragraphs = split_paragraphs(text)
    lines = classify_lines(paragraphs)
    assert "".join(line.text for line in lines).replace(" ", "") == "".join(
        paragraphs
    ).replace(" ", "")

    norms = []
    for line in lines:
        if line.kind is LawLineKind.NORM:
            marker = split_marker(line.text)
            norms.append(line.text if marker is None else marker.text)
    in_provisions = [
        line for provision in article.provisions for line in provision.text.split("\n")
    ]
    assert in_provisions == norms

    header = article.header
    outside_norms = [
        *(note.text for note in header.amendments),
        *header.comments,
        *(unit.text for unit in header.repealed),
    ]
    assert sorted(outside_norms) == sorted(
        line.text
        for line in lines
        if line.kind in (LawLineKind.COMMENT, LawLineKind.REPEALED)
    )


@pytest.mark.parametrize(
    "text",
    [TAX_CODE, OFFENCES_CODE, LABOUR_CODE, FEDERAL_LAW_WITHOUT_TITLE, FEDERAL_LAW],
)
def test_header_notes_and_provisions_together_give_the_source_text(text: str) -> None:
    _assert_nothing_is_lost(text, parse_article(text))


def test_header_of_the_article_comes_from_the_source_lines() -> None:
    header = parse_article(TAX_CODE).header
    assert header.act == "Налоговый кодекс Российской Федерации (НК РФ)"
    assert header.act_short == "НК РФ"
    assert header.article == "501"
    assert header.title == "Ставки сбора"
    assert header.source_url == "https://example.org/tax/501"
    assert header.retrieved == date(2031, 6, 5)
    assert header.citation_title == "НК РФ, ст. 501"


def test_tax_code_is_addressed_by_clauses_and_subclauses() -> None:
    article = parse_article(TAX_CODE)
    assert _labels(article) == [
        "п. 1",
        "п. 1, подп. 1",
        "п. 1, подп. 1.1",
        "п. 2",
        "п. 3.1-1",
    ]


def test_unit_marker_is_not_part_of_the_provision_text() -> None:
    texts = _texts(parse_article(TAX_CODE))
    assert texts["п. 1"] == "Сбор взимается по ставке 0 процентов при реализации:"
    assert texts["п. 1, подп. 1"] == "товаров, вывезенных за пределы территории;"


def test_paragraphs_without_number_belong_to_the_unit_above() -> None:
    texts = _texts(parse_article(TAX_CODE))
    assert texts["п. 1, подп. 1.1"] == (
        "услуг по перевозке товаров.\n"
        "Положения настоящего подпункта применяются при наличии договора."
    )
    assert texts["п. 2"] == (
        "Ставка 10 процентов применяется при реализации:\nкниг;\nжурналов."
    )


def test_subclause_keeps_the_introduction_of_its_clause_as_heading() -> None:
    subclause = parse_article(TAX_CODE).provisions[1]
    assert subclause.headings == (
        Heading(
            LawUnit(LawUnitKind.CLAUSE, "1"),
            "Сбор взимается по ставке 0 процентов при реализации:",
        ),
    )
    assert subclause.part is DocumentPart.BODY


def test_repealed_units_keep_their_addresses_and_stay_out_of_provisions() -> None:
    article = parse_article(TAX_CODE)
    assert [(unit.address.label, unit.text) for unit in article.header.repealed] == [
        ("п. 1, подп. 2", "2) утратил силу с 1 января 2029 г.;"),
        ("п. 2, абз. 3", "абзац третий утратил силу;"),
        ("п. 3", "3. Утратил силу с 1 января 2030 г."),
    ]
    assert all("утратил" not in text.lower() for text in _texts(article).values())


def test_amendment_gets_the_full_address_of_the_unit_it_precedes() -> None:
    article = parse_article(TAX_CODE)
    assert _targets(article)[:3] == ["п. 1", "п. 1, подп. 1.1", "п. 3.1-1"]
    first = article.header.amendments[0]
    assert first.effective_from == date(2031, 7, 1)
    assert first.source_act == "Федеральный закон от 2 марта 2031 г. № 77-ФЗ"


def test_amendment_of_a_unit_missing_from_the_text_keeps_the_name_it_was_given() -> (
    None
):
    article = parse_article(TAX_CODE)
    assert _targets(article)[3] == "п. 4"
    assert "п. 4" not in _labels(article)


def test_comments_without_address_are_kept_apart_from_norms() -> None:
    article = parse_article(TAX_CODE)
    assert article.header.comments == (
        "Федеральным законом от 1 марта 2030 г. N 11-ФЗ в статью 501 настоящего "
        "Кодекса внесены изменения, вступающие в силу с 1 января 2031 г.",
        "Подпункт 1.1 применяется к услугам, оказанным начиная с 1 августа 2031 г.",
        "О применении пункта 2 см. письмо ведомства",
    )


def test_offences_code_is_addressed_by_parts_and_notes() -> None:
    article = parse_article(OFFENCES_CODE)
    assert _labels(article) == ["ч. 1", "ч. 1.1", "прим.", "прим. 1", "прим. 2"]
    assert _targets(article) == ["ч. 1", "ч. 1.1", "прим."]
    assert len(article.header.comments) == 2


def test_sanction_stays_with_its_disposition() -> None:
    texts = _texts(parse_article(OFFENCES_CODE))
    assert texts["ч. 1"] == (
        "Нарушение порядка учета, если оно не содержит признаков преступления, -\n"
        "влечет наложение административного штрафа в размере одной тысячи рублей."
    )


def test_sanction_is_not_split_from_disposition_even_in_a_long_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lex_law_parsing, "MAX_PROVISION_LENGTH", 10)
    assert _labels(parse_article(OFFENCES_CODE))[:2] == ["ч. 1", "ч. 1.1"]


def test_notes_are_numbered_anew_under_their_own_heading() -> None:
    note = parse_article(OFFENCES_CODE).provisions[3]
    assert note.text == "Под учетом понимается ведение журнала."
    assert note.headings == (Heading(LawUnit(LawUnitKind.NOTE, None), "Примечания:"),)


def test_labour_code_parts_are_counted_by_paragraphs_starting_a_sentence() -> None:
    article = parse_article(LABOUR_CODE)
    assert _labels(article) == [
        "ч. 1",
        "ч. 3",
        "ч. 4",
        "ч. 4, п. 1",
        "ч. 4, п. 2",
        "ч. 5",
    ]
    assert _texts(article)["ч. 3"] == (
        "Отпуск не предоставляется:\n"
        "лицам, принятым на срок до двух месяцев;\n"
        "лицам, не достигшим возраста восемнадцати лет."
    )


def test_repealed_part_keeps_its_number() -> None:
    header = parse_article(LABOUR_CODE).header
    assert [(unit.address.label, unit.text) for unit in header.repealed] == [
        ("ч. 2", "Часть вторая утратила силу.")
    ]


def test_future_amendment_of_the_whole_article_is_kept_with_its_date() -> None:
    header = parse_article(LABOUR_CODE).header
    whole_article = [note for note in header.amendments if note.target is None]
    assert [note.effective_from for note in whole_article] == [
        date(2032, 3, 1),
        date(2031, 9, 1),
    ]


def test_numbered_item_of_an_unnumbered_part_is_a_clause() -> None:
    article = parse_article(LABOUR_CODE)
    assert _targets(article)[2] == "ч. 4, п. 2"
    assert article.provisions[4].headings == (
        Heading(
            LawUnit(LawUnitKind.PART, "4"), "Отпуск продлевается в следующих случаях:"
        ),
    )


def test_norm_after_the_article_number_is_the_first_part() -> None:
    article = parse_article(FEDERAL_LAW_WITHOUT_TITLE)
    assert article.header.title == ""
    assert article.header.act_short == "500-ФЗ"
    assert _texts(article) == {
        "ч. 1": "Установить базовую ставку с 1 января 2031 года в сумме 100 рублей.",
        "ч. 2": "Базовая ставка пересматривается ежегодно.",
    }


def test_federal_law_is_addressed_by_parts_and_clauses() -> None:
    assert _labels(parse_article(FEDERAL_LAW)) == [
        "ч. 1",
        "ч. 1, п. 1",
        "ч. 1, п. 2",
        "ч. 2",
    ]


def test_long_unit_is_divided_by_paragraphs_with_the_same_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lex_law_parsing, "MAX_PROVISION_LENGTH", 70)
    article = parse_article(LABOUR_CODE)
    divided = [item for item in article.provisions if item.label.startswith("ч. 3")]
    assert [item.number for item in divided] == [
        LawAddress((LawUnit(LawUnitKind.PART, "3"),), ParagraphRange(1, 2)),
        LawAddress((LawUnit(LawUnitKind.PART, "3"),), ParagraphRange(3, 3)),
    ]
    assert [item.label for item in divided] == ["ч. 3, абз. 1–2", "ч. 3, абз. 3"]
    _assert_nothing_is_lost(LABOUR_CODE, article)


def test_continuation_of_a_divided_list_keeps_its_introduction_as_heading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lex_law_parsing, "MAX_PROVISION_LENGTH", 70)
    first, second = parse_article(LABOUR_CODE).provisions[1:3]
    assert first.headings == ()
    assert second.headings == (
        Heading(LawUnit(LawUnitKind.PART, "3"), "Отпуск не предоставляется:"),
    )
    assert second.text == "лицам, не достигшим возраста восемнадцати лет."


def test_repealed_paragraph_keeps_its_number_in_a_divided_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lex_law_parsing, "MAX_PROVISION_LENGTH", 55)
    labels = _labels(parse_article(TAX_CODE))
    assert [label for label in labels if label.startswith("п. 2")] == [
        "п. 2, абз. 1–2",
        "п. 2, абз. 4",
    ]


def test_text_before_the_first_unit_has_no_address() -> None:
    text = FEDERAL_LAW.replace(
        "1. Согласие дается",
        "Настоящая статья применяется к учету.\n1. Согласие дается",
    )
    first = parse_article(text).provisions[0]
    assert first.number is None
    assert first.part is DocumentPart.PREAMBLE


def test_text_without_article_title_is_not_an_article() -> None:
    with pytest.raises(KnowledgeBaseError, match="Статья"):
        parse_article("Источник: архив\nПриказ о приеме на работу.\n")

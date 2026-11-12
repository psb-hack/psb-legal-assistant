import pytest

from lex_law_lines import (
    ArticleTitle,
    LawLine,
    LawLineKind,
    MarkerClosing,
    UnitMarker,
    classify_lines,
    is_law_article,
    is_repealed_part,
    read_article_title,
    repealed_paragraph_count,
    split_marker,
)

ARTICLE = [
    "Источник: Статья 501. НК РФ Ставки сбора. Налоговый кодекс (НК РФ)",
    "URL: https://example.org/tax/501",
    "Дата выгрузки: 05.06.2031",
    "Статья 501 изменена с 1 июля 2031 г. - Федеральный закон от 2 марта 2031 г. № 77-ФЗ",
    "Статья 501. Ставки сбора",
    "1. Сбор взимается по ставке 10 процентов с 1 января 2031 года.",
    "О применении пункта 1 см. письмо ведомства",
    "2. Утратил силу с 1 января 2030 г.",
]


def _kinds(paragraphs: list[str]) -> list[LawLineKind]:
    return [line.kind for line in classify_lines(paragraphs)]


def test_every_line_of_an_article_gets_its_kind() -> None:
    assert _kinds(ARTICLE) == [
        LawLineKind.SOURCE,
        LawLineKind.SOURCE,
        LawLineKind.SOURCE,
        LawLineKind.COMMENT,
        LawLineKind.ARTICLE_TITLE,
        LawLineKind.NORM,
        LawLineKind.COMMENT,
        LawLineKind.REPEALED,
    ]


def test_classification_keeps_text_and_order() -> None:
    assert [line.text for line in classify_lines(ARTICLE)] == ARTICLE


def test_line_with_short_year_is_a_comment_and_full_year_is_a_norm() -> None:
    assert _kinds(["Положения пункта 2 применяются с 1 мая 2031 г. включительно."]) == [
        LawLineKind.COMMENT
    ]
    assert _kinds(["Положения пункта 2 применяются с 1 мая 2031 года."]) == [
        LawLineKind.NORM
    ]


def test_short_year_is_found_after_non_breaking_space() -> None:
    assert _kinds(["Пункт 2 применяется с 1 мая 2031 г. включительно."]) == [
        LawLineKind.COMMENT
    ]


@pytest.mark.parametrize(
    "norm",
    [
        "Работник вправе отказаться.",
        "на время выполнения временных работ;",
        "Отпуск не предоставляется:",
        "Нарушение порядка учета, если оно совершено повторно, -",
        "Нарушение порядка учета —",
        "лицам, не достигшим возраста восемнадцати лет,",
    ],
)
def test_line_ending_with_punctuation_is_a_norm(norm: str) -> None:
    assert _kinds([norm]) == [LawLineKind.NORM]


def test_line_without_final_punctuation_is_a_comment() -> None:
    assert _kinds(["Обратите внимание на порядок расчета срока давности"]) == [
        LawLineKind.COMMENT
    ]


def test_source_fields_are_recognised_only_before_the_article_title() -> None:
    paragraphs = ["Статья 5. Источники права", "Источник: обычай делового оборота."]
    assert _kinds(paragraphs) == [LawLineKind.ARTICLE_TITLE, LawLineKind.NORM]


def test_article_title_with_a_norm_after_the_number_is_split_in_two() -> None:
    lines = classify_lines(["Статья 2. Установить базовую ставку в сумме 100 рублей."])
    assert lines == [
        LawLine(LawLineKind.ARTICLE_TITLE, "Статья 2."),
        LawLine(LawLineKind.NORM, "Установить базовую ставку в сумме 100 рублей."),
    ]


def test_only_the_first_title_line_is_the_article_title() -> None:
    paragraphs = ["Статья 5. Отпуск", "Статья 6. Применяется с учетом статьи 5."]
    assert _kinds(paragraphs) == [LawLineKind.ARTICLE_TITLE, LawLineKind.NORM]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Статья 70. Испытание при приеме", ArticleTitle("70", "Испытание при приеме")),
        ("Статья 13.11. Нарушение порядка", ArticleTitle("13.11", "Нарушение порядка")),
        ("Статья 2. Установить ставку 5 процентов.", ArticleTitle("2", "")),
        ("Статья 7.", ArticleTitle("7", "")),
        ("Статья 59 изменена с 1 марта 2031 г. - Федеральный закон", None),
        ("Статья 13.11 дополнена частью 10 с 1 марта 2031 г.", None),
    ],
)
def test_article_title_is_read(line: str, expected: ArticleTitle | None) -> None:
    assert read_article_title(line) == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("1. Текст пункта.", UnitMarker("1", MarkerClosing.DOT, "Текст пункта.")),
        ("1.8-3. Ставка.", UnitMarker("1.8-3", MarkerClosing.DOT, "Ставка.")),
        (
            "2.1) тяжелое положение;",
            UnitMarker("2.1", MarkerClosing.BRACKET, "тяжелое положение;"),
        ),
        ("7) утратил силу;", UnitMarker("7", MarkerClosing.BRACKET, "утратил силу;")),
        ("13 процентов - если сумма меньше;", None),
        ("13.5 процента от суммы;", None),
        ("Без номера.", None),
    ],
)
def test_unit_marker_is_split_from_text(line: str, expected: UnitMarker | None) -> None:
    assert split_marker(line) == expected


@pytest.mark.parametrize(
    "stub",
    [
        "4. Утратил силу с 1 января 2030 г.",
        "2) утратил силу с 1 января 2029 г.;",
        "1.2-1. Утратил силу с 31 августа 2030 г. - Федеральный закон от 1 июля 2030 г. N 9-ФЗ",
        "7) утратил силу по истечении одного месяца со дня опубликования закона.",
        "Абзац второй исключен.",
        "абзац четвертый утратил силу;",
        "Абзац утратил силу с 1 января 2030 г. - Федеральный закон от 1 июля 2029 г. N 8-ФЗ",
        "абзацы третий - шестой утратили силу с 1 января 2030 г.",
        "Часть вторая утратила силу.",
    ],
)
def test_line_in_place_of_a_repealed_unit_is_not_a_norm(stub: str) -> None:
    assert _kinds([stub]) == [LawLineKind.REPEALED]


@pytest.mark.parametrize(
    "norm",
    [
        "17. Действия оператора, за исключением случаев, указанных в части 1, -",
        "3) услуг, за исключением услуг, утративших силу договоров;",
        "Абзац второй настоящего пункта не применяется.",
    ],
)
def test_norm_mentioning_exclusion_is_not_a_repealed_unit(norm: str) -> None:
    assert _kinds([norm]) == [LawLineKind.NORM]


@pytest.mark.parametrize(
    ("line", "count"),
    [
        ("Абзац второй исключен.", 1),
        ("абзац четвертый утратил силу;", 1),
        ("Абзац утратил силу с 1 января 2030 г.", 1),
        ("абзацы третий - шестой утратили силу с 1 января 2030 г.", 4),
        ("Абзацы 4 - 5 утратили силу с 1 января 2030 г. - Федеральный закон", 2),
        ("абзацы первый - десятый утратили силу;", 10),
        ("абзацы четвёртый — седьмой исключены.", 4),
        ("абзацы восьмой - девятый утратили силу;", 2),
        ("4. Утратил силу с 1 января 2030 г.", None),
        ("Часть вторая утратила силу.", None),
    ],
)
def test_repealed_paragraphs_are_counted(line: str, count: int | None) -> None:
    assert repealed_paragraph_count(line) == count


def test_repealed_part_is_recognised() -> None:
    assert is_repealed_part("Часть вторая утратила силу.")
    assert not is_repealed_part("Часть вторая настоящей статьи не применяется.")


def test_law_article_is_recognised_by_source_header_and_title() -> None:
    assert is_law_article("\n\n".join(ARTICLE))


def test_company_document_is_not_a_law_article() -> None:
    assert not is_law_article("ДОГОВОР ХРАНЕНИЯ № Х-01/2030\n1. Предмет договора\n")
    assert not is_law_article("Статья 5. Отпуск\nОтпуск предоставляется ежегодно.\n")
    assert not is_law_article("Источник: архив отдела кадров\nПриказ о приеме.\n")

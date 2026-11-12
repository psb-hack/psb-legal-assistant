from datetime import date

import pytest

from lex_document_header import read_header
from lex_documents import DocumentKind, DocumentStatus, Party

STORAGE_CONTRACT = """\
ДОГОВОР ХРАНЕНИЯ № Х-01/2030
г. Тверь — 5 мая 2030 г.
ООО «Ромашка» (ИНН 6900000001), именуемое в дальнейшем «Хранитель», с одной стороны, и
ИП Сидоров Семён Семёнович (ИНН 690000000002), именуемый в дальнейшем «Поклажедатель»,
с другой стороны, заключили настоящий договор о нижеследующем.
1. Предмет договора
1.1. Хранитель принимает оборудование. Тариф действует с 01.01.2031.
1.2. Прежний договор хранения № Х-09/2029 утратил силу с 05.05.2030.
"""

SUPPLEMENTARY_AGREEMENT = """\
ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ № 2
к договору хранения № Х-01/2030 от 5 мая 2030 г.
г. Тверь — 10 июня 2031 г.
ООО «Ромашка» («Хранитель») и ООО «Василёк» («Поклажедатель») заключили соглашение:
Изложить пункт 1.2 договора в следующей редакции:
«1.2. Срок хранения — 24 месяца».
"""

REPEALED_REGULATION = """\
РЕДАКЦИЯ УТРАТИЛА СИЛУ с 01.07.2031 после утверждения новой (приказ № 9 от 15.06.2031).
РЕГЛАМЕНТ ЗАКУПОК ООО «РОМАШКА»
Утверждён приказом директора № 3 от 01.02.2030
Редакция от 01.02.2030
1. Общие положения
1.1. Регламент определяет порядок закупок.
"""

CURRENT_REGULATION = """\
РЕГЛАМЕНТ ЗАКУПОК ООО «РОМАШКА»
Утверждён приказом директора № 9 от 15.06.2031
Редакция от 01.07.2031. Действует с 01.07.2031. Редакция от 01.02.2030 утрачивает силу.
1. Общие положения
1.1. Регламент определяет порядок закупок.
"""

LETTER = """\
ООО «РОМАШКА»
Бухгалтерия
Исх. № Б-3 от 02.03.2031
Руководителям подразделений
О сроках сдачи авансовых отчётов
Уважаемые коллеги!
Авансовый отчёт сдаётся в течение 3 рабочих дней.
Прежний порядок утратил силу с 01.03.2031.
"""

MEMO = """\
ООО «РОМАШКА»
Отдел кадров
СЛУЖЕБНАЯ ЗАПИСКА № ОК-4 от 20.04.2031
О графике отпусков
График отпусков утверждается до 15 декабря.
"""


def test_contract_title_and_number_come_from_its_first_line() -> None:
    header = read_header(STORAGE_CONTRACT)
    assert header.kind is DocumentKind.CONTRACT
    assert header.title == "Договор хранения № Х-01/2030"
    assert header.number == "Х-01/2030"


def test_contract_date_is_read_from_place_and_date_line() -> None:
    assert read_header(STORAGE_CONTRACT).dated == date(2030, 5, 5)


def test_parties_keep_name_taxpayer_number_and_role() -> None:
    assert read_header(STORAGE_CONTRACT).parties == (
        Party("ООО «Ромашка»", "6900000001", "Хранитель"),
        Party("ИП Сидоров Семён Семёнович", "690000000002", "Поклажедатель"),
    )


def test_statements_in_contract_clauses_do_not_reach_the_header() -> None:
    header = read_header(STORAGE_CONTRACT)
    assert header.effective_from is None
    assert header.status is DocumentStatus.IN_FORCE


def test_supplementary_agreement_is_named_together_with_its_contract() -> None:
    header = read_header(SUPPLEMENTARY_AGREEMENT)
    assert header.kind is DocumentKind.SUPPLEMENTARY_AGREEMENT
    assert header.title == (
        "Дополнительное соглашение № 2 к договору хранения № Х-01/2030 от 5 мая 2030 г."
    )
    assert header.number == "2"


def test_supplementary_agreement_is_dated_by_itself_not_by_its_contract() -> None:
    assert read_header(SUPPLEMENTARY_AGREEMENT).dated == date(2031, 6, 10)


def test_parties_in_short_form_have_no_taxpayer_number() -> None:
    assert read_header(SUPPLEMENTARY_AGREEMENT).parties == (
        Party("ООО «Ромашка»", None, "Хранитель"),
        Party("ООО «Василёк»", None, "Поклажедатель"),
    )


def test_repeal_banner_marks_the_document_as_repealed() -> None:
    header = read_header(REPEALED_REGULATION)
    assert header.status is DocumentStatus.REPEALED
    assert header.repealed_from == date(2031, 7, 1)


def test_order_named_in_repeal_banner_does_not_date_the_document() -> None:
    header = read_header(REPEALED_REGULATION)
    assert header.dated == date(2030, 2, 1)
    assert header.number is None


def test_regulation_title_is_read_without_capitals() -> None:
    header = read_header(REPEALED_REGULATION)
    assert header.kind is DocumentKind.REGULATION
    assert header.title == "Регламент закупок ООО «Ромашка»"


def test_repeal_of_previous_edition_leaves_new_edition_in_force() -> None:
    header = read_header(CURRENT_REGULATION)
    assert header.status is DocumentStatus.IN_FORCE
    assert header.repealed_from is None
    assert header.edition == date(2031, 7, 1)
    assert header.effective_from == date(2031, 7, 1)


def test_edition_naming_itself_as_repealed_is_repealed() -> None:
    header = read_header(
        "РЕГЛАМЕНТ ЗАКУПОК\n"
        "Редакция от 01.02.2030\n"
        "Редакция от 01.02.2030 утратила силу с 01.07.2031.\n"
        "1. Общие положения\n"
    )
    assert header.status is DocumentStatus.REPEALED
    assert header.repealed_from == date(2031, 7, 1)


def test_citation_title_names_the_edition() -> None:
    assert read_header(CURRENT_REGULATION).citation_title == (
        "Регламент закупок ООО «Ромашка» в редакции от 01.07.2031"
    )
    assert read_header(STORAGE_CONTRACT).citation_title == (
        "Договор хранения № Х-01/2030"
    )


def test_letter_is_recognised_by_its_outgoing_number() -> None:
    header = read_header(LETTER)
    assert header.kind is DocumentKind.LETTER
    assert header.title == "Письмо № Б-3 от 02.03.2031"
    assert header.number == "Б-3"
    assert header.dated == date(2031, 3, 2)


def test_letter_body_does_not_reach_the_header() -> None:
    assert read_header(LETTER).status is DocumentStatus.IN_FORCE


def test_memo_title_keeps_number_and_date() -> None:
    header = read_header(MEMO)
    assert header.kind is DocumentKind.MEMO
    assert header.title == "Служебная записка № ОК-4 от 20.04.2031"
    assert header.number == "ОК-4"
    assert header.dated == date(2031, 4, 20)
    assert header.parties == ()


@pytest.mark.parametrize(
    ("first_line", "kind"),
    [
        ("Контракт № 7 на поставку бумаги", DocumentKind.CONTRACT),
        ("Положение о закупках", DocumentKind.REGULATION),
        ("Докладная записка № 2 от 03.03.2031", DocumentKind.MEMO),
        ("Письмо № 15 от 04.04.2031", DocumentKind.LETTER),
    ],
)
def test_kind_follows_the_genre_word_of_the_title(
    first_line: str, kind: DocumentKind
) -> None:
    assert read_header(f"{first_line}\n1. Раздел\n").kind is kind


def test_genre_word_inside_another_word_does_not_set_the_kind() -> None:
    header = read_header("Договорённости сторон по итогам встречи\n1. Раздел\n")
    assert header.kind is DocumentKind.UNKNOWN


def test_title_typed_in_normal_case_keeps_its_abbreviations() -> None:
    header = read_header("Положение о работе с СЭД и ЭЦП\n1. Общие положения\n")
    assert header.title == "Положение о работе с СЭД и ЭЦП"


def test_unknown_genre_is_named_by_its_first_line() -> None:
    header = read_header("ПРИКАЗ № 5 от 01.02.2031\nО назначении ответственного\n")
    assert header.kind is DocumentKind.UNKNOWN
    assert header.title == "Приказ № 5 от 01.02.2031"
    assert header.number == "5"
    assert header.dated == date(2031, 2, 1)


def test_text_without_header_has_empty_title() -> None:
    header = read_header("1. Общие положения\n1.1. Регламент обязателен.\n")
    assert header.kind is DocumentKind.UNKNOWN
    assert header.title == ""
    assert header.number is None
    assert header.dated is None

from datetime import date

from lex_document_numbers import DocumentNumber
from lex_documents import DocumentId, DocumentKind
from lex_entity_catalog import Counterparty, build_catalog
from synthetic_counterparties import TEXTS, knowledge_base

CATALOG = build_catalog(knowledge_base())


def _counterparty(name: str) -> Counterparty:
    (found,) = (item for item in CATALOG.counterparties if item.name == name)
    return found


def test_counterparties_are_collected_from_headers_in_order_of_appearance() -> None:
    assert [(item.name, item.taxpayer_number) for item in CATALOG.counterparties] == [
        ("ООО «Дельта-Порт»", "6901000011"),
        ("ООО «Горизонт»", "6901000099"),
        ("ООО «Дельта-Портал»", "6901000022"),
        ("ООО «Южный Причал»", "6901000033"),
        ("ООО «Логистик»", "6901000044"),
        ("ООО «Логистика»", "6901000055"),
        ("АО «Кедр»", "6901000066"),
    ]


def test_party_without_taxpayer_number_is_the_counterparty_of_the_same_name() -> None:
    supplier = _counterparty("ООО «Дельта-Порт»")
    assert supplier.documents == (DocumentId("d01"), DocumentId("d03"))
    assert supplier.roles == ("Поставщик",)


def test_roles_of_one_counterparty_are_gathered_across_documents() -> None:
    assert _counterparty("ООО «Горизонт»").roles == (
        "Покупатель",
        "Арендатор",
        "Заказчик",
    )


def test_name_tokens_drop_legal_form_quotes_and_hyphen() -> None:
    tokens = _counterparty("ООО «Дельта-Порт»").name_tokens
    assert [token.nominative for token in tokens] == ["дельта", "порт"]
    assert "дельтой" in tokens[0].forms
    assert "портал" not in tokens[1].forms


def test_word_used_in_lowercase_in_the_base_is_marked_as_common() -> None:
    southern, pier = _counterparty("ООО «Южный Причал»").name_tokens
    assert (southern.nominative, southern.common) == ("южный", False)
    assert (pier.nominative, pier.common) == ("причал", True)


def test_documents_carry_kind_parsed_number_and_edition() -> None:
    entries = {entry.id: entry for entry in CATALOG.documents}
    assert entries[DocumentId("d01")].number == DocumentNumber("д", 21, 2031)
    assert entries[DocumentId("d01")].kind is DocumentKind.CONTRACT
    assert entries[DocumentId("d01")].title == "Договор поставки № Д-21/2031"
    # «№ 1» допсоглашения без договора ни на что не указывает.
    assert entries[DocumentId("d03")].number is None
    assert entries[DocumentId("d03")].kind is DocumentKind.SUPPLEMENTARY_AGREEMENT
    assert entries[DocumentId("d08")].number == DocumentNumber("ок", 4, None)
    assert entries[DocumentId("d09")].edition == date(2031, 2, 1)
    assert entries[DocumentId("d09")].number is None


def test_law_articles_are_not_catalogued() -> None:
    assert len(knowledge_base().documents) == len(TEXTS)
    assert len(CATALOG.documents) == len(TEXTS) - 1


def test_same_name_with_two_taxpayer_numbers_is_two_counterparties() -> None:
    texts = (
        TEXTS[0],
        TEXTS[0].replace("6901000011", "6901000012").replace("Д-21", "Д-22"),
        TEXTS[2],
    )
    namesakes = [
        (item.taxpayer_number, item.documents)
        for item in build_catalog(knowledge_base(texts)).counterparties
        if item.name == "ООО «Дельта-Порт»"
    ]
    assert namesakes == [
        ("6901000011", (DocumentId("d01"),)),
        ("6901000012", (DocumentId("d02"),)),
        (None, (DocumentId("d03"),)),
    ]


def test_name_without_quotes_loses_only_the_legal_form() -> None:
    contract = TEXTS[0].replace(
        "ООО «Дельта-Порт» (ИНН 6901000011)", "ИП Соколов Пётр Ильич"
    )
    (person, _) = build_catalog(knowledge_base((contract,))).counterparties
    assert person.name == "ИП Соколов Пётр Ильич"
    assert [token.nominative for token in person.name_tokens] == [
        "соколов",
        "петр",
        "ильич",
    ]

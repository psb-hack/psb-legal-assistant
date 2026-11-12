import pytest

from lex_declension import case_forms


@pytest.mark.parametrize(
    ("nominative", "form"),
    [
        ("Северный", "северного"),
        ("Северный", "северным"),
        ("Склад", "складу"),
        ("Альфа", "альфой"),
        ("Трейд", "трейдом"),
        ("Трейдинг", "трейдинга"),
        ("Меридиан", "меридианом"),
        ("Виктория", "викторией"),
        ("Энергия", "энергию"),
        ("Сталь", "сталью"),
        ("Заря", "зарей"),
        ("Хороший", "хорошего"),
        ("Северная", "северной"),
        ("Золотое", "золотого"),
        ("Новые", "новых"),
        ("Технологии", "технологиями"),
        ("Системы", "систем"),
        ("Развитие", "развитием"),
        ("Меркурий", "меркурия"),
        ("Строй", "строя"),
        ("Городок", "городка"),
        ("Купец", "купцом"),
        ("Купец", "купец"),
        ("Берёза", "березы"),
    ],
)
def test_case_form_of_the_same_word_is_generated(nominative: str, form: str) -> None:
    assert form in case_forms(nominative)


@pytest.mark.parametrize(
    ("nominative", "other_word"),
    [
        ("Трейд", "трейдинг"),
        ("Трейд", "трейдинга"),
        ("Трейдинг", "трейда"),
        ("Трейдинг", "трейдом"),
        ("Виктор", "викторией"),
        ("Виктория", "виктором"),
        ("Логистик", "логистикой"),
        ("Гарант", "гарантия"),
        ("Гарантия", "гаранта"),
        ("Лидер", "лидеров"),
        ("Банк", "банкинг"),
        ("Строй", "стройка"),
        ("Мира", "миром"),
        ("Склад", "складской"),
        ("Порт", "портал"),
    ],
)
def test_another_word_is_not_a_case_form(nominative: str, other_word: str) -> None:
    assert other_word not in case_forms(nominative)


@pytest.mark.parametrize("word", ["Trade", "А1", "ЮГ", "Склад3"])
def test_latin_short_and_numbered_words_are_not_declined(word: str) -> None:
    assert case_forms(word) == {word.lower()}


def test_forms_are_comparison_keys() -> None:
    assert "березе" in case_forms("БЕРЁЗА")

from lex_paragraphs import split_paragraphs


def test_each_non_empty_line_is_a_paragraph() -> None:
    assert split_paragraphs("Первый абзац.\n\n  Второй абзац.  \n") == [
        "Первый абзац.",
        "Второй абзац.",
    ]


def test_byte_order_mark_and_windows_line_endings_are_dropped() -> None:
    assert split_paragraphs("﻿ДОГОВОР\r\n1. Предмет\r\n") == [
        "ДОГОВОР",
        "1. Предмет",
    ]

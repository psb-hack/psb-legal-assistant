"""Иерархия ошибок ядра. Ловим конкретный тип там, где есть действие."""


class LexError(Exception):
    """Базовая ошибка ядра."""


class KnowledgeBaseError(LexError):
    """База не читается, пуста или нарушен её контракт."""


class ModelGatewayError(LexError):
    """Сбой вызова прокси (сеть, 4xx/5xx после повторов, пустой ответ)."""


class VerificationError(LexError):
    """Утверждение не подтверждено положением из контекста."""


class SubmissionError(LexError):
    """Нельзя записать корректный submission.csv."""

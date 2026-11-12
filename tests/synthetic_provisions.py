"""Выдуманные положения для тестов векторного и гибридного поиска."""

from __future__ import annotations

from lex_documents import ProvisionId
from lex_sanitizer import AdmittedProvision

SUPPLY = "Договор поставки № Д-21/2031 (ООО «Дельта-Порт»)"
LEASE = "Договор аренды № Ю-09/2031 (ООО «Южный Причал»)"
PREPAYMENT = AdmittedProvision(
    ProvisionId("d01.4"), f"{SUPPLY}, п. 4.1", "Покупатель вносит предоплату 30 %."
)
WARRANTY = AdmittedProvision(
    ProvisionId("d01.7"), f"{SUPPLY}, п. 5.2", "Гарантийный срок — 12 месяцев."
)
SUBLEASE = AdmittedProvision(
    ProvisionId("d04.9"), f"{LEASE}, п. 4.2", "Субаренда — с согласия Арендодателя."
)
RENT = AdmittedProvision(
    ProvisionId("d04.5"), f"{LEASE}, п. 3.1", "Плата — 90 000 рублей в месяц."
)
PROVISIONS = (PREPAYMENT, WARRANTY, SUBLEASE, RENT)

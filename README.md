# Юридический ассистент ПСБ

Команда Hack&Change 2026, кейс ПСБ «ИИ-ассистент для юристов банка».

Ассистент отвечает юристу на вопросы **только по предоставленной базе** договоров, внутренних регламентов и статей законов. В каждом ответе есть ссылка на конкретный документ и пункт. Если данных нет, ассистент говорит об этом прямо и не додумывает. Отказывает только тогда, когда ответа в базе действительно нет.

**Статус:** подготовка. Задачи поставлены, документация и golden-набор готовы, кода сервиса пока нет. Работа идёт по [доске в Jira](https://d12-team.atlassian.net/jira/software/projects/PSB/boards/100).

## С чего начать

1. **Прочитать за 30 минут:** [кейс и правила оценки](docs/01-case.md), [разбор базы знаний](docs/02-knowledge-base.md), [как должен отвечать ассистент](docs/03-scenarios.md).
2. **Получить доступ к доске** — проект PSB в Jira d12-team. Нет доступа — напишите тимлиду.
3. **Скачать данные** из [папки организаторов](https://drive.google.com/drive/folders/1lendasLVlcKUcuM8mOrWKHD-C2dWtm9F) в `data/`: `knowledge-base.zip`, `demo_questions.csv`, `demo_output.csv`, `sample_submission.zip`, `requirements.txt`. Каталог `data/` не коммитится.
4. **Взять задачу:** верхнюю свободную в колонке To Do. Правила веток, коммитов и PR — в [процессе работы](docs/08-workflow.md).
5. **Писать код по [стандартам](docs/06-engineering.md).** Задача закрывается по Definition of Done оттуда же.

## Главное о задаче

- Сабмит — ZIP с `solution.py`. Платформа запускает его в Docker на скрытых вопросах, ответы оценивает LLM-судья **относительно baseline** организаторов. Score = 0,7 × MatchRate + 0,3 × WinRate.
- **Один выдуманный факт обнуляет вопрос.** Отсюда обязательный верификатор ответа.
- **Необоснованный отказ тоже штрафуется.** Если ответ в базе есть, его нужно найти.
- Модель на платформе одна: DeepSeek V4.1 Flash. Эмбеддинги — bge-m3, реранкер — Qwen3 Reranker 8B, всё через прокси организаторов.
- Лимиты: 5 сабмитов в день, 20 минут на прогон, в архиве только корневые `.py`-файлы, зависимости — только из списка организаторов.
- Для допуска к экспертам нужен **работающий веб-чат**, показывающий источники ответа.

Подробно — в [docs/01-case.md](docs/01-case.md).

## Порядок задач

Задачи выполняются сверху вниз — так же они стоят на доске. Это очередь, а не расписание: несколько человек берут следующие свободные задачи параллельно, если задача не зависит от незавершённой задачи выше. Golden-набор (PSB-36) не требует кода, его можно начинать с первого дня.

Метки: **p0** — без этого нет сильного сабмита или допуска к экспертам; **p1** — заметно поднимает балл; **p2** — полировка.

| # | Задача | Метка | Эпик |
|---|---|---|---|
| | **Фаза 1. Первый рабочий сабмит** | | |
| 1 | [PSB-2](https://d12-team.atlassian.net/browse/PSB-2) Структура кода под плоский ZIP-сабмит (ADR) | p0 | A |
| 2 | [PSB-4](https://d12-team.atlassian.net/browse/PSB-4) Клиент OpenAI-совместимого прокси: chat, embeddings, rerank | p0 | A |
| 3 | [PSB-3](https://d12-team.atlassian.net/browse/PSB-3) Сборка ZIP и прогон в контейнере, повторяющем платформу | p0 | A |
| 4 | [PSB-7](https://d12-team.atlassian.net/browse/PSB-7) CI: линтер, типы, тесты и сборка архива на каждый PR | p1 | A |
| 5 | [PSB-5](https://d12-team.atlassian.net/browse/PSB-5) Бюджет времени и гарантированный submission.csv | p0 | A |
| 6 | [PSB-9](https://d12-team.atlassian.net/browse/PSB-9) Структурный разбор документов компании | p0 | B |
| 7 | [PSB-10](https://d12-team.atlassian.net/browse/PSB-10) Разбор статей законов и вынос редакционных пометок | p0 | B |
| 8 | [PSB-12](https://d12-team.atlassian.net/browse/PSB-12) Карантин инструкций, встроенных в документы базы | p0 | B |
| 9 | [PSB-13](https://d12-team.atlassian.net/browse/PSB-13) Каталог сущностей и алиасов | p0 | B |
| 10 | [PSB-11](https://d12-team.atlassian.net/browse/PSB-11) Карта связей между документами | p0 | B |
| 11 | [PSB-23](https://d12-team.atlassian.net/browse/PSB-23) Системный промпт и контракт ответа | p0 | D |
| 12 | [PSB-16](https://d12-team.atlassian.net/browse/PSB-16) Гибридный поиск: BM25 + bge-m3 + слияние рангов | p0 | C |
| 13 | [PSB-17](https://d12-team.atlassian.net/browse/PSB-17) Маршрутизация по сущностям из вопроса и истории диалога | p0 | C |
| 14 | [PSB-24](https://d12-team.atlassian.net/browse/PSB-24) Диалоги: последняя реплика и недоверие к истории | p0 | D |
| 15 | [PSB-6](https://d12-team.atlassian.net/browse/PSB-6) Первый сабмит и журнал сабмитов | p0 | A |
| | **Фаза 2. Достоверность, сценарии и допуск к экспертам** | | |
| 16 | [PSB-36](https://d12-team.atlassian.net/browse/PSB-36) Golden-набор: 60+ кейсов по всем сценариям | p0 | F |
| 17 | [PSB-37](https://d12-team.atlassian.net/browse/PSB-37) Локальный прогон и отчёт регрессий | p0 | F |
| 18 | [PSB-31](https://d12-team.atlassian.net/browse/PSB-31) Детерминированный верификатор ответа | p0 | E |
| 19 | [PSB-32](https://d12-team.atlassian.net/browse/PSB-32) Перегенерация и безопасный откат | p0 | E |
| 20 | [PSB-18](https://d12-team.atlassian.net/browse/PSB-18) Выбор редакции по дате события | p0 | C |
| 21 | [PSB-22](https://d12-team.atlassian.net/browse/PSB-22) Классификатор сценария запроса | p0 | D |
| 22 | [PSB-25](https://d12-team.atlassian.net/browse/PSB-25) Ложные предпосылки: выявление и исправление | p0 | D |
| 23 | [PSB-26](https://d12-team.atlassian.net/browse/PSB-26) Частичная информация и обоснованный отказ | p0 | D |
| 24 | [PSB-41](https://d12-team.atlassian.net/browse/PSB-41) HTTP API ассистента | p0 | G |
| 25 | [PSB-42](https://d12-team.atlassian.net/browse/PSB-42) Чат-интерфейс с панелью источников | p0 | G |
| 26 | [PSB-43](https://d12-team.atlassian.net/browse/PSB-43) Docker: Dockerfile, compose.yaml, .dockerignore, .env.example | p0 | G |
| 27 | [PSB-44](https://d12-team.atlassian.net/browse/PSB-44) README по требованиям финальной сдачи | p0 | G |
| | **Фаза 3. Усиление** | | |
| 28 | [PSB-28](https://d12-team.atlassian.net/browse/PSB-28) Устойчивость к смене роли, поддельным системным сообщениям и давлению | p1 | D |
| 29 | [PSB-27](https://d12-team.atlassian.net/browse/PSB-27) Противоправные запросы | p1 | D |
| 30 | [PSB-29](https://d12-team.atlassian.net/browse/PSB-29) Неоднозначный объект: уточнять или отвечать по всем вариантам | p1 | D |
| 31 | [PSB-38](https://d12-team.atlassian.net/browse/PSB-38) Локальный LLM-судья по методике организаторов | p1 | F |
| 32 | [PSB-19](https://d12-team.atlassian.net/browse/PSB-19) Реранкер Qwen3 с бюджетом и отключением | p1 | C |
| 33 | [PSB-14](https://d12-team.atlassian.net/browse/PSB-14) Предрасчёт индекса и проверка совпадения базы | p1 | B |
| 34 | [PSB-39](https://d12-team.atlassian.net/browse/PSB-39) Профилирование скорости и числа вызовов | p1 | F |
| 35 | [PSB-33](https://d12-team.atlassian.net/browse/PSB-33) Проверка, что пункт действительно подтверждает утверждение | p1 | E |
| 36 | [PSB-20](https://d12-team.atlassian.net/browse/PSB-20) Расширение контекста соседними положениями | p2 | C |
| 37 | [PSB-45](https://d12-team.atlassian.net/browse/PSB-45) Полировка интерфейса и доступность | p1 | G |
| 38 | [PSB-34](https://d12-team.atlassian.net/browse/PSB-34) Метрики достоверности в логах и интерфейсе | p2 | E |
| | **Фаза 4. Защита** | | |
| 39 | [PSB-47](https://d12-team.atlassian.net/browse/PSB-47) Презентация PDF | p1 | H |
| 40 | [PSB-48](https://d12-team.atlassian.net/browse/PSB-48) Сценарий живой демонстрации | p1 | H |
| 41 | [PSB-49](https://d12-team.atlassian.net/browse/PSB-49) Сценарий внедрения в банке | p2 | H |
| 42 | [PSB-50](https://d12-team.atlassian.net/browse/PSB-50) Подготовка к вопросам жюри | p2 | H |

## Эпики

| Эпик | Направление |
|---|---|
| [PSB-1](https://d12-team.atlassian.net/browse/PSB-1) A | Каркас, сабмит и инфраструктура |
| [PSB-8](https://d12-team.atlassian.net/browse/PSB-8) B | База знаний: разбор, связи, безопасность |
| [PSB-15](https://d12-team.atlassian.net/browse/PSB-15) C | Поиск и отбор контекста |
| [PSB-21](https://d12-team.atlassian.net/browse/PSB-21) D | Логика ответа и поведенческие сценарии |
| [PSB-30](https://d12-team.atlassian.net/browse/PSB-30) E | Контроль достоверности |
| [PSB-35](https://d12-team.atlassian.net/browse/PSB-35) F | Оценка качества |
| [PSB-40](https://d12-team.atlassian.net/browse/PSB-40) G | Веб-интерфейс и поставка |
| [PSB-46](https://d12-team.atlassian.net/browse/PSB-46) H | Презентация и защита |

## Документация

| Документ | О чём |
|---|---|
| [01 — Кейс и оценка](docs/01-case.md) | Формат сабмита, окружение, формула судьи, критерии экспертов |
| [02 — База знаний](docs/02-knowledge-base.md) | Все 21 документ, ключевые пункты, связи, редакции, ловушки |
| [03 — Сценарии](docs/03-scenarios.md) | Контракт ответа и поведение в 9 ситуациях |
| [04 — Архитектура](docs/04-architecture.md) | Целевой пайплайн, компоненты, бюджет вызовов |
| [05 — Качество](docs/05-quality.md) | Golden-набор, локальный судья, дисциплина сабмитов |
| [06 — Стандарты](docs/06-engineering.md) | Чистый код, тесты, Definition of Done, без «нейрослопа» |
| [07 — Безопасность](docs/07-security.md) | Секреты, инъекции через базу и вопрос, ПДн |
| [08 — Процесс](docs/08-workflow.md) | Доска, ветки, коммиты, PR, сабмиты |
| [09 — Финальная сдача](docs/09-final-delivery.md) | Чек-лист материалов и допуска |
| [Журнал сабмитов](docs/submissions.md) | Что отправляли и что получили |
| [Golden-набор](eval/README.md) | 39 кейсов: 18 демо и 21 командный |

## Структура репозитория

```
docs/        документация и ADR
eval/        golden-набор
.github/     шаблон PR
```

Целевая структура с кодом описана в [архитектуре](docs/04-architecture.md#целевая-структура-репозитория). Раздел README про запуск через Docker, переменные окружения и порты появится вместе с кодом (PSB-44).

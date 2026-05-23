# 3 Модуль: Inverted Index + Elias Gamma/Delta

## Структура проекта

```text
module3_index_project/
│
├── data/
│   └── pages.csv.gz
│
├── main.py
├── preprocessing.py
├── elias.py
├── indexer.py
├── search_engine.py
├── check_csv.py
├── requirements.txt
└── README.md
```
## Что делает код

1. Загружает исходный файл `pages.csv.gz`.
2. Берёт колонки:
   - `id`
   - `final_url`
   - `title`
   - `text_content`
3. Выполняет предобработку текста:
   - нижний регистр;
   - токенизация;
   - лемматизация русских и английских слов.
4. Строит инвертированный индекс.
5Сжимает posting lists через:
   - Gap + Elias Gamma;
   - Gap + Elias Delta.
5Сравнивает:
   - размер обычного индекса;
   - размер Gamma-сжатого индекса;
   - размер Delta-сжатого индекса;
   - скорость поиска по обычному и сжатым индексам.


## Как запустить

### 1. Установить зависимости

```bash
pip install -r requirements.txt
```

### 2. Проверить CSV

```bash
python check_csv.py
```

### 3. Установить сертификаты (опционально)

```bash
//Пример для MacOS
/Applications/Python\ 3.12/Install\ Certificates.command
```

### 4. Запустить основной код

```bash
python main.py
```

## Быстрый тест на части документов

В `main.py` можно временно поставить:

```python
MAX_DOCS = 1000
```

Для полного запуска по всем документам:

```python
MAX_DOCS = None
```

## Как поменять поисковый запрос

В `main.py` измени строку:

```python
QUERY_TEXT = ""
```

Например:

```python
QUERY_TEXT = "Машинное обучение"
```

или:

```python
QUERY_TEXT = "accuracy"
```

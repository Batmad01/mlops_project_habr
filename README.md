# MLOps Проект: Классификация тем статей с Хабра

В проекте выполняется multilabel-классификация статей по 57 темам с использованием `PyTorch Lightning`, для версионирования данных используется `DVC`

## Описание проекта

- Оптимизатор: AdamW
- Функция потерь: BCEWithLogitsLoss
- Метрики качества: f1_score (основная), precision, recall, hammin_loss (дополнительные)
- Версионирование данных: DVC
- Гиперпараметры обучения: Hydra
- Логирование: Hydra, Pl (TensorBoardLogger)

## Дополнительно

- В качестве модели была выбрана `rubert-tiny2`, её обучение осуществляется с настраиваемыми гиперпараметрами, которые можно задать в файле конфига hydra (`conf/config.yaml`) - для обучения простой бейзлайн модели или более продвинутой
- Для управления зависимостями используется `uv`, они зафиксированы в `pyproject.toml` и `uv.lock`
- В качестве code quality tools используются: `git pre commit hooks`, `ruff`, `prettier` и `codespell`, их настройки приведены в `.pre-commit-config.yaml`

## Setup

- Настройка проекта:

```bash
uv sync
```

- Загрузка данных:

```bash
uv run get_data.py
```

- Починка CUDA:

```bash
uv pip uninstall torch
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
```

- Запуск обучения:

```bash
uv run main.py train
```

- Просмотр залогированных метрик модели в TensorBoard:

```bash
uv run tensorboard --logdir tb_logs/habr_pytorch/version_0
```

## Структура проекта

```plaintext
├── README.md                   # Документация по проекту
├── checkpoints                 # Директория, куда сохраняются лучшие модели при обучении
├── conf
│   └── config.yaml             # Конфиг Hydra
├── data
│   ├── dataset.parquet         # Сэмпл датасета (10%) для быстрого обучения модели
│   └── sample.parquet          # Полный датасет (если потребуются все данные)
├── data.dvc                    # Метаданные DVC для управления версиями
├── get_data.py                 # Скрипт для загрузки датасета
├── habr_articles_classifier
│   ├── model.py                # Архитектура модели
│   ├── module.py               # Классы датасета и дата модуля
│   ├── train.py                # Скрипт для обучения модели
│   └── utils.py                # Вспомогательные функции
├── logs
│   └── hydra                   # Логи Hydra
├── main.py                     # Энтрипоинт для запуска проекта (обучения модели)
├── pyproject.toml              # Настройки пакетов питона
├── tb_logs
│   └── habr_pytorch            # Логи Pl и метрики моделей, файлы TensorBoardLogger для графиков
└── uv.lock                     # Лок зависимостей для менеджера uv
```

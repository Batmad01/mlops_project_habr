import re

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MultiLabelBinarizer


# Функция для очистки текста от лишних символов
def clean_text(text: str) -> str:
    """
    Очищает текст, удаляя HTML-теги, специальные символы, пунктуацию и цифры,
    приводит текст к нижнему регистру

    Параметры:
    text (str): Исходный текст для очистки

    Возвращает:
    str: Очищенный текст
    """

    # Замена переносов строк на пробелы
    text = text.replace("\n", " ")
    # Удаление HTML-тегов
    text = re.sub(r"<.*?>", "", text)
    # Удаление специальных символов, пунктуации
    text = re.sub(r"[^\w\s]", "", text)
    # Удаление цифр
    text = re.sub(r"\d+", "", text)
    return text


# Функция предобработка датасета
def df_preprocess(
    df: pd.DataFrame,
) -> tuple[
    pd.Series,
    np.array,
    VarianceThreshold,
    dict[int, str],
    MultiLabelBinarizer,
]:
    """
    Предобработка датасета: фильтрация, объединение текстовых токенов,
    кодировка меток(хабов) и преобразование рейтингов.

    Параметры:
        df (pd.DataFrame): Исходный DataFrame.

    Возвращает:
        Tuple[
            pd.Series,                   # Предобработанный Series
            np.array,                    # Обработанные метки классов
            VarianceThreshold,           # Объект для отбора по порогу дисперсии
            Dict[int, str],              # Отображение индексов в названия меток
            MultiLabelBinarizer,         # Объект MultiLabelBinarizer
        ]
    """
    # Уберем слишком короткие (неинформативные) статьи
    df.columns = ["title", "hubs", "content", "url"]
    df["cleaned"] = df["content"].apply(clean_text)
    df["text_length"] = df["cleaned"].apply(len)
    df = df[df["text_length"] > 100].copy()

    # Объединяем текстовые токены в единую строку
    df["text"] = df["title"] + ". " + df["cleaned"]
    df = df.drop(columns=["title", "cleaned", "content", "text_length", "url"]).copy()

    # Извлекаем уникальные метки из тегов
    unique_labels = set()
    df["hubs"].str.split(", ").apply(unique_labels.update)  # Собираем уникальные метки

    # Маппинг и обратный маппинг хабов
    label_to_index = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    # Преобразуем строки в списки индексов
    df["labels"] = df["hubs"].apply(lambda x: [label_to_index[label] for label in x.split(", ")])
    df = df.drop(columns="hubs")

    # Преобразование hubs_ecoded в формат матрицы с уникальными метками хабов
    mlb = MultiLabelBinarizer()
    y_multi = mlb.fit_transform(df["labels"])

    # Удаляем метки, которые встречаются в менее чем 1% случаев
    selector = VarianceThreshold(threshold=0.01)
    y_multi_ = selector.fit_transform(y_multi)

    # Убираем строки с пустыми хабами
    indices = np.where(y_multi_.sum(axis=1) != 0)[0]
    X_reduced = df["text"].iloc[indices]
    y_reduced = y_multi_[indices]

    return (X_reduced, y_reduced)

# app/models/train_model.py
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import re
from typing import Union

# Optional: если у тебя есть отдельный preprocess, попробуем импортировать его
try:
    from app.utils.preprocess import preprocess_text as external_clean_text
except Exception:
    external_clean_text = None


# -------------------------
# Конфигурация по умолчанию
# -------------------------
DEFAULT_CATEGORIES = [
    {"id": 7, "ru_name": "Попутка"},
    {"id": 8, "ru_name": "Недвижимость"},
    {"id": 9, "ru_name": "Авто"},
    {"id": 10, "ru_name": "Скот"},
    {"id": 11, "ru_name": "Купить/Продать"},
    {"id": 12, "ru_name": "Работа"},
    {"id": 13, "ru_name": "Новости района"},
]   


# -------------------------
# Utilities
# -------------------------
def simple_clean_text(text: str) -> str:
    """
    Простая очистка текста: lowercase, удаление non-cyrillic/latin/digits/spaces,
    удаление лишних пробелов.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # заменим разнообразные небуквенные символы на пробел
    text = re.sub(r"[^\w\s\u0400-\u04FF]", " ", text)  # позволяет кириллицу
    # убираем многократные пробелы
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_categories(path: Union[str, None] = None):
    """
    Загружает категории. Если path не указан — возвращает DEFAULT_CATEGORIES.
    """
    if path:
        p = Path(path)
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return DEFAULT_CATEGORIES


# -------------------------
# Основная логика обучения
# -------------------------
def train(
    csv_path: str,
    out_dir: str = "data/models",
    categories_path: Union[str, None] = None
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 20000,
):
    csv_path = Path(csv_path)
    assert csv_path.exists(), f"Файл с данными не найден: {csv_path}"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем данные
    df = pd.read_csv(csv_path)
    if "description" not in df.columns or "category_id" not in df.columns:
        raise ValueError("CSV должен содержать колонки: description, category_id")

    df = df.dropna(subset=["description", "category_id"])
    df["description"] = df["description"].astype(str)

    # Очистка текста
    if external_clean_text:
        df["clean"] = df["description"].apply(external_clean_text)
    else:
        df["clean"] = df["description"].apply(simple_clean_text)

    # Загрузка/маппинг категорий
    categories = load_categories(categories_path)

    # Построим словари соответствий: ru_name -> id, и допустимые id
    name_to_id = {str(c.get("ru_name")).strip(): c.get("id") for c in categories}
    valid_ids = set(c.get("id") for c in categories)

    # Нормализация category_id в числовой id (учтём строки с числами и русские названия)
    def normalize_category_id(val):
        if pd.isna(val):
            return None
        if isinstance(val, (int, np.integer)):
            return int(val)
        s = str(val).strip()
        # если это число в строке
        if s.isdigit():
            return int(s)
        # если это русское название из categories
        if s in name_to_id:
            return int(name_to_id[s])
        return None

    df["category_id_norm"] = df["category_id"].apply(normalize_category_id)

    # Уберём записи с неизвестной/пустой категорией
    before = len(df)
    df = df.dropna(subset=["category_id_norm"]).copy()
    df["category_id_norm"] = df["category_id_norm"].astype(int)
    if len(df) < before:
        dropped = before - len(df)
        print(f"INFO: Отфильтровано записей с неизвестными категориями: {dropped}")

    # Удалим редкие классы (<2 объектов), чтобы работал stratify
    counts = df["category_id_norm"].value_counts()
    keep_ids = set(counts[counts >= 2].index.tolist())
    rare_ids = set(counts[counts < 2].index.tolist())
    if rare_ids:
        print("WARNING: Удаляем редкие категории (<2 образцов):", rare_ids)
        df = df[df["category_id_norm"].isin(keep_ids)].copy()

    # Если классов стало меньше 2 — отключим stratify
    unique_labels = sorted(df["category_id_norm"].unique().tolist())
    use_stratify = df["category_id_norm"].values if len(unique_labels) >= 2 else None
    if use_stratify is None:
        print("WARNING: Недостаточно классов для стратификации — stratify=None")

    # создаём ordered mapping: category_id -> label_index только по оставшимся классам
    cat_ids = sorted(unique_labels)
    id_to_index = {cid: idx for idx, cid in enumerate(cat_ids)}
    index_to_id = {v: k for k, v in id_to_index.items()}

    df["label"] = df["category_id_norm"].map(id_to_index)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"].values,
        df["label"].values,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"].values if use_stratify is not None else None,
    )

    # Vectorizer
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Model
    clf = LogisticRegression(
        solver="liblinear",  # для малого датасета ok
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
    )
    clf.fit(X_train_tfidf, y_train)

    # Predict & Metrics
    y_pred = clf.predict(X_test_tfidf)
    y_prob = clf.predict_proba(X_test_tfidf) if hasattr(clf, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print("Classification report (per label):")
    for label_idx, stats in report.items():
        # sklearn returns 'accuracy' as key as well
        if label_idx.isdigit() if isinstance(label_idx, str) else isinstance(label_idx, (int,)):
            print(label_idx, stats)
    # Сохраняем модели и артефакты
    vectorizer_path = out_dir / "vectorizer.pkl"
    model_path = out_dir / "classifier.pkl"
    metadata_path = out_dir / "metadata.json"
    metrics_path = out_dir / "metrics.json"

    joblib.dump(tfidf, vectorizer_path)
    joblib.dump(clf, model_path)

    metadata = {
        "id_to_index": id_to_index,
        "index_to_id": index_to_id,
        "categories": categories,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    metrics = {
        "accuracy": float(acc),
        "report": report,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Saved vectorizer ->", vectorizer_path)
    print("Saved classifier ->", model_path)
    print("Saved metadata ->", metadata_path)
    print("Saved metrics ->", metrics_path)

    return {
        "vectorizer": str(vectorizer_path),
        "model": str(model_path),
        "metadata": str(metadata_path),
        "metrics": str(metrics_path),
    }


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + LogisticRegression classifier")
    parser.add_argument("--data", type=str, default="data/training_data.csv", help="CSV file with training data")
    parser.add_argument("--out", type=str, default="data/models", help="Output folder for model artifacts")
    parser.add_argument("--categories", type=str, default=None, help="Path to categories.json (optional)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--max-features", type=int, default=20000, help="Max TF-IDF features")
    args = parser.parse_args()

    result = train(
        csv_path=args.data,
        out_dir=args.out,
        categories_path=args.categories,
        test_size=args.test_size,
        max_features=args.max_features,
    )
    print("Done.", result)


if __name__ == "__main__":
    main()

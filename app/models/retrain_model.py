# app/models/retrain_model.py
import hashlib
import json
from pathlib import Path
import pandas as pd
from app.models.train_model import train, simple_clean_text
from typing import Union

def compute_hash(text: str) -> str:
    """Возвращает хэш для строки (текста объявления)."""
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def append_new_data(
    new_csv_path: str,
    master_csv_path: str = "data/training_data.csv",
    used_hashes_path: str = "data/used_hashes.txt",
):
    """
    Добавляет новые confident объявления в общий датасет без дубликатов.
    Возвращает количество реально добавленных строк.
    """
    new_csv = Path(new_csv_path)
    master_csv = Path(master_csv_path)
    used_hashes = Path(used_hashes_path)

    # создаём, если не существует
    master_csv.parent.mkdir(parents=True, exist_ok=True)
    if not master_csv.exists():
        master_df = pd.DataFrame(columns=["description", "category_id"])
    else:
        master_df = pd.read_csv(master_csv)

    # читаем новые confident данные
    new_df = pd.read_csv(new_csv)
    if "description" not in new_df.columns or "category_id" not in new_df.columns:
        raise ValueError("CSV должен содержать 'description' и 'category_id'")

    # читаем хэши уже использованных текстов
    used = set()
    if used_hashes.exists():
        with used_hashes.open("r", encoding="utf-8") as f:
            used = {line.strip() for line in f if line.strip()}

    # отфильтровываем новые
    added_rows = []
    for _, row in new_df.iterrows():
        desc = str(row["description"])
        h = compute_hash(desc)
        if h not in used:
            added_rows.append(row)
            used.add(h)

    if not added_rows:
        print("🔹 Новых данных для обучения нет.")
        return 0

    added_df = pd.DataFrame(added_rows)
    combined = pd.concat([master_df, added_df], ignore_index=True)

    # сохраняем общий датасет и хэши
    combined.to_csv(master_csv, index=False)
    with used_hashes.open("w", encoding="utf-8") as f:
        f.write("\n".join(sorted(used)))

    print(f"✅ Добавлено {len(added_rows)} новых записей (всего теперь {len(combined)})")
    return len(added_rows)


def retrain_with_new_data(
    new_csv_path: str,
    categories_path: Union[str, None] = None,
    out_dir: str = "data/models",
):
    """
    Основная функция перетренировки:
    1. добавляет новые confident данные в общий датасет
    2. вызывает train() из train_model.py
    """
    added = append_new_data(new_csv_path)
    if added == 0:
        print("⚠️ Обучение не запущено — нет новых данных.")
        return None

    print("🚀 Запуск переобучения на расширенном датасете...")
    result = train(
        csv_path="data/training_data.csv",
        out_dir=out_dir,
        categories_path=categories_path,
    )
    print("🎯 Переобучение завершено:", result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Incremental retraining utility")
    parser.add_argument("--new-data", type=str, required=True, help="CSV с новыми confident объявлениями")
    parser.add_argument("--categories", type=str, default=None, help="Path к categories.json (опционально)")
    args = parser.parse_args()

    retrain_with_new_data(args.new_data, categories_path=args.categories)

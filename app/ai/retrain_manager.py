# app/ai/retrain_manager.py
import threading
import pandas as pd
from pathlib import Path
from app.models.retrain_model import retrain_with_new_data

# Настройки
NEW_CONFIDENT_PATH = Path("data/new_confident.csv")
RETRAIN_THRESHOLD = 10  # сколько confident объявлений нужно для переобучения
LOCK = threading.Lock()


def append_confident_ad(description: str, category_id: int):
    """
    Добавляет confident объявление в буфер CSV.
    Возвращает текущее количество записей после добавления.
    """
    NEW_CONFIDENT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if NEW_CONFIDENT_PATH.exists():
        df = pd.read_csv(NEW_CONFIDENT_PATH)
    else:
        df = pd.DataFrame(columns=["description", "category_id"])

    new_row = pd.DataFrame([{"description": description, "category_id": category_id}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(NEW_CONFIDENT_PATH, index=False)

    count = len(df)
    print(f"✅ Добавлено confident объявление ({count}/{RETRAIN_THRESHOLD})")
    return count


def trigger_retrain_if_needed():
    """
    Проверяет, накопилось ли достаточно confident объявлений для переобучения.
    Если да — запускает обучение в отдельном потоке.
    """
    if not NEW_CONFIDENT_PATH.exists():
        return

    df = pd.read_csv(NEW_CONFIDENT_PATH)
    count = len(df)

    if count < RETRAIN_THRESHOLD:
        return  # ждем накопления

    # запускаем в отдельном потоке, чтобы не блокировать API
    def retrain_thread():
        try:
            print("🚀 Достигнут порог, запускаем переобучение модели...")
            retrain_with_new_data(str(NEW_CONFIDENT_PATH))
            print("✅ Обучение завершено, очищаем временные данные...")
            NEW_CONFIDENT_PATH.unlink(missing_ok=True)
        except Exception as e:
            print("❌ Ошибка при переобучении:", e)

    with LOCK:  # защита от повторного запуска
        t = threading.Thread(target=retrain_thread, daemon=True)
        t.start()

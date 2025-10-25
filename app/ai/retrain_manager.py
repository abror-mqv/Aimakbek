# app/ai/retrain_manager.py
import threading
import pandas as pd
from pathlib import Path
from app.models.retrain_model import retrain_with_new_data
import logging
import os

# Настройки
NEW_CONFIDENT_PATH = Path("data/new_confident.csv")
RETRAIN_THRESHOLD = 10  # сколько confident объявлений нужно для переобучения
LOCK = threading.Lock()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def append_confident_ad(description: str, category_id: int):
    """
    Добавляет confident объявление в буфер CSV.
    Возвращает текущее количество записей после добавления.
    """
    try:
        logger.info(f"Рабочая директория: {os.getcwd()}")
        NEW_CONFIDENT_PATH.parent.mkdir(parents=True, exist_ok=True)

        if NEW_CONFIDENT_PATH.exists():
            df = pd.read_csv(NEW_CONFIDENT_PATH)
            logger.info(f"CSV найден, текущих записей: {len(df)}")
        else:
            df = pd.DataFrame(columns=["description", "category_id"])
            logger.info("CSV не найден, создаем новый DataFrame")

        new_row = pd.DataFrame([{"description": description, "category_id": category_id}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(NEW_CONFIDENT_PATH, index=False)
        count = len(df)
        logger.info(f"✅ Добавлено confident объявление ({count}/{RETRAIN_THRESHOLD})")
        return count

    except Exception as e:
        logger.error(f"❌ Ошибка при добавлении confident объявления: {e}", exc_info=True)
        return 0


def trigger_retrain_if_needed():
    """
    Проверяет, накопилось ли достаточно confident объявлений для переобучения.
    Если да — запускает обучение в отдельном потоке.
    """
    try:
        if not NEW_CONFIDENT_PATH.exists():
            logger.info("CSV с confident объявлениями не найден, ждем накопления...")
            return

        df = pd.read_csv(NEW_CONFIDENT_PATH)
        count = len(df)
        logger.info(f"Проверка порога: {count}/{RETRAIN_THRESHOLD}")

        if count < RETRAIN_THRESHOLD:
            logger.info("Недостаточно данных для переобучения")
            return

        # запускаем в отдельном потоке, чтобы не блокировать API
        def retrain_thread():
            try:
                logger.info("🚀 Достигнут порог, запускаем переобучение модели...")
                retrain_with_new_data(str(NEW_CONFIDENT_PATH))
                logger.info("✅ Обучение завершено, очищаем временные данные...")
                NEW_CONFIDENT_PATH.unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"❌ Ошибка при переобучении: {e}", exc_info=True)

        with LOCK:  # защита от повторного запуска
            t = threading.Thread(target=retrain_thread, daemon=True)
            t.start()

    except Exception as e:
        logger.error(f"❌ Ошибка при проверке порога для retrain: {e}", exc_info=True)

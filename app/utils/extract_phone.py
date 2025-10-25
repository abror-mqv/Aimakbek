import re

def extract_phone_number(text: str) -> str | None:
    if not text:
        return None

    # 🔹 Убираем все символы, кроме цифр, + и пробелов — оставляем потенциальные номера
    cleaned = re.sub(r"[^\d\+\s]", " ", text)

    # 🔹 Находим все последовательности из 6–13 цифр (с возможным плюсом)
    candidates = re.findall(r"(?:\+?\d[\d\s\-]{6,15}\d)", cleaned)

    if not candidates:
        return None

    # 🔹 Перебираем найденные номера и нормализуем
    for c in candidates:
        phone = re.sub(r"\D", "", c)  # удаляем всё кроме цифр

        # Приводим к стандарту +996
        if phone.startswith("0") and len(phone) == 10:
            phone = "+996" + phone[1:]
        elif phone.startswith("996") and len(phone) == 12:
            phone = "+" + phone
        elif phone.startswith("7") and len(phone) == 9:
            # допустим кто-то написал без нуля — добавим +996
            phone = "+996" + phone
        elif not phone.startswith("+"):
            phone = "+" + phone

        # Отфильтруем мусор — у реальных номеров Кыргызстана обычно 13 символов (+996XXXYYYYYY)
        if 12 <= len(phone) <= 13:
            return phone

    return None

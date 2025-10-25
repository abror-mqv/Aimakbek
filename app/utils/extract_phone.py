from typing import Union
import re

def extract_phone_number(text: str) -> Union[str, None]:
    if not text:
        return None

    cleaned = re.sub(r"[^\d\+\s]", " ", text)
    candidates = re.findall(r"(?:\+?\d[\d\s\-]{6,15}\d)", cleaned)

    if not candidates:
        return None

    for c in candidates:
        phone = re.sub(r"\D", "", c)

        if phone.startswith("0") and len(phone) == 10:
            phone = "+996" + phone[1:]
        elif phone.startswith("996") and len(phone) == 12:
            phone = "+" + phone
        elif phone.startswith("7") and len(phone) == 9:
            phone = "+996" + phone
        elif not phone.startswith("+"):
            phone = "+" + phone

        if 12 <= len(phone) <= 13:
            return phone

    return None

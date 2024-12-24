def parse_symbols(text):
    """Parse text symbols like △, 〇, and numerical corrections."""
    text = text.strip()
    if "△" in text:
        return -1 if text == "△" else int(text.strip("△")) * -1
    elif "〇" in text:
        return int(text.strip("〇"))
    elif text.isdigit():
        return int(text)
    return None

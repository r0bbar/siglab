def is_int_string(s: str) -> bool:
    if not s:
        return False
    return s.lstrip('-+').isdigit()

def is_float_string(s: str) -> bool:
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False

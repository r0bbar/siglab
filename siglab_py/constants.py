import enum
from typing import Union, List, Dict, Any

JSON_SERIALIZABLE_TYPES = Union[str, bool, int, float, None, List[Any], Dict[Any, Any]]

class LogLevel(enum.Enum):
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0
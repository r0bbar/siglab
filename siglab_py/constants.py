import enum
from typing import Union, List, Dict, Any

INVALID : int = -1

JSON_SERIALIZABLE_TYPES = Union[str, bool, int, float, None, List[Any], Dict[Any, Any]]

class LogLevel(enum.Enum):
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0

class TrendDirection(enum.Enum):
    UNDEFINED = 0
    HIGHER_HIGHS = 1
    LOWER_HIGHS = 2
    SIDEWAYS = 3
    HIGHER_LOWS = 4
    LOWER_LOWS = 5

    def to_string(self) -> str:
        return self.name.lower() if self != TrendDirection.UNDEFINED else ''

PositionStatus = enum.Enum("PositionStatus", 'UNDEFINED OPEN CLOSED SL')
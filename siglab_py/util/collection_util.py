from typing import Dict
import json
from datetime import datetime
import numpy as np

import logging

# What's this for? To avoid the following error when you json.dumps: TypeError: Object of type int64 is not JSON serializable
def recursive_clean_dict(data : Dict):
    logger = logging.getLogger()

    def _clean_val(v):
        if v is None:
            return None  # or whatever you prefer

        # NumPy integer types (int8/16/32/64, uint8/16/32/64)
        if isinstance(v, np.integer):
            return int(v)

        # NumPy boolean (very common escape)
        if isinstance(v, (np.bool_, bool)):  # bool just in case
            return bool(v)

        # NumPy floats
        if isinstance(v, (np.floating, np.float16, np.float32, np.float64)):
            return float(v)

        # NumPy datetime / timedelta (if you ever have them)
        if isinstance(v, np.datetime64):
            return np.datetime_as_string(v, unit='s')  # or your preferred format
        if isinstance(v, np.timedelta64):
            return str(v)  # or convert to seconds etc.

        # datetime objects (already good)
        if isinstance(v, datetime):
            return v.strftime("%Y%m%d %H:%M:%S")

        # catch anything else suspicious
        if isinstance(v, (np.void, np.str_, np.bytes_, np.generic)):
            return str(v)

        return v

    cleaned = {}
    for k, v in data.items():
        _v = _clean_val(v)
        
        if isinstance(_v, dict):
            cleaned[k] = recursive_clean_dict(v)

        elif isinstance(_v, list):
            l = []
            for x in _v:
                if isinstance(x, dict):
                    _x = recursive_clean_dict(x)
                else:
                    _x = _clean_val(x)
                l.append(_x)

            cleaned[k] = l
        
        else:
            cleaned[k] = _v

    return cleaned

    

from typing import Dict
import json
from datetime import datetime
import numpy as np

import logging

# What's this for? To avoid the following error when you json.dumps: TypeError: Object of type int64 is not JSON serializable
def recursive_clean_dict(data : Dict):
    logger = logging.getLogger()

    def _clean_val(v):
        if not v:
            return v
        if isinstance(v, np.integer): # ← catches int8/16/32/64, uint*, …
            return int(v)
        elif isinstance(v, (np.floating, np.float32, np.float64)):
            return float(v)
        elif isinstance(v, datetime):
            return v.strftime("%Y%m%d %H:%M:%S")
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

    

import unittest
import json
import numpy as np
from datetime import datetime

from util.collection_util import recursive_clean_dict

class CollectionUtilTests(unittest.TestCase):

    def test_recursive_clean_dict(self):
        data = {
            "name": "Anna",
            "age": np.int64(25),
            "height": np.float64(1.68),
            "created": datetime(2025, 6, 15, 14, 30),
            "settings": {"theme": "dark", "count": np.int64(42)},
            "scores": [np.int64(10), 20, np.int64(30)]
        }    
        cleaned = recursive_clean_dict(data)

        with self.assertRaises(TypeError): # Expected TypeError: Object of type int64 is not JSON serializable
            json.dumps(data)
            
        json.dumps(cleaned)
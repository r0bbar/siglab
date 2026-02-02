import unittest
import os
import time
from typing import List

from util.io_util import purge_old_file

# @unittest.skip("Skip all integration tests.")
class IOUtilTests(unittest.TestCase):

    def test_purge_old_file(self):
        test_files : List[str] = []
        dir : str = os.path.dirname(os.path.abspath(__file__))
        filename_regex_list : List[str] = [ "lo_candles_entry_.*\.csv" ]
        max_age_sec : int = 0 # Files older than 'max_age_sec' will be deleted if regex matches.

        for i in range(3):
            test_file : str = f"{dir}\\lo_candles_entry_SOLUSDTUSDT_{i}_{int(time.time())}.csv"
            test_files.append(test_file)

            with open(
                test_file,
                'w', encoding='utf-8'
            ) as f:
                f.write("hello test\n")
                
        files_purged : List[str] = purge_old_file(
            dir = dir,
            filename_regex_list = filename_regex_list,
            max_age_sec = max_age_sec
        )

        assert(files_purged == test_files)
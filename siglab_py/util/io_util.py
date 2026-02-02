import io
import os
import time
import re
from typing import List

def purge_old_file(
    dir : str,
    filename_regex_list : List[str],
    max_age_sec : int= 60*60*24 # default: 24 hrs
) -> int:
    timestamp_now_sec : int = int(time.time())
    num_files_purged : int = 0
    files_in_dir : List[str] = list(os.listdir(dir))
    files_purged : List[str] = []

    for item in files_in_dir:
        for regex in filename_regex_list:
            if re.match(regex, item):
                fullfilename : str = f"{dir}\{item}"
                file_created_timestamp_sec : int = int(os.path.getctime(fullfilename))
                age_sec : int = timestamp_now_sec - file_created_timestamp_sec
                if age_sec>=max_age_sec:
                    files_purged.append(fullfilename)
                    os.remove(fullfilename)

    return files_purged
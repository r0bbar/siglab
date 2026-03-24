import io
import os
import time
import re
from typing import List

def classify_path(s: str) -> str:
    # Linux generally uses forward slash: /usr/mary/tmp
    # Windows uses back slash C:\Users\Mary\OneDrive\Documents
    os_type : str = None
    unix_pattern = re.compile(r'^/|^\./|^\.\./|^~')
    windows_pattern = re.compile(r'^[a-zA-Z]:\\|^[a-zA-Z]:/')
    
    _s : Union[None, str] = None
    if '/' in s or '\\' in s:
        last_slash_idx = max(
            s.rfind('/'),
            s.rfind('\\')
        )
        _s = s[:last_slash_idx]

    s_win = s.replace('/', '\\')
    s_nix = s.replace('\\','/')

    if windows_pattern.search(s) or (_s and windows_pattern.search(_s)) or (s_win and windows_pattern.search(s_win)):
        os_type = "win"
    elif unix_pattern.search(s) or (_s and unix_pattern.search(_s)) or (s_nix and unix_pattern.search(s_nix)):
        os_type = "*nux"

    return os_type

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
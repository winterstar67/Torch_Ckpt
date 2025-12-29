import os
# from pathlib import Path


def find_root():
    dir_path = os.path.dirname(__file__)
    list_dir = os.listdir(dir_path)
    while "proj_dir.py" not in list_dir:
        dir_path = os.path.dirname(dir_path)
        list_dir = os.listdir(dir_path)

    return dir_path


PROJ_DIR = find_root()

def find_from_proj(target='str'):
    dir_path = PROJ_DIR
    file_dir_list = os.listdir(PROJ_DIR)
    while target not in file_dir_list:
        dir_names = []
        for file_name in file_dir_list:
            if "." not in file_name:
                dir_names.append(file_name)
            else:
                pass
        

        
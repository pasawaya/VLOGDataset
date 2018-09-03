
import os
import shutil


def safe_makedirs(dirs):
    if not isinstance(dirs, list):
        dirs = [dirs]

    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


def del_dirs(dirs):
    if not isinstance(dirs, list):
        dirs = [dirs]

    for dir_name in dirs:
        if os.path.isdir(dir_name):
            shutil.rmtree(dir_name)

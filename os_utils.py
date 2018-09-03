
import os


def safe_makedirs(dirs):
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

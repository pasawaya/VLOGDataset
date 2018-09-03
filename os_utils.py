
import os


def safe_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

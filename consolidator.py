
import os
import argparse
import shutil

parser = argparse.ArgumentParser(description='Download VLOG dataset in batches.')
parser.add_argument('batch_directory',
                    default='~/Desktop',
                    nargs=1,
                    type=str,
                    help='Location containing batch data.')
args = parser.parse_args()


path = args.batch_directory[0]
videos_path = os.path.join(path, 'videos')
annotations_path = os.path.join(path, 'annotations')
if not os.path.exists(videos_path):
    os.makedirs(videos_path)
if not os.path.exists(annotations_path):
    os.makedirs(annotations_path)

for directory in os.listdir(path):
    if os.path.isdir(os.path.join(path, directory)):
        for subdirectory in os.listdir(os.path.join(path, directory)):
            if os.path.isdir(os.path.join(path, directory, subdirectory)):
                destination = videos_path if 'videos_' in directory else annotations_path
                if not os.path.exists(os.path.join(destination, subdirectory)):
                    shutil.move(os.path.join(path, directory, subdirectory), destination)

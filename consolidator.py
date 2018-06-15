
import os
import argparse
import shutil

parser = argparse.ArgumentParser(description='Consolidates batches into a single data folder.')
parser.add_argument('batch_directory',
                    default='~/Desktop',
                    nargs=1,
                    type=str,
                    help='Location containing batch data.')
args = parser.parse_args()


root = args.batch_directory[0]
videos_path = os.path.join(root, 'videos')
annotations_path = os.path.join(root, 'annotations')
if not os.path.exists(videos_path):
    os.makedirs(videos_path)
if not os.path.exists(annotations_path):
    os.makedirs(annotations_path)

for directory in os.listdir(root):
    directory = os.path.join(root, directory)
    if os.path.isdir(directory):
        for subdirectory in os.listdir(directory):
            if os.path.isdir(os.path.join(directory, subdirectory)):
                destination = videos_path if 'videos_' in directory else annotations_path
                if not os.path.exists(os.path.join(destination, subdirectory)):
                    shutil.move(os.path.join(directory, subdirectory), destination)
                shutil.rmtree(os.path.join(directory, subdirectory))


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

# Move all batch directories to main video and annotation directories
for directory in os.listdir(root):
    directory = os.path.join(root, directory)
    if os.path.isdir(directory) and "videos_" in directory or "annotations_" in directory:
        for subdirectory in os.listdir(directory):
            if os.path.isdir(os.path.join(directory, subdirectory)):
                destination = videos_path if 'videos_' in directory else annotations_path
                if not os.path.exists(os.path.join(destination, subdirectory)):
                    shutil.move(os.path.join(directory, subdirectory), destination)


for directory in os.listdir(annotations_path):
    annotations_directory = os.path.join(annotations_path, directory)
    if os.path.isdir(annotations_directory):
        video_directory = os.path.join(videos_path, directory)

        # Delete videos with no annotations
        for root, subdir, files in os.walk(annotations_directory):

            # Check if video has any annotations
            has_annotations = '.png' in ' '.join(files)

            # Delete directories without annotations
            if not has_annotations:
                shutil.rmtree(video_directory)
                shutil.rmtree(annotations_directory)
                break

        # If video has annotations, resize and in-paint
        if os.path.exists(video_directory):
            print('remains')



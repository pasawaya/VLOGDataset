
import os
import argparse

parser = argparse.ArgumentParser(description='Download VLOG dataset in batches.')
parser.add_argument('host',
                    nargs=1,
                    type=str,
                    help='IP address of server to download videos to')
parser.add_argument('username',
                    nargs=1,
                    type=str,
                    help='Server username')
parser.add_argument('start',
                    default=0,
                    nargs=1,
                    type=int,
                    help='The index of the first video to download')
parser.add_argument('stop',
                    default=17097,
                    nargs=1,
                    type=int,
                    help='The index of the last video to download')
parser.add_argument('increment',
                    default=100,
                    nargs=1,
                    type=int,
                    help='Batch size')
parser.add_argument('output_path',
                    default='~/Desktop',
                    nargs=1,
                    type=str,
                    help='Local location to transfer data to.')
args = parser.parse_args()


username = args.username[0]
host = args.host[0]
start = args.start[0]
stop = args.stop[0]
increment = args.increment[0]
destination = args.output_path[0]

for i in range(start, stop, increment):
    print("\n---------------------------")
    print("Downloading Videos " + str(i) + "-" + str(i + increment - 1))
    print("---------------------------")

    ssh_command = 'ssh ' + username + '@' + host

    # Download command
    commands = ['source vlog_dataset/envs/vlog/bin/activate',
                'cd vlog_dataset/VLOGDataset',
                'python main.py --n_videos=' + str(increment) + ' --start_video_id=' + str(i)]
    download_command = ssh_command + ' "' + '; '.join(commands) + '"'

    # Copy to local machine command
    video_destination = os.path.join(destination, 'videos_batch_' + str(i))
    annotations_destination = os.path.join(destination, 'annotations_batch_' + str(i))
    scp_command = 'scp -r ' + username + '@' + host + '/home/' + username + '/vlog_dataset/VLOGDataset/'
    scp_commands = [scp_command + 'videos ' + video_destination,
                    scp_command + 'annotations ' + annotations_destination]

    # Clean up command
    commands = ['rm -r /home/zlz/vlog_dataset/VLOGDataset/videos',
                'rm -r /home/zlz/vlog_dataset/VLOGDataset/annotations']
    clean_command = ssh_command + ' "' + '; '.join(commands) + '"'

    # Execute commands
    commands = [download_command] + scp_commands + [clean_command]
    for command in commands:
        os.system(command + ' & wait')

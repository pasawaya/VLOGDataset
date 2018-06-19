
import os
import argparse

parser = argparse.ArgumentParser(description='Download VLOG dataset in batches.')
parser.add_argument('host',
                    type=str,
                    help='IP address of server to download videos to')
parser.add_argument('username',
                    type=str,
                    help='Server username')
parser.add_argument('start',
                    default=0,
                    type=int,
                    help='The index of the first video to download')
parser.add_argument('stop',
                    default=17097,
                    type=int,
                    help='The index of the last video to download')
parser.add_argument('increment',
                    default=100,
                    type=int,
                    help='Batch size')
parser.add_argument('output_path',
                    default='~/Desktop',
                    type=str,
                    help='Local location to transfer data to')
args = parser.parse_args()

increment = args.increment
server = args.username + '@' + args.host

for i in range(args.start, args.stop, increment):
    first = str(i)
    last = str(i + increment - 1)

    print("\n---------------------------")
    print("Downloading Videos " + first + "-" + last)
    print("---------------------------")

    ssh_command = 'ssh ' + server

    # Download command
    remote_commands = ['source vlog_dataset/envs/vlog/bin/activate',
                       'cd vlog_dataset/VLOGDataset',
                       'python main.py --n_videos=' + str(increment) + ' --start_video_id=' + str(i)]
    download_command = ssh_command + ' "' + '; '.join(remote_commands) + '"'

    # Copy to local machine command
    remote_path = '/home/' + args.username + '/vlog_dataset/VLOGDataset/data'
    local_path = os.path.join(args.output_path, 'data_batch_' + first + "_" + last)
    scp_command = 'scp -r ' + server + ':' + remote_path + ' ' + local_path

    # Clean up command
    clean_command = ssh_command + ' "rm -r ' + remote_path + '"'

    # Execute commands
    commands = [download_command, scp_command, clean_command]
    for command in commands:
        os.system(command + ' & wait')

import os
import sys
import random
import pdb


def merge_commands(commands, gpu_cnt=10, max_job_cnt=10000, shuffle=True, put_device_id=False):
    sys.stderr.write(f"Created {len(commands)} commands \n")
    if len(commands) == 0:
        return
    if shuffle:
        random.shuffle(commands)
    merge_cnt = (len(commands) + gpu_cnt - 1) // gpu_cnt
    merge_cnt = min(merge_cnt, max_job_cnt)
    current_device_idx = 0
    for idx in range(0, len(commands), merge_cnt):
        end = min(len(commands), idx + merge_cnt)
        concatenated_commands = "; ".join(commands[idx:end])
        if put_device_id:
            concatenated_commands = concatenated_commands.replace('cuda', f'cuda:{current_device_idx}')
        print(concatenated_commands)
        current_device_idx += 1
        current_device_idx %= gpu_cnt


def check_exists(logdir):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.exists(os.path.join(root_dir, '../', logdir, 'test_losses.log'))


def process_command(command):
    arr = command.split(' ')
    logdir = arr[arr.index('main.py') + 1]
    if check_exists(os.path.join('results', logdir)):
        sys.stderr.write(f"Skipping {logdir}\n")
        return []
    else:
        return [command]


#######################################################################################
# Comparing the tangling and the rotated mnist dataset
#######################################################################################
commands = []
device = 'cpu'
datasets = ['rmnist', 'tmnist']
epoch = 5
zs = [8]
zus = [2]
klu = 10
klqq = 50

for dataset in datasets:
    for z, zu in zip(zs, zus):
        command = f"python main.py cvae_{dataset}_klqq={klqq}_epoch={epoch}_z={z}_zu={zu} -d {dataset} -m Doubleburgess -md Doubleburgess -l CVAE --lr 0.001 -b 128 -e {epoch} -z {z} -zu {zu} --gamma-klu {klu} --gamma-klqq {klqq}"
        commands += process_command(command)

#######################################################################################
# CelebA split
#######################################################################################

#######################################################################################
# CelebA paired
#######################################################################################

#######################################################################################
# Video data
#######################################################################################

merge_commands(commands, gpu_cnt=1, put_device_id=True)

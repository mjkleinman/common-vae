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
    return os.path.exists(os.path.join(root_dir, '../../', logdir, 'test_losses.log'))


def process_command(command):
    arr = command.split(' ')
    logdir = arr[arr.index('main.py') + 1]
    if check_exists(os.path.join('results-paper', logdir)):
        sys.stderr.write(f"Skipping {logdir}\n")
        return []
    else:
        return [command]


commands = []
# device = 'cuda'
# datasets = ['ddsprites']
# epoch = 70
# zs = [8, 12]
# zus = [3, 6]
# klus = [0, 10]
# klqqs = [0.1]
# seeds = [0]
# batchs = [128]
#
# for seed in seeds:
#     for batch in batchs:
#         for dataset in datasets:
#             for z, zu in zip(zs, zus):
#                 for klu in klus:
#                     for klqq in klqqs:
#                         command = f"python main.py cvae_{dataset}_randSample_klqq={klqq}_klu={klu}_epoch={epoch}_batch={batch}_z={z}_zu={zu}_seed={seed} -s {seed} -d {dataset} -m Doubleburgess -md Doubleburgess -l CVAE --lr 0.001 -b {batch} -e {epoch} -z {z} -zu {zu} --gamma-klu {klu} --gamma-klqq {klqq} --device {device}"
#                         commands += process_command(command)

# #######################################################################################
# # Shapes3d correlated with both decoders (RUNNING ON W1)
# #######################################################################################

# commands = []
device = 'cuda'
datasets = ['dshapescorr']
epoch = 70
zs = [9]
zus = [3]
klus = [10] # 10 25 50
klqqs = [0.1]
batchs = [128]
seeds = [0, 1, 2]
decoder = ['Doubleburgess', 'Doubleburgessindeprecon']

for seed in seeds:
    for dataset in datasets:
        for z, zu in zip(zs, zus):
            for klu in klus:
                for klqq in klqqs:
                    for batch in batchs:
                        for dec in decoder:
                            command = f"python main.py cvae_{dataset}_randSample_dec={dec}_klqq={klqq}_klu={klu}_epoch={epoch}_batch={batch}_z={z}_zu={zu}_seed={seed} -s {seed} -d {dataset} -m Doubleburgess -md ${dec} -l CVAE --lr 0.001 -b {batch} -e {epoch} -z {z} -zu {zu} --gamma-klu {klu} --gamma-klqq {klqq} --device {device}"
                            commands += process_command(command)

# #######################################################################################
# Oct 9
# New Decoder only decoding on infomration per view
# #######################################################################################

# device = 'cuda'
# datasets = ['dshapescorr']
# epoch = 70
# zs = [9]
# zus = [3]
# klus = [10]
# klqqs = [0.1]
# batchs = [128]
# seeds = [0, 1, 2]
#
# for seed in seeds:
#     for dataset in datasets:
#         for z, zu in zip(zs, zus):
#             for klu in klus:
#                 for klqq in klqqs:
#                     for batch in batchs:
#                         command = f"python main.py cvae_{dataset}_randSample_PERVIEWRECON_klqq={klqq}_klu={klu}_epoch={epoch}_batch={batch}_z={z}_zu={zu}_seed={seed} -s {seed} -d {dataset} -m Doubleburgess -md Doubleburgessindeprecon -l CVAE --lr 0.001 -b {batch} -e {epoch} -z {z} -zu {zu} --gamma-klu {klu} --gamma-klqq {klqq} --device {device}"
#                         commands += process_command(command)

# merge_commands(commands, gpu_cnt=1, put_device_id=True)

# #######################################################################################
# Oct 9
# New Decoder only decoding on infomration per view
# #######################################################################################
device = 'cuda'
datasets = ['dshapes']
epoch = 70
zs = [9]
zus = [3]
klus = [10]
klqqs = [0.1]
batchs = [128]
seeds = [0, 1, 2]

for seed in seeds:
    for dataset in datasets:
        for z, zu in zip(zs, zus):
            for klu in klus:
                for klqq in klqqs:
                    for batch in batchs:
                        command = f"python main.py cvae_{dataset}_randSample_PERVIEWRECON_klqq={klqq}_klu={klu}_epoch={epoch}_batch={batch}_z={z}_zu={zu}_seed={seed} -s {seed} -d {dataset} -m Doubleburgess -md Doubleburgessindeprecon -l CVAE --lr 0.001 -b {batch} -e {epoch} -z {z} -zu {zu} --gamma-klu {klu} --gamma-klqq {klqq} --device {device}"
                        commands += process_command(command)

# #######################################################################################
# Oct 9
# New Decoder only decoding on infomration per view for dspprites
# #######################################################################################
device = 'cuda'
datasets = ['ddsprites2']
epoch = 70
zs = [7]
zus = [2]
klus = [10]
klqqs = [0.1]
seeds = [0, 1, 2]
batchs = [128]

for seed in seeds:
    for batch in batchs:
        for dataset in datasets:
            for z, zu in zip(zs, zus):
                for klu in klus:
                    for klqq in klqqs:
                        command = f"python main.py cvae_{dataset}_randSample_PERVIEWRECON_klqq={klqq}_klu={klu}_epoch={epoch}_batch={batch}_z={z}_zu={zu}_seed={seed} -s {seed} -d {dataset} -m Doubleburgess -md Doubleburgessindeprecon -l CVAE --lr 0.001 -b {batch} -e {epoch} -z {z} -zu {zu} --gamma-klu {klu} --gamma-klqq {klqq} --device {device}"
                        commands += process_command(command)


merge_commands(commands, gpu_cnt=1, put_device_id=True)
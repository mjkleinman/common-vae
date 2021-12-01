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
    return os.path.exists(os.path.join(root_dir, '../', logdir, 'train_losses.log'))


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
#commands = []
#device = 'cuda'
#datasets = ['rmnist', 'tmnist']
#epoch = 50
#zs = [8, 16]
#zus = [2, 4]
#klu = 10
#klqqs = [0.1, 0.5]
#
# for dataset in datasets:
#    for z, zu in zip(zs, zus):
#        for klqq in klqqs:
#            command = f"python main.py cvae_{dataset}_randSample_noAnneal_klqq={klqq}_epoch={epoch}_z={z}_zu={zu} -d {dataset} -m Doubleburgess -md Doubleburgess -l CVAE --lr 0.001 -b 128 -e {epoch} -z {z} -zu {zu} --gamma-klu {klu} --gamma-klqq {klqq} --device {device}"
#            commands += process_command(command)
#
########################################################################################
# dsprites common
# #######################################################################################

#commands = []
#device = 'cuda'
#datasets = ['ddsprites']
#epoch = 70
#zs = [8, 10]
#zus = [3, 4]
#klus = [10, 25, 50]
#klqqs = [0.1, 0.5]
#
# for dataset in datasets:
#    for z, zu in zip(zs, zus):
#        for klu in klus:
#            for klqq in klqqs:
#                command = f"python main.py cvae_{dataset}_randSample_klqq={klqq}_klu={klu}_epoch={epoch}_z={z}_zu={zu} -d {dataset} -m Doubleburgess -md Doubleburgess -l CVAE --lr 0.001 -b 128 -e {epoch} -z {z} -zu {zu} --gamma-klu {klu} --gamma-klqq {klqq} --device {device}"
#                commands += process_command(command)
#
#######################################################################################
# debugging dsprites
# #######################################################################################

#commands = []
#device = 'cuda'
#datasets = ['ddsprites2']
#epoch = 70
#zs = [7, 9]
#zus = [2, 3]
#klus = [10]
#klqqs = [0.1]
#
# for dataset in datasets:
#    for z, zu in zip(zs, zus):
#        for klu in klus:
#            for klqq in klqqs:
#                command = f"python main.py cvae_{dataset}_randSample_klqq={klqq}_klu={klu}_epoch={epoch}_z={z}_zu={zu} -d {dataset} -m Doubleburgess -md Doubleburgess -l CVAE --lr 0.001 -b 128 -e {epoch} -z {z} -zu {zu} --gamma-klu {klu} --gamma-klqq {klqq} --device {device}"
#                commands += process_command(command)
#

#######################################################################################
# debugging dshapes2
# #######################################################################################

#commands = []
#device = 'cuda'
#datasets = ['dshapes2']
#epoch = 70
#zs = [7]
#zus = [1]
#klus = [10]
#klqqs = [0.1]
#batchs = [32, 64]
#
# for dataset in datasets:
#    for z, zu in zip(zs, zus):
#        for klu in klus:
#            for klqq in klqqs:
#                for batch in batchs:
#                    command = f"python main.py cvae_{dataset}_randSample_klqq={klqq}_klu={klu}_epoch={epoch}_batch={batch}_z={z}_zu={zu} -d {dataset} -m Doubleburgess -md Doubleburgess -l CVAE --lr 0.001 -b {batch} -e {epoch} -z {z} -zu {zu} --gamma-klu {klu} --gamma-klqq {klqq} --device {device}"
#                    commands += process_command(command)
#
#######################################################################################
# CelebA split
########################################################################################
#epoch = 100
#dataset = 'dceleba'
#z = 64
#zu = 16
#klqq = 10
#klu = 10
#device = 'cuda'
# command = f"python main.py cvae_{dataset}_klqq={klqq}_epoch={epoch}_z={z}_zu={zu} -d {dataset} -m Doubleburgess -md Doubleburgess -l CVAE --lr 0.001 -b 128 -e {epoch} -z {z} -zu {zu} --gamma-klu {klu} --gamma-klqq {klqq}"
#commands += process_command(command)
#######################################################################################
# CelebA paired
#######################################################################################
commands = []
epoch = 50  # 100
dataset = 'pceleba'
zs = [32, 64]  # , 64]
zus = [8, 16]  # , 16]
klqqs = [0.1, 1]
klu = 10
device = 'cuda'
batchs = [32]

for z, zu in zip(zs, zus):
    for klqq in klqqs:
        for batch in batchs:
            command = f"python main.py cvae_{dataset}_klqq={klqq}_klu={klu}_epoch={epoch}_batch={batch}_z={z}_zu={zu} -d {dataset} -m Doubleburgess -md Doubleburgess -l CVAE --lr 0.001 -b 128 -e {epoch} -z {z} -zu {zu} --gamma-klu {klu} --gamma-klqq {klqq} -b {batch}"
            commands += process_command(command)


#######################################################################################
# Action
#######################################################################################
epoch = 30  # 100
datasets = ['ddspritesd', 'dshapesd']
zs = [10]  # 5  # , 64]
device = 'cuda'
batch = 64
free_bits = [0]
lrs = [0.0001]  # , 0.0005]
seeds = [0]
klqq = 0.01
betas = [2, 4, 16]

for seed in seeds:
    for fb in free_bits:
        for lr in lrs:
            for z in zs:
                for beta in betas:
                    for dataset in datasets:
                        command = f"python main.py avae_actpost_beta={beta}_klqq={klqq}_{dataset}_fb={fb}_epoch={epoch}_z={z}_lr={lr}_batch={batch}_seed={seed} -d {dataset} -m Burgess -md Burgess -l avae --free-bits {fb} --lr {lr} -b {batch} -e {epoch} -z {z} -zu 0 --no-test -s {seed} --gamma-klqq {klqq} --avae-beta {beta}"
                        commands += process_command(command)

#
#######################################################################################
# Video data
#######################################################################################

merge_commands(commands, gpu_cnt=1, put_device_id=True)

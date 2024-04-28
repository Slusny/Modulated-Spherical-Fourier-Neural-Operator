import os
import sys
import numpy as np
import glob
import argparse
import shutil
import re
#

# Argparser takes the path to the directory where checkpoints should be deleted
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--dest", type=str, default=None)
parser.add_argument("--keep-num", type=int)
parser.add_argument("--keep", type=str,nargs='+', default=[],)
   

args = parser.parse_args()

path = args.path

if path[-1] == "/":path = path[:-1]
if args.dest is not None:
    del_path = path
    path = args.dest
else:
    del_path = path+"-delete"
    # os.rename(path,del_path)
    os.system(f"mv {path} {del_path}")
    os.makedirs(path)

for file in ["losses.npy","val_means.npy","val_stds.npy"]:
    if os.path.exists(os.path.join(del_path,file)):
        shutil.move(os.path.join(del_path,file), os.path.join(path,file))

cp_list = list(sorted(glob.glob(os.path.join(del_path,"checkpoint_*")),key=len))
beta_list = list(sorted(glob.glob(os.path.join(del_path,"beta_*")),key=len))
gamma_list = list(sorted(glob.glob(os.path.join(del_path,"gamma_*")),key=len))


if len(args.keep) > 0:
    checkpoint_file = re.sub(r"\d+","{}",cp_list[-1].split("/")[-1])
    beta_file = re.sub(r"\d+","{}",beta_list[-1].split("/")[-1])
    gamma_file = re.sub(r"\d+","{}",gamma_list[-1].split("/")[-1])
    checkpoint_list_shorten = [os.path.join(del_path,checkpoint_file.format(checkpoint)) for checkpoint in args.keep]
    for cp in checkpoint_list_shorten:
        if not os.path.exists(cp):
            print(f"checkpoint {cp} does not exist")
            sys.exit(1)
    beta_list_shorten = [os.path.join(del_path,beta_file.format(beta)) for beta in args.keep]
    gamma_list_shorten = [os.path.join(del_path,gamma_file.format(gamma)) for gamma in args.keep]
elif args.keep_num > 1:
    checkpoint_list_shorten = cp_list[::len(cp_list)//args.keep_num]
    beta_list_shorten = beta_list[::len(beta_list)//args.keep_num]
    gamma_list_shorten = gamma_list[::len(gamma_list)//args.keep_num]
    # cp 
    if len(checkpoint_list_shorten)>args.keep_num:
        checkpoint_list_shorten[-1] = cp_list[-1]
    else:
        checkpoint_list_shorten.append(cp_list[-1])
    # beta
    if len(beta_list_shorten)>args.keep_num:
        beta_list_shorten[-1] = beta_list[-1]
    else:
        beta_list_shorten.append(beta_list[-1])
    # gamma
    if len(gamma_list_shorten)>args.keep_num:
        gamma_list_shorten[-1] = gamma_list[-1]
    else:
        gamma_list_shorten.append(gamma_list[-1])  
    
    checkpoint_list_shorten.pop(0)
    beta_list_shorten.pop(0)
    gamma_list_shorten.pop(0)
else:
    print("missing keep or keep_num argument")

for cp in checkpoint_list_shorten:
    if not os.path.exists(cp):print(f"checkpoint {cp} does not exist");continue
    shutil.move(cp, os.path.join(path,cp.split("/")[-1]))

for beta in beta_list_shorten:
    if not os.path.exists(cp):print(f"beta {beta} does not exist");continue
    shutil.move(beta, os.path.join(path,beta.split("/")[-1]))

for gamma in gamma_list_shorten:
    if not os.path.exists(gamma):print(f"beta {gamma} does not exist");continue
    shutil.move(gamma, os.path.join(path,gamma.split("/")[-1]))
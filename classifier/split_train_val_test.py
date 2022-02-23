from glob import glob
from tqdm import tqdm
import os
import random
import shutil


def copy_images(image_list,target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    for f in image_list:
        shutil.copy(f, target_dir)

dst_files = ["train","val","test"]

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

DIR = "traffic_lights/"
folders = os.listdir(DIR)
dst_dir = "dataset/"

for folder in folders:
    files = glob(DIR+folder+"/*")
    print(len(files))
    random.shuffle(files)
    train_inds =  int(len(files) * train_ratio)
    val_inds =  int(len(files) * val_ratio)

    train = files[:train_inds]
    val = files[train_inds-1:train_inds+val_inds]
    test = files[train_inds+val_inds-1:]
        
    copy_images(train, os.path.join(dst_dir,"train",folder))
    copy_images(val, os.path.join(dst_dir,"val",folder))
    copy_images(test, os.path.join(dst_dir,"test",folder))
    

import os
import argparse
from easydict import EasyDict as edict

import torch
import numpy as np

from utils.transforms import *


############################ depth config #############################
# depth = 6
depth = None
#######################################################################

# Dataset path
PATH = edict()

PATH.DB_ROOT = '/home/hscho/workspace/src/MLPD/Synscapes'
if depth is not None:
    PATH.JSON_GT_FILE = f'/home/hscho/workspace/src/MLPD/src/synscapes_scripts/imagesets/{depth}_synscape_test_annotation.json'

else:
    PATH.JSON_GT_FILE = '/home/hscho/workspace/src/MLPD/src/synscapes_scripts/imagesets/synscape_test_annotation.json'


# train
train = edict()

train.day = "all"
if depth is not None:
    train.img_set = f"/home/hscho/workspace/src/MLPD/src/synscapes_scripts/imagesets/{depth}_train_indices.txt"
else :
    train.img_set = f"/home/hscho/workspace/src/MLPD/src/synscapes_scripts/imagesets/train_indices.txt"
    

train.checkpoint = None

train.batch_size = 8 # batch size
train.start_epoch = 0  # start at this epoch
train.epochs = 40  # number of epochs to run without early-stopping
train.epochs_since_improvement = 3  # number of epochs since there was an improvement in the validation metric
train.best_loss = 100.  # assume a high loss at first
train.lr = 1e-4   # learning rate
train.momentum = 0.9  # momentum
train.weight_decay = 5e-4  # weight decay
train.grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
train.print_freq = 10
train.annotation = "AR-CNN" # AR-CNN, Sanitize, Original 





# test & eval
test = edict()
test.result_path = './result' ### coco tool. Save Results(jpg & json) Path
test.day = "all" # all, day, night

if depth is not None:
    test.img_set = f"/home/hscho/workspace/src/MLPD/src/synscapes_scripts/imagesets/{depth}_test_indices.txt"
else :
    test.img_set = f"/home/hscho/workspace/src/MLPD/src/synscapes_scripts/imagesets/test_indices.txt"

test.annotation = "AR-CNN"

# test.input_size = [512, 640]
test.input_size = [455, 570] # potenit
# test.input_size = [720, 1440]

### test model ~ eval.py
test.checkpoint = "/home/hscho/workspace/src/MLPD/src/synscapes_scripts/jobs/2024-07-29_08h12m_depth:6|epoch:100/checkpoint_ssd300.pth.tar065"
test.batch_size = 8

### train_eval.py
test.eval_batch_size = 1


# KAIST Image Mean & STD
## RGB
IMAGE_MEAN = [0.3465,  0.3219,  0.2842]
IMAGE_STD = [0.2358, 0.2265, 0.2274]
## Lwir
LWIR_MEAN = [0.1598]
LWIR_STD = [0.0813]

                    
# dataset
dataset = edict()

dataset.OBJ_LOAD_CONDITIONS = {    
                                  'train': {'hRng': (12, np.inf), 'xRng':(5, 635), 'yRng':(5, 507), 'wRng':(-np.inf, np.inf)}, 
                                  'test': {'hRng': (-np.inf, np.inf), 'xRng':(5, 635), 'yRng':(5, 507), 'wRng':(-np.inf, np.inf)}, 
                              }
# dataset.workers = 12
dataset.workers = 0

# Fusion Dead Zone
'''
Fusion Dead Zone
The input image of the KAIST dataset is input in order of [RGB, thermal].
Each case is as follows :
orignal, blackout_r, blackout_t, sidesblackout_a, sidesblackout_b, surroundingblackout
'''
FDZ_case = edict()

FDZ_case.original = ["None", "None"]

FDZ_case.blackout_r = ["blackout", "None"]
FDZ_case.blackout_t = ["None", "blackout"]

FDZ_case.sidesblackout_a = ["SidesBlackout_R", "SidesBlackout_L"]
FDZ_case.sidesblackout_b = ["SidesBlackout_L", "SidesBlackout_R"]
FDZ_case.surroundingblackout = ["None", "SurroundingBlackout"]


# main
args = edict(path=PATH,
             train=train,
             test=test,
             dataset=dataset,
             FDZ_case=FDZ_case)
# device
args.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
# args.device = 'cpu'

# depth
args.depth = depth

# n of class
args.n_classes = 11 # backg

args.exp_time = None
args.exp_name = None


## Semi Unpaired Augmentation
args.upaired_augmentation = ["TT_RandomHorizontalFlip",
                             "TT_FixedHorizontalFlip",
                             "TT_RandomResizedCrop"]
## Train dataset transform                             

# args["train"].img_transform = Compose([ ColorJitter(0.3, 0.3, 0.3), 
#                                         ColorJitterLWIR(contrast=0.3) ])

args["train"].img_transform = Compose([ Resize(test.input_size, "train"), 
                                        ColorJitter(0.3, 0.3, 0.3), 
                                        ColorJitterLWIR(contrast=0.3),
                                        ToTensor(),
                                        Normalize(IMAGE_MEAN, IMAGE_STD, 'R'),
                                        Normalize(LWIR_MEAN, LWIR_STD, 'T')
                                        ])
                               
args["train"].co_transform = Compose([  TT_RandomHorizontalFlip(p=0.5, flip=0.5), 
                                        TT_RandomResizedCrop([512,640], \
                                                                scale=(0.25, 4.0), \
                                                                ratio=(0.8, 1.2)), 
                                        ToTensor(), \
                                        Normalize(IMAGE_MEAN, IMAGE_STD, 'R'), \
                                        Normalize(LWIR_MEAN, LWIR_STD, 'T') ], \
                                        args=args)

## Test dataset transform
# args["test"].img_transform = Compose([ ])    
args["test"].img_transform = Compose([Resize(test.input_size, "train"), \
                                     ToTensor(), \
                                     Normalize(IMAGE_MEAN, IMAGE_STD, 'R'),
                                     Normalize(LWIR_MEAN, LWIR_STD, 'T')                       
                                    ])
args["test"].co_transform = Compose([Resize(test.input_size, "test"), \
                                     ToTensor(), \
                                     Normalize(IMAGE_MEAN, IMAGE_STD, 'R'), \
                                     Normalize(LWIR_MEAN, LWIR_STD, 'T')                        
                                    ])
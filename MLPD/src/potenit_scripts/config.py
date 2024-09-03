import os
import argparse
from easydict import EasyDict as edict

import torch
import numpy as np

from utils.transforms import *

# Dataset path
PATH = edict()
PATH.DB_ROOT = '/home/hscho/workspace/src/MLPD/potenit'  # potenit
PATH.JSON_GT_FILE = os.path.join('MDPI_test_solution.json')

# train
train = edict()

train.day = "all"
train.img_set = f"MDPI_train.txt" # potenit

train.checkpoint = None
# train.checkpoint = '/home/hscho/workspace/src/MLPD/src/jobs/2024-07-16_14h16m_/checkpoint_ssd300.pth.tar030'

# train.batch_size = 12 # batch size 
train.batch_size = 8 # batch size // hscho

train.start_epoch = 0  # start at this epoch
train.epochs = 10  # number of epochs to run without early-stopping
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

# test.img_set = f"test-{test.day}-20.txt" # kaist
test.img_set = "MDPI_test.txt" # potenit

test.annotation = "AR-CNN"

# test.input_size = [512, 640]
test.input_size = [455, 570]

### test model ~ eval.py
test.checkpoint = "./jobs/best_checkpoint.pth.tar"
test.batch_size = 4

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
    'train': {'hRng': (-np.inf, np.inf), 'xRng':(-np.inf, np.inf), 'yRng':(-np.inf, np.inf), 'wRng':(-np.inf, np.inf)},
    'test': {'hRng': (-np.inf, np.inf), 'xRng':(-np.inf, np.inf), 'yRng':(-np.inf, np.inf), 'wRng':(-np.inf, np.inf)}
}
# dataset.OBJ_LOAD_CONDITIONS = {    
#                                   'train': {'hRng': (12, np.inf), 'xRng':(5, 635), 'yRng':(5, 507), 'wRng':(-np.inf, np.inf)}, 
#                                   'test': {'hRng': (-np.inf, np.inf), 'xRng':(5, 635), 'yRng':(5, 507), 'wRng':(-np.inf, np.inf)}, 
#                               }

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

args.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
# args.device = "cpu"

args.exp_time = None
args.exp_name = None

args.n_classes = 2

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
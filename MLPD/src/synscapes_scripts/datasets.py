from torchvision.transforms import ToPILImage
import sys,os,json
import torchvision
import cv2
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.utils.data as data

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from utils.utils import *

from PIL import Image
import OpenEXR
import Imath
import numpy
import numexpr as ne
import readEXR
from get_paths import label_mapper
import copy

FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class SynscapesDataset(data.Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        args = kwargs['args']
        condition = kwargs['condition']
        self.parse_image_path = []
        # self.visualize_dots = visualize_dot()
        self.depth_threshold = args.depth
        
        self.args = args
        assert condition in args.dataset.OBJ_LOAD_CONDITIONS
        
        self.mode = condition
        self.image_set = args[condition].img_set
        self.img_transform = args[condition].img_transform
        # self.co_transform = args[condition].co_transform        
        self.co_transform = None
        self.cond = args.dataset.OBJ_LOAD_CONDITIONS[condition]
        self.annotation = args[condition].annotation       
        self.paths = {'rgb' : [], 'depth' : [], 'annotation' : []}

        # open indices file    
        with open(os.path.join('./imagesets', self.image_set), 'r') as file:
            lines = file.readlines()
        # allocate image indices
        for line in lines:
            line = line.strip("\n")

            # rgb, depth image paths
            self.paths['rgb'].append(os.path.join(self.args.path.DB_ROOT, 'img','rgb', f'{line}.png'))
            self.paths['depth'].append(os.path.join(self.args.path.DB_ROOT, 'img','depth', f'{line}.exr'))
            self.paths['annotation'].append(os.path.join(self.args.path.DB_ROOT, 'meta', f'{line}.json'))
            
            
    def __str__(self):
        return self.__class__.__name__ + '_' + self.image_set
        
    def __len__(self):
        return len(self.paths['rgb'])  

    def __getitem__(self, index): 
        vis_img, depth_img, boxes, labels, depths = self.pull_item(index)
        return vis_img, depth_img, boxes, labels, depths, torch.ones(1,dtype=torch.int)*index  
    
    def pull_item(self, index):          
        
        vis_img_path = self.paths['rgb'][index]
        depth_img_path = self.paths['depth'][index]
        annotation_path = self.paths['annotation'][index]
        
        # open image
        vis_img = Image.open(vis_img_path) # pil image
        depth_img = readEXR.read_exr(depth_img_path) # ndarray
        # depth_img_raw = copy.deepcopy(depth_img)

        # normalization 
        max_depth = depth_img.max() # 9023.712
        min_depth = depth_img.min() # 5.405345
        depth_cv2 = (((depth_img - min_depth) / (max_depth - min_depth)) * 255).astype('uint8')
        depth_img = Image.fromarray(depth_cv2)
        width, height = vis_img.size # (570, 455) (1440, 720)

        # get annotation
        boxes, labels, depths, boxes_for_visualize = self.get_annotation(json_path = annotation_path, w = width, h = height)
        # visualization # PIL, (455, 570, 3)
        # self.visualize_beforeTF(vis_img, boxes_for_visualize, depths, labels, index, "dataset_before_transform", depth_img.size)# PIL, (455, 570, 3)
        # self.visualize_dots.draw_dot(boxes_for_visualize, labels, save_path = './visualize_dots')
        
        vis_boxes = boxes
        depth_boxes = copy.deepcopy(boxes)
        
        vis_boxes = np.array(vis_boxes)
        depth_boxes = np.array(depth_boxes)
        labels = torch.tensor(labels)
        depths = torch.tensor(depths)
        
        if index == 1:
            print(f"before transforms size ---> width:{width}, height:{height}")
        # paired annotation
        if self.mode == 'train': 
            # before transform = (455, 570, 3) <- (h, w, c)
            vis_img, depth_img, boxes_vis, boxes_depth, _ = self.img_transform(vis_img, depth_img, vis_boxes, depth_boxes)
            
            # visualize after transformation  # after transform = torch.sizes(3, 455, 570)
            # self.visualize_afterTF(vis_img, vis_boxes, depths, labels, index, "dataset_after_transform", input_size=(570, 455))

            if index == 1:
                _, height, width = depth_img.shape                                                                   
                print(f"after transforms size ---> width:{width}, height:{height}")
            
            return vis_img, depth_img, boxes_vis, labels, depths

        else :
            if self.img_transform is not None:
                vis_img, depth_img, boxes_vis, boxes_depth, _ = self.img_transform(vis_img, depth_img, vis_boxes, depth_boxes)
            if index == 1:
                _, width, height = depth_img.shape
                print(f"after transforms size ---> width:{width}, height:{height}")
            # self.visualize_afterTF(vis_img, vis_boxes, depths, labels, index, "dataset_after_transform_test", input_size=(570, 455))
           
            return vis_img, depth_img, boxes_vis, labels, depths
        
    def get_annotation(self, json_path, w, h): # w, h
        
        annotation = {'bbox':[], 'category_id':[], 'depth':[], 'boxes_for_visualize':[]}
        
        with open(json_path, 'r') as json_file:
            data = json.load(json_file) # dict_keys(['camera', 'instance', 'scene'])
        json_file.close()
            
        data_camera = data['camera'] 
        data_instance = data['instance'] # dict_keys(['bbox2d', 'bbox3d', 'class', 'occluded', 'truncated'])
        data_scene = data['scene']
        
        for i, id in enumerate(data_instance['bbox2d']):
            if data_instance['occluded'][id] >= 0.5:
                continue
            if data_instance['truncated'][id] >= 0.5:
                continue
            if data_instance['class'][id] not in [24, 25, 26, 27, 28, 29, 30, 31, 32, 33]: 
                continue
            
            data_bbox = data_instance['bbox2d'][id]
            category_id = data_instance['class'][id] - (24-1)  # 1 ~ 10
            bbox = [data_bbox['xmin'], data_bbox['ymin'], data_bbox['xmax'], data_bbox['ymax']]
            
            ###################### planar depth ###########################
            depth_avg = round((data_bbox['zmin'] + data_bbox['zmax'])/2, 2)
            ################################################################
            if self.depth_threshold is not None:
                if depth_avg > self.depth_threshold: # if depth is bigger than depth_thr, than ignore
                    continue
        
            # append informations
            annotation['boxes_for_visualize'].append([ cur_pt * w if i % 2 == 0 else cur_pt * h for i, cur_pt in enumerate(copy.deepcopy(bbox))])
            annotation['bbox'].append(bbox)
            annotation['category_id'].append(category_id)
            annotation['depth'].append(depth_avg)
        
        if len(annotation['bbox']) == 0:
            return [[0,0,0,0]], [0], [0], [[0,0,0,0]] # backgound = 0

        return annotation['bbox'], annotation['category_id'], annotation['depth'], annotation['boxes_for_visualize']


    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        vis_img = list()
        depth_img = list()
        boxes = list()
        labels = list()
        depths = list()
        indices = list() 

        for b in batch:
            vis_img.append(b[0])
            depth_img.append(b[1])
            boxes.append(b[2])
            labels.append(b[3])
            depths.append(b[4]) 
            indices.append(b[5])
        
        vis_img = torch.stack(vis_img, dim=0)
        depth_img = torch.stack(depth_img, dim=0)

        return vis_img, depth_img, boxes, labels, depths, indices

       
        # return vis, depth, boxes, labels, index  
    def visualize_beforeTF(self, vis_img, vis_boxes, depths, labels, img_id, save_path = "./", input_size = (640,512)):
            
        save_path = f'{save_path}_{self.mode}'    
        os.makedirs(os.path.join(save_path), exist_ok=True)
        fnt = ImageFont.load_default()
        draw1 = ImageDraw.Draw(vis_img)
        for i, (box,label,depth) in enumerate(zip(vis_boxes, labels, depths)):
            # (xmin, ymin, xmax, ymax)
            draw1.rectangle(box, outline="red", width=2)
            draw1.text(((box[0]+box[2])/2, (box[1]+box[3])/2), f"{depth}", font=fnt, fill ="red")
            draw1.text((box[0] + (box[2] - box[0]), box[1]), f"{label}", font=fnt)
            
        vis_img.save(os.path.join(save_path, f"vis_img_{img_id}.jpg"))    
    
    def visualize_afterTF(self, vis_img, vis_boxes, depths, labels, img_id, save_path = "./", input_size = (640,512)):
        
        os.makedirs(os.path.join(save_path), exist_ok=True)
        transform_ = ToPILImage()
        vis_img = transform_(vis_img)
        
        fnt = ImageFont.load_default()
        draw1 = ImageDraw.Draw(vis_img)
        for i, (box,label,depth) in enumerate(zip(vis_boxes, labels, depths)):
            bbox = [box[0]*input_size[0], box[1]*input_size[1], box[2]*input_size[0], box[3]*input_size[1]]

            draw1.rectangle(bbox, outline="green", width=2)
            draw1.text(((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2), f"{depth}", font=fnt, fill ="red")
            draw1.text((bbox[0] + (bbox[2] - bbox[0]), bbox[1]), f"{label}", font=fnt, fill ="blue")
            
        vis_img.save(os.path.join(save_path, f"vis_img_{img_id}.jpg"))
    
    def depth_politics():
        # depth를 센서 과제에서 기준으로 제시한 것을 가지고 가깝고 멀고를 구분해서 통계냄
        pass
    
    def calculate_mean():
        pass
    
    def calculate_std():
        pass
    
class visualize_dot:
    def __init__(self):
        img_path = '/home/hscho/workspace/src/MLPD/Synscapes/img/rgb/9.png'
        self.vis_img = Image.open(img_path)
        self.draw = ImageDraw.Draw(self.vis_img)
        self.radius = 1
        self.bk = 0 # break 
    
    def draw_dot(self, vis_boxes, labels, save_path):
        
        save_path = f'{save_path}'    
        os.makedirs(os.path.join(save_path), exist_ok=True)
    
        for i, (box,label) in enumerate(zip(vis_boxes, labels)):
            if label == 24 - 24: # person
                color = (255,0,0) # r
            elif label == 26 - 24: # car
                color = (0,255,0) # g
            elif label == 32 - 24: # motorcycle
                color = (0,0,255) # b
            else:
                color = (0,0,0) # black
            x = (box[0]+box[2])/2
            y = (box[1]+box[3])/2
            self.draw.ellipse((x - self.radius, y - self.radius, x + self.radius, y + self.radius), fill=color)  
            
        self.bk += 1
        if self.bk % 1000 == 0: 
            self.vis_img.save(os.path.join(save_path, f"vis_dots.jpg"))
    
    

class LoadBox(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, bbs_format='xyxy'):
        assert bbs_format in ['xyxy', 'xywh']                
        self.bbs_format = bbs_format
        self.pts = ['x', 'y', 'w', 'h']

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """                
        res = [ [0, 0, 0, 0, -1] ]

        for obj in target.iter('object'):           
            name = obj.find('name').text.lower().strip()            
            bbox = obj.find('bndbox')
            bndbox = [ int(bbox.find(pt).text) for pt in self.pts ]

            if self.bbs_format in ['xyxy']:
                bndbox[2] = min( bndbox[2] + bndbox[0], width )
                bndbox[3] = min( bndbox[3] + bndbox[1], height )

            bndbox = [ cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox) ]
            
            bndbox.append(1)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind, occ]
            
        return np.array(res, dtype=np.float)  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

def parse_imgs_about_depth(dataset, save_path):
    with open('save_path', 'w') as file:
        data = file.writelines(dataset.parse_image_path)
    file.close()

if __name__ == '__main__':
    """Debug KAISTPed class"""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    import matplotlib.pyplot as plt
    import config
    
    args = config.args
    ###################### check #######################
    condition = 'train'
    ####################################################
    
    test_dataset = SynscapesDataset(args = args, condition = condition)
    dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True,
                              num_workers=0,
                              collate_fn=test_dataset.collate_fn,
                              pin_memory=True)
    
    from tqdm import tqdm
    for is_img, depth_img, boxes, labels, depths, idx in tqdm(dataloader):
        print(is_img.shape)
    
    
    ############## for parsing dataset w.r.t depth #######################    
    # with open('./parsed_image_paths.txt', 'w') as file:
    #     data = file.writelines(test_dataset.parse_image_path)
    # file.close()
    #######################################################################
    
    
    # from matplotlib import patches
    # from matplotlib import pyplot as plt
    # from utils.functional import to_pil_image, unnormalize
    # import config

    # def draw_boxes(axes, boxes, labels, target_label, color):
    #     for x1, y1, x2, y2 in boxes[labels == target_label]:
    #         w, h = x2 - x1 + 1, y2 - y1 + 1
    #         axes[0].add_patch(patches.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, lw=1))
    #         axes[1].add_patch(patches.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, lw=1))

    # args = config.args
    # test = config.test

    # fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    # dataset = KAISTPed(args, condition='test')

    # # HACK(sohwang): KAISTPed always returns empty boxes in test mode
    # dataset.mode = 'train'

    # vis, lwir, boxes, labels, indices = dataset[1300]

    # vis_mean = dataset.co_transform.transforms[-2].mean
    # vis_std = dataset.co_transform.transforms[-2].std

    # lwir_mean = dataset.co_transform.transforms[-1].mean
    # lwir_std = dataset.co_transform.transforms[-1].std

    # # C x H x W -> H X W x C
    # vis_np = np.array(to_pil_image(unnormalize(vis, vis_mean, vis_std)))
    # lwir_np = np.array(to_pil_image(unnormalize(lwir, lwir_mean, lwir_std)))

    # # Draw images
    # axes[0].imshow(vis_np)
    # axes[1].imshow(lwir_np)
    # axes[0].axis('off')
    # axes[1].axis('off')

    # # Draw boxes on images
    # input_h, input_w = test.input_size
    # xyxy_scaler_np = np.array([[input_w, input_h, input_w, input_h]], dtype=np.float32)
    # boxes = boxes * xyxy_scaler_np

    # draw_boxes(axes, boxes, labels, 3, 'blue')
    # draw_boxes(axes, boxes, labels, 1, 'red')
    # draw_boxes(axes, boxes, labels, 2, 'green')

    # frame_id = dataset.ids[indices.item()]
    # set_id, vid_id, img_id = frame_id[-1]
    # fig.savefig(f'{set_id}_{vid_id}_{img_id}.png')

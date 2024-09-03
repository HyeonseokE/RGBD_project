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


class KAISTPed(data.Dataset):
    def __init__(self, args, condition='train'):
        self.args = args
        assert condition in args.dataset.OBJ_LOAD_CONDITIONS
        
        self.mode = condition
        self.image_set = args[condition].img_set
        self.cond = args.dataset.OBJ_LOAD_CONDITIONS[condition]
        
        self.img_transform = args[condition].img_transform
        # self.co_transform = args[condition].co_transform   
        self.co_transform = None
        
        self.annotation = args[condition].annotation
        self._parser = LoadBox()        

        self._annopath = os.path.join('%s', 'RGBTDv4/Json', '%s', '%s', '%s.json') # potenit
        self._imgpath = os.path.join('%s', 'RGBTDv4/Image', '%s', '%s', '%s', '%s.png') # potenit
        self.ids = list()

        for line in open(os.path.join('../../potenit/RGBTDv4/ImageSet', self.image_set)):
            line = line.strip("\n")     

            self.ids.append((self.args.path.DB_ROOT, line.strip().split('/')))

    def __str__(self):
        return self.__class__.__name__ + '_' + self.image_set

    def __getitem__(self, index): 

        vis_img, depth_img, boxes, labels, depths = self.pull_item(index)
        return vis_img, depth_img, boxes, labels, depths, torch.ones(1,dtype=torch.int)*index  

    def pull_item(self, index):
        
        frame_id = self.ids[index]
        set_id, vid_id, img_id = frame_id[-1]
        
        # potenit
        vis_img = Image.open( self._imgpath % ( *frame_id[:-1], set_id, vid_id, 'RGB', img_id ))
        depth_img = Image.open( self._imgpath % ( *frame_id[:-1], set_id, vid_id, 'Depth', img_id ) )
        
        # normalization 
        max_depth = np.array(depth_img).max()
        min_depth = np.array(depth_img).min()
        depth_cv2 = (((np.array(depth_img) - min_depth) / (max_depth - min_depth)) * 255).astype('uint8')
        depth_img = Image.fromarray(depth_cv2).convert('L')
        width, height = depth_img.size # (570, 455)

        if index == 1:
            print(f"before transforms size ---> width:{width}, height:{height}")

        vis_boxes = []
        vis_boxes_for_visualize = []
        depth_boxes = []
        depths = []
        labels = []
            
        with open(self._annopath % ( *frame_id[:-1], set_id, vid_id, f"J{img_id[1:]}" ), "r") as file:
            data = json.load(file)
        
        if len(data['annotation']) == 0: # annotation이 없는 경우
            # bg annotation
            bndbox = [0, 0, 0, 0] 
            vis_boxes_for_visualize.append(bndbox)
            vis_boxes.append(bndbox)
            depth_boxes.append(bndbox)
            depths.append(torch.tensor(0))
            labels.append(torch.tensor(0)) # bg     
        else:# annotation이 있는경우
            for i in range(len(data["annotation"])):
                try:
                    category = data["annotation"][i]["category_str"]
                    if category == "None":
                        category = torch.tensor(0) # bg
                        depths.append(torch.tensor(0)) # depth
                    if category == "person":
                        category = torch.tensor(1) # person
                        depths.append(data["annotation"][i]["depth"]/1000)   # meter
                        # depths.append(data["annotation"][i]["depth"])          # milimeter
                except:
                    if 'category_str' not in data["annotation"][i].keys():
                        if data["annotation"][i]['category_id'] == 1: # person
                            category = torch.tensor(1) # person
                            depths.append(data["annotation"][i]["depth"]/1000)  # meter
                            # depths.append(data["annotation"][i]["depth"])         # milimeter
                        else:
                            category = torch.tensor(0) # bg
                            depths.append(torch.tensor(0)) # depth
                # (xmin, ymin, xmax, ymax)
                bndbox = data["annotation"][i]["bbox"]
                bndbox = np.array(bndbox, dtype = np.float)
                vis_boxes_for_visualize.append(bndbox.copy())
                bndbox = [ cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox) ]
                vis_boxes.append(bndbox)
                depth_boxes.append(bndbox)
                labels.append(category)
  
        vis_boxes = np.array(vis_boxes)
        depth_boxes = np.array(depth_boxes)
        labels = torch.tensor(labels)
        depths = torch.tensor(depths)

        # paired annotation
        if self.mode == 'train': 
            if self.img_transform is not None:
                vis_img, depth_img, boxes_vis, boxes_depth, _ = self.img_transform(vis_img, depth_img, vis_boxes, depth_boxes)
            # self.visualize(vis_img, vis_boxes, depths, labels, img_id, "dataset_after_transform", input_size=(570, 455))
            
            if index == 1:
                _, width, height = depth_img.shape                                                                   
                print(f"after transforms size ---> width:{width}, height:{height}")
            return vis_img, depth_img, boxes_vis, labels, depths

        else :
            if self.img_transform is not None:
                vis_img, depth_img, boxes_vis, boxes_depth, _ = self.img_transform(vis_img, depth_img, vis_boxes, depth_boxes, img_id)
            if index == 1:
                _, width, height = depth_img.shape
                print(f"after transforms size ---> width:{width}, height:{height}")
            # self.visualize(vis_img, vis_boxes, depths, labels, img_id, "dataset_after_transform_test")

            return vis_img, depth_img, boxes_vis, labels, depths
    
    def __len__(self):
        print(len(self.ids))
        return len(self.ids)

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
        indices = list() # difficulties
        
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
    def visualize(self, vis_img, vis_boxes, depths, labels, img_id, save_path = "./", input_size = (640,512)):
        if save_path in ["dataset_after_transform", "dataset_after_transform_test"]:
            transform_ = ToPILImage()
            vis_img = transform_(vis_img)
        
        fnt = ImageFont.load_default()
        draw1 = ImageDraw.Draw(vis_img)
        for i, (box,label,depth) in enumerate(zip(vis_boxes, labels, depths)):
            # draw1.rectangle(box.tolist(), outline="red", width=2)
            if save_path in ["dataset_after_transform", "dataset_after_transform_test"]:
                bbox = [box[0]*input_size[0], box[1]*input_size[1], box[2]*input_size[0], box[3]*input_size[1]]
            else:
                bbox = [box[0], box[1], box[2], box[3]] # (xmin, ymin, xmax, ymax)
            draw1.rectangle(bbox, outline="red", width=2)
            draw1.text(((box[0]+box[2])/2, (box[1]+box[3])/2), f"{depth}", font=fnt, fill ="red")
            draw1.text((box[1], box[3]), f"{label}", font=fnt, fill ="blue")
            
        vis_img.save(os.path.join(save_path, f"vis_img_{img_id}.jpg"))    
    
    

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


if __name__ == '__main__':
    """Debug KAISTPed class"""
    from matplotlib import patches
    from matplotlib import pyplot as plt
    from utils.functional import to_pil_image, unnormalize
    import config

    def draw_boxes(axes, boxes, labels, target_label, color):
        for x1, y1, x2, y2 in boxes[labels == target_label]:
            w, h = x2 - x1 + 1, y2 - y1 + 1
            axes[0].add_patch(patches.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, lw=1))
            axes[1].add_patch(patches.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, lw=1))

    args = config.args
    test = config.test

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    dataset = KAISTPed(args, condition='test')

    # HACK(sohwang): KAISTPed always returns empty boxes in test mode
    dataset.mode = 'train'

    vis, lwir, boxes, labels, indices = dataset[1300]

    vis_mean = dataset.co_transform.transforms[-2].mean
    vis_std = dataset.co_transform.transforms[-2].std

    lwir_mean = dataset.co_transform.transforms[-1].mean
    lwir_std = dataset.co_transform.transforms[-1].std

    # C x H x W -> H X W x C
    vis_np = np.array(to_pil_image(unnormalize(vis, vis_mean, vis_std)))
    lwir_np = np.array(to_pil_image(unnormalize(lwir, lwir_mean, lwir_std)))

    # Draw images
    axes[0].imshow(vis_np)
    axes[1].imshow(lwir_np)
    axes[0].axis('off')
    axes[1].axis('off')

    # Draw boxes on images
    input_h, input_w = test.input_size
    xyxy_scaler_np = np.array([[input_w, input_h, input_w, input_h]], dtype=np.float32)
    boxes = boxes * xyxy_scaler_np

    draw_boxes(axes, boxes, labels, 3, 'blue')
    draw_boxes(axes, boxes, labels, 1, 'red')
    draw_boxes(axes, boxes, labels, 2, 'green')

    frame_id = dataset.ids[indices.item()]
    set_id, vid_id, img_id = frame_id[-1]
    fig.savefig(f'{set_id}_{vid_id}_{img_id}.png')

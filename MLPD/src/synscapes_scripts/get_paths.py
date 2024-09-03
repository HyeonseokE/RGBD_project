import torch
import torchvision
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import json
import labels

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

def get_img_paths(root):
    '''
    input : images_paths_root
    function : parse image paths
    
    '''
    img_paths = {}
    img_indices = []
    
    for i, data_type in enumerate(os.listdir(root)):
        paths = []
        data_type_path = os.path.join(root, data_type)
        for img in os.listdir(data_type_path):
            if img[:2] == '._':
                continue
            img_path = os.path.join(data_type_path, img)
            paths.append(img_path)
            if data_type == 'rgb':
                index = img.split('.')[0]+'\n'
                img_indices.append(index)
        
        if data_type == 'rgb':
            img_paths['rgb'] = paths
        elif data_type == 'instance':
            img_paths['instance'] = paths
        elif data_type == 'rgb-2k':
            img_paths['rgb-2k'] = paths
        elif data_type == 'depth':
            img_paths['depth'] = paths
        elif data_type == 'class':
            img_paths['class'] = paths
        else:
            print(f'there is no instace type {data_type}')
       
    return img_paths, img_indices

def label_mapper(category_id):
    '''
    github : https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    '''
    # id to label object
    return labels.id2label[category_id].name
    

def get_train_test_index(img_indices, saved_dir):
    x_train, x_test = train_test_split(img_indices, test_size=0.2, random_state=42)
    
    print(f'train_indices : {len(x_train)}')
    print(f'test_indices : {len(x_test)}')
    # save with .txt files
    train_idxfile_path = save_anything(thing = x_train, saved_dir = saved_dir, file_name = 'train_indices', type = 'txt')
    test_idxfile_path = save_anything(thing = x_test, saved_dir = saved_dir, file_name = 'test_indices', type = 'txt')
    
    return train_idxfile_path, test_idxfile_path

def get_train_test_index_depth(img_indices, saved_dir, depth_threshold):
    x_train, x_test = train_test_split(img_indices, test_size=0.2, random_state=42)
    
    print(f'train_indices : {len(x_train)}')
    print(f'test_indices : {len(x_test)}')
    # save with .txt files
    train_idxfile_path = save_anything(thing = x_train, saved_dir = saved_dir, file_name = f'{depth_threshold}_train_indices', type = 'txt')
    test_idxfile_path = save_anything(thing = x_test, saved_dir = saved_dir, file_name = f'{depth_threshold}_test_indices', type = 'txt')
    
    return train_idxfile_path, test_idxfile_path


def get_total_index(img_paths, saved_dir):
    rgb_img_paths = img_paths['rgb']
    
    # append train indices
    total_indices =[]
    
    for i, path in enumerate(rgb_img_paths):
        index = path.split('/')[-1].split('.')[0]
        total_indices.append(f'{index}\n')
    print(f'total_indices : {len(total_indices)}')
    total_idxfile_path = save_anything(thing = total_indices, saved_dir = saved_dir, file_name = 'total_indices', type = 'txt')
    
    return total_idxfile_path


def get_Ndepth_index(meta_root, idxFile_path, saved_dir, filename, depth_threshold):
    # get train or test indices with idxFile_path
    with open(idxFile_path, 'r') as file:
        lines = file.readlines()
    
    objects_over_thr = 0
    obejcts_inner_thr = 0
    numImages_inner_thr = 0
    imgIndices_under_thr = []

    for line in lines:
        depths_under_thr = []
    
        line = line.strip('\n')
        json_path = os.path.join(meta_root, f'{line}.json')
        
        with open(json_path, 'r') as json_file:
            data = json.load(json_file) # dict_keys(['camera', 'instance', 'scene'])
        json_file.close()
            
        data_instance = data['instance'] # dict_keys(['bbox2d', 'bbox3d', 'class', 'occluded', 'truncated'])
        
        for i, id in enumerate(data_instance['bbox2d']):
            if data_instance['occluded'][id] >= 0.5:
                continue
            if data_instance['truncated'][id] >= 0.5:
                continue
            if data_instance['class'][id] not in [24, 25, 26, 27, 28, 29, 30, 31, 32, 33]: # 0 ~ 9
                continue
            # parse depth info
            data_bbox = data_instance['bbox2d'][id]         
            depth = (data_bbox['zmin'] + data_bbox['zmax']) / 2 # tmp = average depth(z_min, z_max)

            if depth_threshold is not None:
                if depth > depth_threshold:
                    objects_over_thr += 1
                    continue
            
            obejcts_inner_thr += 1
            depths_under_thr.append(depth)  

        if len(depths_under_thr) == 0:
            continue
        else:
            numImages_inner_thr += 1
            imgIndices_under_thr.append(f'{line}\n')
   
    # save indices under thr
    print(f"depth_thr : {depth_threshold}, objects_over_thr : {objects_over_thr}, obejcts_inner_thr : {obejcts_inner_thr}, # of images inner_thr: {numImages_inner_thr}")
    underThr_idxfile_path = save_anything(imgIndices_under_thr, filename, saved_dir=saved_dir, type = 'txt')
        
    return underThr_idxfile_path

    
def get_annotationJSON(root, idxFile_path, saved_dir, filename, depth_threshold = None):
   
    if 'train' in idxFile_path:
        pass
    
    # get train or test indices with idxFile_path
    with open(idxFile_path, 'r') as file:
        lines = file.readlines()
    
    # get .json file paths and parse them
    annotations = []
    images_data = []
    cateories = []
    buffer = []
    width = 570
    height = 455

    
    image_id = 0
    n = 0
    for line in lines:
        line = line.strip('\n')
        json_path = os.path.join(root, f'{line}.json')
        
        with open(json_path, 'r') as json_file:
            # dict_keys(['camera', 'instance', 'scene'])
            data = json.load(json_file) 
        json_file.close()

        data_camera = data['camera'] 
        data_instance = data['instance'] # dict_keys(['bbox2d', 'bbox3d', 'class', 'occluded', 'truncated'])
        data_scene = data['scene']
        
        if len(data_instance['bbox2d']) != 0:
            ignore_all = 0 # check if all objects in the image are ignored
            for i, id in enumerate(data_instance['bbox2d']):
                # ignore
                if data_instance['occluded'][id] >= 0.5:
                    continue
                if data_instance['truncated'][id] >= 0.5:
                    continue
                if data_instance['class'][id] not in [24, 25, 26, 27, 28, 29, 30, 31, 32, 33]: # 0 ~ 9
                    continue

                data_bbox = data_instance['bbox2d'][id]
                data_occlusion = data_instance['occluded'][id]
                category_id = data_instance['class'][id]
                category_str = label_mapper(category_id)
                category_id -= (24-1)  # 1 ~ 10

                bbox = [data_bbox['xmin'], data_bbox['ymin'], data_bbox['xmax'], data_bbox['ymax']]
                bbox_wh = [round(bbox[0], 4) * width, round(bbox[1], 4) * height, round(bbox[2]-bbox[0], 4) * width, round(bbox[3]-bbox[1], 4) * height]
                occlusion = 1 if data_occlusion >= 0.5 else 0
                depth = (data_bbox['zmin'] + data_bbox['zmax']) / 2 # tmp = average depth(z_min, z_max)
                
                # ignore
                if depth_threshold is not None:
                    if depth > depth_threshold:
                        continue
                image_id_real = int(line)
                

                annotations.append(
                    {
                    "id": n,
                    "category_str": category_str,
                    "occlusion": occlusion,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "depth": depth,
                    "bbox": bbox_wh,
                    "category_id": category_id,
                    }
                    )
                images_data.append(
                    [{
                    "file_name": f"RGB_L_{image_id_real:07d}",
                    "width": data_camera['intrinsic']['resx'],
                    "id": image_id,
                    "height": data_camera['intrinsic']['resy']
                    }]
                    )
                if category_id not in buffer:
                    buffer.append(category_id)
                    cateories.append(
                                    {
                                    "supercategory": category_str,
                                    "id": category_id,
                                    "name": category_str
                                    }
                                    )
                n += 1
                ignore_all += 1
                
            if ignore_all == 0:
                # get backgound annotation
                annotations.append(
                    {
                    "id": n,
                    "category_str": 'background',
                    "occlusion": 0,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "depth": 0,
                    "bbox": [0,0,0,0],
                    "category_id": 0, # background label = 0
                    }
                    )

                images_data.append(
                    [{
                    "file_name": f"RGB_L_{int(line):07d}",
                    "width": data_camera['intrinsic']['resx'],
                    "id": image_id,
                    "height": data_camera['intrinsic']['resy']
                    }]
                    )
                if 0 not in buffer:
                    buffer.append(0)
                    cateories.append(
                                    {
                                    "supercategory": 'background',
                                    "id": 0,
                                    "name": 'background'
                                    }
                                    )
                n += 1
                
        else:
            # get backgound annotation
            annotations.append(
                {
                "id": n,
                "category_str": 'background',
                "occlusion": 0,
                "iscrowd": 0,
                "image_id": image_id,
                "depth": 0,
                "bbox": [0,0,0,0],
                "category_id": 0, # background label = 0
                }
                )

            images_data.append(
                [{
                "file_name": f"RGB_L_{int(line):07d}",
                "width": data_camera['intrinsic']['resx'],
                "id": image_id,
                "height": data_camera['intrinsic']['resy']
                }]
                )
            if 0 not in buffer:
                buffer.append(0)
                cateories.append(
                                {
                                "supercategory": 'background',
                                "id": 0,
                                "name": 'background'
                                }
                                )
            n += 1   
        image_id += 1    
        
    json_data = {'annotations' : annotations,\
                 "images" : images_data,\
                 "categories" : cateories} 

    annotationJson_path = save_anything(json_data, filename, saved_dir=saved_dir, type = 'json')
        
    return annotationJson_path

            
        
    
def save_anything(thing, file_name, saved_dir=None, type = 'txt'):
    if saved_dir is not None:
        os.makedirs(saved_dir, exist_ok = True)
    
    with open(f'{os.path.join(saved_dir, file_name)}.{type}', 'w') as file:
        if type == 'txt':
            file.writelines(thing)
        elif type == 'json':
            json.dump(thing, file, indent=4)
                
    return f'{os.path.join(saved_dir, file_name)}.{type}'

    

def main(root_path):
    '''
    you can use this function for parse datasets in raw state
    '''
    
    # img root path
    img_root_path = os.path.join(root_path, os.listdir(root_path)[0])
    # meta data root path
    meta_root_path = os.path.join(root_path, os.listdir(root_path)[1])
    # saved dir path
    saved_dir = "./imagesets"
    
    # parse image paths 
    img_paths, img_indices = get_img_paths(img_root_path) # dictionary!! use img_paths.keys() for usage
    
    # where to save the train_test_splited indices 
    os.makedirs(saved_dir, exist_ok = True)
    # get train-test splited indices
    train_idxfile_path, test_idxfile_path = get_train_test_index(img_indices, saved_dir = saved_dir) # get train and test images indices
    # get total dataset indices
    total_idxfile_path = get_total_index(img_paths, saved_dir = saved_dir)
    
    ################################## if you need ####################################
    # # make .json files for train annotation and G.T annotation
    # test_idxfile_path = '/home/hscho/workspace/src/MLPD/src/synscapes_scripts/imagesets/test_indices.txt'
    # train_idxfile_path = '/home/hscho/workspace/src/MLPD/src/synscapes_scripts/imagesets/train_indices.txt'
    ####################################################################################

    trainJson_path = get_annotationJSON(root = meta_root_path, idxFile_path = train_idxfile_path, saved_dir = saved_dir, filename='synscape_train_annotation')
    print(f'trainJson_path : {trainJson_path}')

    testJson_path = get_annotationJSON(root = meta_root_path, idxFile_path = test_idxfile_path, saved_dir = saved_dir, filename='synscape_test_annotation')
    print(f'testJson_path : {testJson_path}')

def main_Ndepth(root_path, depth_threshold = None):
    '''
    you can use this function for parse datasets w.r.t depth 
    <input> 
    - root_path : index ID.txt file which has been parsed w.r.t depth 
    - depth_threshold : depth_threshold to ignore(we ignore depth more than depth_threshold)
    
    <output>
    none
    '''
    # img root path
    img_root_path = os.path.join(root_path, os.listdir(root_path)[0])
    # meta data root path
    meta_root_path = os.path.join(root_path, os.listdir(root_path)[1])
    # saved dir path
    saved_dir = "./imagesets"
    
    # parse image paths 
    img_paths, img_indices = get_img_paths(img_root_path) # dictionary!! use img_paths.keys() for usage
    
    # where to save the total indices 
    os.makedirs(saved_dir, exist_ok = True)
    
    # get total indices
    total_idxfile_path = get_total_index(img_paths, saved_dir = saved_dir)
    
    # get Image_indices under depth_threshold
    underThr_idxfile_path = get_Ndepth_index(meta_root=meta_root_path, idxFile_path=total_idxfile_path, saved_dir=saved_dir, filename=f'{depth_threshold}_total_indices', depth_threshold = depth_threshold)
    
    # get train and test images indices
    with open(underThr_idxfile_path, 'r') as file:
        img_indices_under_thr = file.readlines()
    
    train_idxfile_path, test_idxfile_path = get_train_test_index_depth(img_indices_under_thr, saved_dir = saved_dir, depth_threshold=depth_threshold) 
    
    testJson_path = get_annotationJSON(root = meta_root_path, idxFile_path = test_idxfile_path, saved_dir = saved_dir, filename=f'{depth_threshold}_synscape_test_annotation', depth_threshold = depth_threshold)

    
    
   
    
    
    

if __name__ == '__main__':
    root_path = '/home/hscho/workspace/src/MLPD/Synscapes'
    
    # first trial usage
    main(root_path)
    
    ## parse w.r.t depth threshold

    # if N depth
    main_Ndepth(root_path, depth_threshold = 6)

    # if depth is None, no depth_threshold applied
    # main_Ndepth(root_path, depth_threshold = None)
    
    



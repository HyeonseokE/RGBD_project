import json

import torch
import numpy as np

import argparse
from tqdm import tqdm
import wandb

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def parsed_result(gt_path, result, type='coco'):
    # YOLO: [cx, cy, w, h]
    # COCO: [x, y, w, h]
    with open(result) as f:
        data = json.load(f)
    
    with open(gt_path) as f:
        gt = json.load(f)

    gt = gt['annotations']
    data = data['annotation']
    
    all_ = set([aa['image_id'] for aa in gt])
    ids = set([aa['image_id'] for aa in data])
    
    
    results = []
    gt_data = []
    for id in tqdm(all_, desc="iterations for parsing results"):     
        parse_dict = {}
        gt_dict = {}
        if type.lower() == 'yolo':
            parse_dict['bbox'] = np.array([np.array([list(xywh2xyxy(aa['bbox'])) + [aa['depth'] / 1000] + [aa['category_id']]]) for aa in data if aa['image_id'] == id], dtype=np.float32)
        else:
            parse_dict['bbox'] = np.array([np.array([list(xywh2xyxy(aa['bbox'])) + [aa['depth']] + [aa['category_id']]]) for aa in data if aa['image_id'] == id], dtype=np.float32)

        gt_dict['bbox'] = np.array( [ np.array( [list(xywh2xyxy(aa['bbox'])) + [aa['depth'] / 1000] ])  for aa in gt if aa['image_id'] == id], dtype=np.float32)

        results.append(parse_dict)
        gt_data.append(gt_dict)
        
    return gt_data, results

def cxcywh2xywh(boxes):
    boxes = np.array(boxes)
    return np.concatenate([ (boxes[:, 2:] + boxes[:, :2]) / 2,
                    boxes[:, 2:] ], axis=1)
    
def xywh2xyxy(boxes):
    boxes = np.array(boxes)
    return np.concatenate([ boxes[0:2],
                    boxes[2:] + boxes[:2] ])

def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)

def percent_error(errors, gt_depth, rst_depth, DepthRng = [1, 2, 3, 4, 5, 6]):
    depth_error = abs((gt_depth - rst_depth) / gt_depth)
    errors[0].update(depth_error)
    for idx, rng in enumerate(DepthRng):
        if gt_depth <= rng :
            errors[idx+1].update(depth_error)
    return errors

def each_depth_precision(GT, results):
    wh = torch.tensor([1024, 2048, 1024, 2048])
    
    DepthRng = range(5, 15, 5)
    label2ind = {'person': 1, 'car': 2, 'train': 3, 'rider': 4,
                 'truck': 5, 'motorcycle': 6, 'bicycle': 7, 'bus': 8, 'background': 0}
    rev_label_map = {ind: label for label, ind in label2ind.items()}
    
    per_errors = [AverageMeter() for _ in range(len(DepthRng) + 1)] # Total + DepthRng
    
    Correct = 0
    Incorrect = 0
    
    In_C = [0] * len(DepthRng)
    Cor = [0] * len(DepthRng)
    
    true_bboxes_out = GT[:, :, :4]
    true_depths_out = GT[:, :, 4]
    
    if results.sum() == 0:
        if GT.sum() == 0:
            return Correct, Incorrect, Cor, In_C, None
        else:
            Incorrect = true_bboxes_out.shape[0]
            return Correct, Incorrect, Cor, In_C, None
    # if len(true_bboxes_out) == 1 :
    #     Incorrect = len(det_boxes)
    #     Correct = 0
        
    #     return Correct, Incorrect, Cor, In_C, None
    
    det_boxes = results[:, :, :4]
    det_depths = results[:, :, 4]
    det_labels = results[:, :, 5].flatten().astype(np.int32)
    
    det_labels = [rev_label_map[l] for l in det_labels.tolist()]
    if det_labels == ['background'] :
        return Correct, Incorrect, Cor, In_C, None

    # Decode class integer labels
    # det_labels = [rev_label_map[l] for l in det_labels[0]]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    # if det_labels == ['background'] :
    #     return Correct, Incorrect, InC_1, InC_2, InC_3, InC_4, InC_5, InC_6, Cor_1, Cor_2, Cor_3, Cor_4, Cor_5, Cor_6
    try:
        overlap = find_jaccard_overlap(torch.tensor(det_boxes).squeeze(1) / wh,
                                    torch.tensor(true_bboxes_out[:]).squeeze(1) / wh)
    except:
        import pdb;pdb.set_trace()
        
    overlap_score, max_index = overlap.max(dim=1)
    
    for i in range(len(det_boxes)) :
        # true_depths_out[max_index[i]]*0.2
        if overlap_score[i] < 0.5 :
          continue
       
        ## percent error
        per_errors = percent_error(per_errors, true_depths_out[max_index[i]], det_depths[i], DepthRng)
        
        ## Precision
        if abs(det_depths[i] - true_depths_out[max_index[i]]) < true_depths_out[max_index[i]] * 0.1 :
            Correct = Correct + 1
            
            for idx, rng in enumerate(DepthRng):
                if true_depths_out[max_index[i]] <= rng :
                    Cor[idx] += 1
        else :
            Incorrect = Incorrect +1
            
            for idx, rng in enumerate(DepthRng):
                if true_depths_out[max_index[i]] <= rng :
                    In_C[idx] += 1
        ##
        
    ## Percent error
    per_errors = [per_e.avg for per_e in per_errors]

    return Correct, Incorrect, Cor, In_C, per_errors

# def Depth_Average_Precision(GT, results, logger):
def Depth_Average_Precision(GT, results, result_dir):
    DepthRng = range(5, 15, 5)
    
    PerErrors = [AverageMeter() for _ in range(len(DepthRng) + 1)] # Total + DepthRng
    
    TotalTrue = 0
    TotalFalse = 0
    
    EachTrue = [0] * len(DepthRng)
    EachFalse = [0] * len(DepthRng)
    
    for gt, result in tqdm(zip(GT, results), desc="iterating images..."):
        gt = np.array(gt['bbox'])
        result = np.array(result['bbox'])
        
        ImageByTrue, ImageByFalse, ImageByEachTrue, ImageByEachFalse, ImageByPerErrors = each_depth_precision(gt, result)
        
        # Percent errors
        if ImageByPerErrors is not None:
            for ii, error in enumerate(ImageByPerErrors):
                PerErrors[ii].update(error)
        
        # Precision
        TotalTrue += ImageByTrue
        TotalFalse += ImageByFalse
        
        for ii in range(len(EachFalse)):
            EachFalse[ii] += ImageByEachFalse[ii]
            EachTrue[ii] += ImageByEachTrue[ii]
            
    # logger.info("############### Depth Average Precision ################")
    # logger.info(f"True: {TotalTrue}")
    # logger.info(f"False: {TotalFalse}")
    # logger.info(f"Precision: {(TotalTrue / (TotalFalse + TotalTrue)) * 100:.2f}" )
    # for ii in range(len(EachFalse)):
    #     try:
    #         print(f'Wrong_{DepthRng[ii]}m : {EachFalse[ii]} (F) / {EachTrue[ii]} (T) / {EachTrue[ii] / ( EachTrue[ii] + EachFalse[ii]) * 100}') 
    #     except:
    #         print("divided by zero")

    # logger.info("################### Percent Error ########################")
    # logger.info(f'Total Percent Error: {PerErrors[0].avg[0] }')
    # for ii in range(len(PerErrors) - 1):
    #     logger.info(f'PecentError up to {DepthRng[ii]}m: {PerErrors[ii+1].avg[0]}') 
    
    ########################## save to txt file ##############################
    with open(f'{result_dir}/eval_results.txt', 'a') as file:
        file.write("############### Depth Average Precision ################"+'\n')
        file.write(f"True: {TotalTrue}"+'\n')
        file.write(f"False: {TotalFalse}"+'\n')
        file.write(f"Precision: {(TotalTrue / (TotalFalse + TotalTrue)) * 100:.2f}"+'\n' )
        for ii in range(len(EachFalse)):
            try:
                file.write(f'Wrong_{DepthRng[ii]}m : {EachFalse[ii]} (F) / {EachTrue[ii]} (T) / {EachTrue[ii] / ( EachTrue[ii] + EachFalse[ii]) * 100}'+'\n') 
            except:
                file.write("divided by zero"+'\n')

        file.write("################### Percent Error ########################"+'\n')
        file.write(f'Total Percent Error: {PerErrors[0].avg}'+'\n')
        for ii in range(len(PerErrors) - 1):
            file.write(f'PecentError up to {DepthRng[ii]}m: {PerErrors[ii+1].avg}'+'\n')        
    file.close()
    print(f'submission result saved : {result_dir}/eval_results.txt')
    ############################################################################
    
    #########################  print log  #################################
    log_dict={}
    print("############### Depth Average Precision ################")
    print(f"True: {TotalTrue}")
    print(f"False: {TotalFalse}")
    print(f"Precision: {(TotalTrue / (TotalFalse + TotalTrue)) * 100:.2f}" )
    log_dict['depth_Precision'] = (TotalTrue / (TotalFalse + TotalTrue)) * 100
    
    for ii in range(len(EachFalse)):
        try:
            print(f'Wrong_{DepthRng[ii]}m : {EachFalse[ii]} (F) / {EachTrue[ii]} (T) / {EachTrue[ii] / ( EachTrue[ii] + EachFalse[ii]) * 100}') 
        except:
            print("divided by zero")

    print("################### Percent Error ########################")
    print(f'Total Percent Error: {PerErrors[0].avg}')
    log_dict['Total_Percent_Error'] = PerErrors[0].avg
    
    for ii in range(len(PerErrors) - 1):
        print(f'PecentError up to {DepthRng[ii]}m: {PerErrors[ii+1].avg}')
        log_dict[f'PecentError_{DepthRng[ii]}m'] = PerErrors[ii+1].avg
    ###########################################################################
    try:
        wandb.log(log_dict)
    except:
        pass
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rst_file', default=None, type=str, help='Model Result')
    parser.add_argument('--GT', default=None, type=str, help='GT json file')
    parser.add_argument('--type', default='YOLO', type=str, help='YOLO or COCO')
    args = parser.parse_args()
    
    GT, results = parsed_result(args.GT, args.rst_file,  type=args.type)
    
    Depth_Average_Precision(GT=GT, results=results)

import os
from os.path import join as opj
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple
import argparse
import config
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import KAISTPed
from utils.transforms import FusionDeadZone
from utils.evaluation_script import evaluate, evaluate_potenit
from vis import visualize

from model import SSD300
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image, ImageDraw, ImageFont

import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from torchvision.utils import save_image



def val_epoch(model: SSD300, dataloader: DataLoader, input_size: Tuple, min_score: float = 0.1) -> Dict:
    """Validate the model during an epoch

    Parameters
    ----------
    model: SSD300
        SSD300 model for multispectral pedestrian detection defined by src/model.py
    dataloader: torch.utils.data.dataloader
        Dataloader instance to feed training data(images, labels, etc) for KAISTPed dataset
    input_size: Tuple
        A tuple of (height, width) for input image to restore bounding box from the raw prediction
    min_score: float
        Detection score threshold, i.e. low-confidence detections(< min_score) will be discarded

    Returns
    -------
    Dict
        A Dict of numpy arrays (K x 5: xywh + score) for given image_id key
    """

    model.eval()

    height, width = input_size
    xyxy_scaler_np = np.array([[width, height, width, height]], dtype=np.float32) 
    
    device = next(model.parameters()).device # next() : retrieve next item(weight) # .device : retrieve parameters device name 
    results = dict()
    with torch.no_grad():
        for i, blob in enumerate(tqdm(dataloader, desc='Evaluating')): # "desc" : 진행바 앞에 텍스트 출력
            
            image_vis, image_depth, boxes, labels, depths, indices = blob

            image_vis = image_vis.to(device)
            image_depth = image_depth.to(device)

            # Forward prop.
            predicted_locs, predicted_scores, predicted_depths = model(image_vis, image_depth)
            # Detect objects in SSD output
            
            detections = model.detect_objects(predicted_locs, predicted_scores, predicted_depths,
                                                     min_score=min_score, max_overlap=0.425, top_k=200)

            det_boxes_batch, det_labels_batch, det_labels_depth, det_scores_batch = detections[:4]

            for boxes_t, labels_t, scores_t, image_id in zip(det_boxes_batch, det_labels_batch, det_scores_batch, indices):
                boxes_np = boxes_t.cpu().numpy().reshape(-1, 4)
                scores_np = scores_t.cpu().numpy().mean(axis=1).reshape(-1, 1)

                # TODO(sohwang): check if labels are required
                # labels_np = labels_t.cpu().numpy().reshape(-1, 1)
                xyxy_np = boxes_np * xyxy_scaler_np
                xywh_np = xyxy_np
                xywh_np[:, 2] -= xywh_np[:, 0]
                xywh_np[:, 3] -= xywh_np[:, 1]
                results[image_id.item() + 1] = np.hstack([xywh_np, scores_np])
    return results

def visualize_RGB(transform, fnt, result, image, image_id, mode):

    vis_ = (image - image.min()) / (image.max() - image.min()) # [0, 1] 범위로 정규화
    vis_ = (vis_ * 255).type(torch.uint8).squeeze() # uint8, 0 ~ 255로 변환
    vis_ = transform(vis_)
    draw1 = ImageDraw.Draw(vis_)

    for i, info in enumerate(result):
        box = info[:4]
        score = info[4]
        depth = info[5]
        draw1.rectangle(box.tolist(), outline="red", width=2)
        draw1.text(((box[0]+box[2])/2, (box[1]+box[3])/2), f"{depth}", font=fnt, fill ="red")

    ToTensor()(vis_)
    vis_.save(f'result({mode})_visualization_R/model_inference_R_after_treat_resize_{image_id.item() + 1}.jpg')

        #################################################################################

def visualize_Heatmap(transform, image_depth, image_id, mode):

    depth_ = transform(image_depth.squeeze())
    draw2 = ImageDraw.Draw(depth_)
            
    depth_heatmap =  (np.array(image_depth.cpu().squeeze())*255).astype("uint8")
    depth_heatmap_equalized1 = cv2.equalizeHist(depth_heatmap)
    
        # np.array(depth_heatmap_equalized.squeeze().cpu())
    sns.heatmap(np.array(depth_heatmap_equalized1), 
                cmap='RdYlGn_r', 
                cbar = False, 
                xticklabels = False, 
                yticklabels = False) # depthmap heatmap으로 시각화
    
    plt.savefig(f"result({mode})_visualization_Heat/depth_heatmap_{image_id.item() + 1}.jpg", dpi=400) # heatmap 저장


def visualize_FoVImage(transform, fnt, color, result):
    pass

def val_epoch_potenit(model: SSD300, dataloader: DataLoader, input_size: Tuple, min_score: float = 0.1, isVisualize: bool = False)  -> Dict:
    """Validate the model during an epoch

    Parameters
    ----------
    model: SSD300
        SSD300 model for multispectral pedestrian detection defined by src/model.py
    dataloader: torch.utils.data.dataloader
        Dataloader instance to feed training data(images, labels, etc) for KAISTPed dataset
    input_size: Tuple
        A tuple of (height, width) for input image to restore bounding box from the raw prediction
    min_score: float
        Detection score threshold, i.e. low-confidence detections(< min_score) will be discarded

    Returns
    -------
    Dict
        A Dict of numpy arrays (K x 5: xywh + score) for given image_id key
    """
    print(f"inference start with {dataloader.dataset.mode} dataset")
    model.eval()

    height, width = input_size
    xyxy_scaler_np = np.array([[width, height, width, height]], dtype=np.float32) 
    transform_ = ToPILImage()  
    fnt_ = ImageFont.load_default()
    
    device = next(model.parameters()).device # next() : retrieve next item(weight) # .device : retrieve parameters device name 
    results = dict()
    with torch.no_grad():
        for i, blob in enumerate(tqdm(dataloader, desc='Evaluating')): # "desc" : 진행바 앞에 텍스트 출력
            
            image_vis, image_depth, boxes, labels, depths, indices = blob
            ############################################################# 시각화 ##########
            # vis_ = ToPILImage()(image_vis[0])
            # depth_ = ToPILImage()(image_depth[0])
            # draw1 = ImageDraw.Draw(vis_)
            # draw2 = ImageDraw.Draw(depth_)
            
            # fnt = ImageFont.load_default()
            # color = (0,255,0)
            # # 시각화(after)
            # # (xmin, ymin, xmax, ymax)
            # for i, box in enumerate(boxes[0]):
            #     draw1.rectangle(list(box), outline="red", width=2)
            #     draw1.text(((box[0]+box[2])/2, (box[1]+box[3])/2), f"{depths[0][i]}", font=fnt, fill ="red")
            #     draw2.rectangle(list(box), outline="red", width=2)
            #     draw2.text(((box[0]+box[2])/2, (box[1]+box[3])/2), f"{depths[0][i]}", font=fnt, fill ="red")
            
            # vis_.save(f'before_inference_R.jpg')
            # depth_.save(f'before_inference_D.jpg')
            #################################################################################
            
            image_vis = image_vis.to(device)
            image_depth = image_depth.to(device)

            # Forward prop.  predicted_locs = (gcx, gcy, cw, ch)
            predicted_locs, predicted_scores, predicted_depths = model(image_vis, image_depth)
            
            # Detect objects in SSD output
            
            # detections = model.detect_objects(predicted_locs, predicted_scores, predicted_depths,
            #                                          min_score=min_score, max_overlap=0.425, top_k=200)
            detections = model.detect_objects(predicted_locs, predicted_scores, predicted_depths,
                                                     min_score=0.45, max_overlap=0.55, top_k=100)
            det_boxes_batch, det_labels_batch, det_scores_batch, det_depth_batch = detections[:4]
            ############################################################# 시각화 ##########
            # vis_ = ToPILImage()(image_vis[0])
            # depth_ = ToPILImage()(image_depth[0])
            # draw1 = ImageDraw.Draw(vis_)
            # draw2 = ImageDraw.Draw(depth_)
            
            # fnt = ImageFont.load_default()
            # color = (0,255,0)
            # # 시각화(after)
            # # (xmin, ymin, xmax, ymax)
            # for i, box in enumerate(det_boxes_batch[0]):
            #     draw1.rectangle(list(box), outline="red", width=2)
            #     draw1.text(((box[0]+box[2])/2, (box[1]+box[3])/2), f"{det_depth_batch[0][i]}", font=fnt, fill ="red")
            #     draw2.rectangle(list(box), outline="red", width=2)
            #     draw2.text(((box[0]+box[2])/2, (box[1]+box[3])/2), f"{det_depth_batch[0][i]}", font=fnt, fill ="red")
            
            # vis_.save(f'model_inference_R_before_treat_resize.jpg')
            # depth_.save(f'model_inference_D_before_treat_resize.jpg')
            
            #################################################################################

            
            for boxes_t, labels_t, scores_t, depths_t, image_id in zip(det_boxes_batch, det_labels_batch, det_scores_batch, det_depth_batch, indices):
                boxes_np = boxes_t.cpu().numpy().reshape(-1, 4)
                scores_np = scores_t.cpu().numpy().mean(axis=1).reshape(-1, 1)
                depths_np = depths_t.cpu().numpy().reshape(-1, 1)

                # TODO(sohwang): check if labels are required
                # labels_np = labels_t.cpu().numpy().reshape(-1, 1)
                xyxy_np = boxes_np * xyxy_scaler_np
                # xywh_np = xyxy_np
                # xywh_np[:, 2] -= xywh_np[:, 0]
                # xywh_np[:, 3] -= xywh_np[:, 1]
                # results[image_id.item() + 1] = np.hstack([xywh_np, scores_np])image, image_id
                results[image_id.item() + 1] = np.hstack([xyxy_np, scores_np, depths_np])
            if isVisualize == True:
                visualize_RGB(transform=transform_, fnt = fnt_, result=results[image_id.item() + 1], image=image_vis,image_id= image_id, mode = dataloader.dataset.mode)
                visualize_Heatmap(transform=transform_, image_depth = image_depth, image_id = image_id, mode = dataloader.dataset.mode)
                # visualize_FoVImage(transform, fnt, color, result, image, image_id, mode = dataloader.dataset.mode)
                

    return results



# def run_inference(model_path: str, fdz_case: str) -> Dict:
def run_inference(model_path: str, num_workers, isVisualize:bool) -> Dict:
    """Load model and run inference

    Load pretrained model and run inference on KAIST dataset with FDZ setting.

    Parameters
    ----------
    model_path: str
        Full path of pytorch model
    fdz_case: str
        Fusion dead zone case defined in utils/transforms.py:FusionDeadZone

    Returns
    -------
    Dict
        A Dict of numpy arrays (K x 5: xywh + score) for given image_id key
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path)['model']
    model = model.to(device)

    # model = nn.DataParallel(model)

    input_size = config.test.input_size

    # Load dataloader for Fusion Dead Zone experiment
    # FDZ = [FusionDeadZone(config.FDZ_case[fdz_case], tuple(input_size))]
    # config.test.img_transform.add(FDZ)
    print(f"evaluation with num_workers of {num_workers}")
        
    args = config.args
    # batch_size = config.test.batch_size * torch.cuda.device_count()
    batch_size = 1
    if isVisualize:
        batch_size = 1
    test_dataset = KAISTPed(args, condition="test")
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              collate_fn=test_dataset.collate_fn,
                                              pin_memory=True)
    
    results = val_epoch_potenit(model, test_loader, input_size, isVisualize = isVisualize)
    return results

def run_inference_train(model_path: str, num_workers, isVisualize:bool) -> Dict:
    """Load model and run inference

    Load pretrained model and run inference on KAIST dataset with FDZ setting.

    Parameters
    ----------
    model_path: str
        Full path of pytorch model
    fdz_case: str
        Fusion dead zone case defined in utils/transforms.py:FusionDeadZone

    Returns
    -------
    Dict
        A Dict of numpy arrays (K x 5: xywh + score) for given image_id key
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path)['model'] 
    model = model.to(device)

    # model = nn.DataParallel(model)

    input_size = config.test.input_size #

    # Load dataloader for Fusion Dead Zone experiment
    # FDZ = [FusionDeadZone(config.FDZ_case[fdz_case], tuple(input_size))]
    # config.test.img_transform.add(FDZ)

    args = config.args
    # batch_size = config.test.batch_size * torch.cuda.device_count()
    batch_size = 1
    train_dataset = KAISTPed(args, condition="train")
    
    if isVisualize:
        batch_size = 1
        
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              collate_fn=train_dataset.collate_fn,
                                              pin_memory=True)

    results = val_epoch_potenit(model, train_loader, input_size, isVisualize = isVisualize)
    return results

def save_results(results: Dict, result_filename: str):
    """Save detections

    Write a result file (.txt) for detection results.
    The results are saved in the order of image index.

    Parameters
    ----------
    results: Dict
        Detection results for each image_id: {image_id: box_xywh + score}
    result_filename: str
        Full path of result file name

    """

    if not result_filename.endswith('.txt'):
        result_filename += '.txt'

    with open(result_filename, 'w') as f:
        for image_id, detections in sorted(results.items(), key=lambda x: x[0]):
            for x, y, w, h, score in detections:
                f.write(f'{image_id},{x:.4f},{y:.4f},{w:.4f},{h:.4f},{score:.8f}\n')
                
def save_results_potenit(results: Dict, result_filename: str):
    """Save detections

    Write a result file (.txt) for detection results.
    The results are saved in the order of image index.

    Parameters
    ----------
    results: Dict
        Detection results for each image_id: {image_id: box_xywh + score}
    result_filename: str
        Full path of result file name

    """
    if not result_filename.endswith('.txt'):
        result_filename += '.txt'

    with open(result_filename, 'w') as f:
        for image_id, detections in sorted(results.items(), key=lambda x: x[0]):
            for x_min, y_min, x_max, y_max, score, depth in detections:
                # np.hstack([xyxy_np, scores_np, depths_np])
                f.write(f'{image_id},{x_min:.4f},{y_min:.4f},{x_max:.4f},{y_max:.4f},{score:.8f},{depth:.4f}\n')

def visualization_all():
    pass


def main_train(arguments):

    # fdz_case = arguments.FDZ.lower() # 'original'
    model_path = Path(arguments.model_path).stem.replace('.', '_') # 'checkpoint_ssd300_pth

    # Run inference to get detection results
    os.makedirs(arguments.result_dir, exist_ok=True) # '../result'
    result_filename = opj(arguments.result_dir,  f'{model_path}_TEST_det') # '../result/original_checkpoint_ssd300_pth_TEST_det'

    # Run inference
    results = run_inference_train(arguments.model_path, arguments.num_workers, isVisualize = arguments.isVisualize)
    # import pickle
    # with open('run_inference.pickle', 'wb') as f:
    #     pickle.dump(results, f)
        
    import pickle 
    with open('run_inference.pickle', 'rb') as f:
        results = pickle.load(f)
        
    # Save results
    save_results_potenit(results, result_filename)

    # Eval results
    phase = "Multispectral"
    # 'MDPI_test_solution.json'  # ../result/original_checkpoint_ssd300_pth_TEST_det
    # import pdb;pdb.set_trace()
    evaluate_potenit(config.PATH.JSON_GT_FILE, result_filename + '.txt', phase) 
    
    # Visualizing
    # if arguments.vis:
    #     vis_dir = opj(arguments.result_dir, 'vis', model_path, fdz_case)
    #     os.makedirs(vis_dir, exist_ok=True)
    #     visualize(result_filename + '.txt', vis_dir, fdz_case)

def main_test(arguments):
    
    # fdz_case = arguments.FDZ.lower() # 'original'
    model_path = Path(arguments.model_path).stem.replace('.', '_') # 'checkpoint_ssd300_pth

    # Run inference to get detection results
    os.makedirs(arguments.result_dir, exist_ok=True) # '../result'
    # result_filename = opj(arguments.result_dir,  f'{fdz_case}_{model_path}_TEST_det') # '../result/original_checkpoint_ssd300_pth_TEST_det'
    result_filename = opj(arguments.result_dir,  f'{model_path}_TEST_det')
    
    # Run inference
    # results = run_inference_train(arguments.model_path, fdz_case, isVisualize = True)
    # results = run_inference(arguments.model_path, arguments.num_workers, isVisualize=arguments.isVisualize)
    
    ############### savinig with pickle ################
    # import pickle
    # with open('run_inference.pickle', 'wb') as f:
    #     pickle.dump(results, f)
    
    #####################################################
        
    import pickle 
    with open('run_inference.pickle', 'rb') as f:
        results = pickle.load(f)
        
    # Save results
    save_results_potenit(results, result_filename)

    # Eval results
    phase = "Multispectral"
    # 'MDPI_test_solution.json'  
    # result saved path = ../result/checkpoint_ssd300_pth_TEST_det
    # import pdb;pdb.set_trace()
    evaluate_potenit(config.PATH.JSON_GT_FILE, result_filename + '.txt', phase) 
    
    # import pdb;pdb.set_trace()
    # Visualizing
    # if arguments.vis:
    #     vis_dir = opj(arguments.result_dir, 'vis', model_path, fdz_case)
    #     os.makedirs(vis_dir, exist_ok=True)
    #     visualize(result_filename + '.txt', vis_dir, fdz_case)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--model_path', type=str, default='/home/hscho/workspace/ssd2.5d/MLPD-Multi-Label-Pedestrian-Detection/src/jobs/2024-07-05_14h54m_/checkpoint_ssd300.pth.tar039',
                         help='Pretrained model for evaluation.')
    parser.add_argument('--result_dir', type=str, default='../result',
                        help='Save result directory')
    parser.add_argument('--vis', action='store_true', 
                        help='Visualizing the results')
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--isVisualize', type=bool, default = False)
    parser.add_argument('--mode', type=str, default='train')
    
    arguments = parser.parse_args()
    

    if arguments.mode == "train":
        main_train(arguments)
    else:
        main_test(arguments)

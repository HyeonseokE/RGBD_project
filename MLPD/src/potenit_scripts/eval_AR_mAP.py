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
from utils.evaluation_script import evaluate, evaluate_potenit, eval_potenit
from utils.coco import COCO
from utils.cocoeval import COCOeval
from vis import visualize

from model import SSD300
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image, ImageDraw, ImageFont

import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from torchvision.utils import save_image

# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval

import config
from utils import depth_eval
import json



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
    xyxy_scaler_np = np.array([[570, 455, 570, 455]], dtype=np.float32)
    transform_ = ToPILImage()  
    fnt_ = ImageFont.load_default()
    
    device = next(model.parameters()).device # next() : retrieve next item(weight) # .device : retrieve parameters device name 
    results = dict()
    with torch.no_grad():
        for i, blob in enumerate(tqdm(dataloader, desc='Evaluating')): # "desc" : 진행바 앞에 텍스트 출력
            
            image_vis, image_depth, boxes, labels, depths, indices = blob
            
            image_vis = image_vis.to(device)
            image_depth = image_depth.to(device)

            # Forward prop.  predicted_locs = (gcx, gcy, cw, ch)
            predicted_locs, predicted_scores, predicted_depths = model(image_vis, image_depth)
            
            # Detect objects in SSD output
            detections = model.detect_objects(predicted_locs, predicted_scores, predicted_depths,
                                                     min_score=0.45, max_overlap=0.55, top_k=100)
            # detections = model.detect_objects(predicted_locs, predicted_scores, predicted_depths,
            #                                          min_score=0.1, max_overlap=0.45, top_k=100)
            det_boxes_batch, det_labels_batch, det_scores_batch, det_depth_batch = detections[:4]
            
            for boxes_t, labels_t, scores_t, depths_t, image_id in zip(det_boxes_batch, det_labels_batch, det_scores_batch, det_depth_batch, indices):
                boxes_np = boxes_t.cpu().numpy().reshape(-1, 4)
                scores_np = scores_t.cpu().numpy().mean(axis=1).reshape(-1, 1)
                
                ####################### Notice ###############################
                # depth_eval 평가는 meter단위로 맞춰서 진행함.
                ##############################################################
                # depths_np = (depths_t.cpu().numpy().reshape(-1, 1))/1000 # if milimeter
                depths_np = depths_t.cpu().numpy().reshape(-1, 1)      # if meter
                ##############################################################
                labels_t = labels_t.cpu().numpy().reshape(-1, 1)

                xyxy_np = boxes_np * xyxy_scaler_np
                xywh_np = xyxy_np
                xywh_np[:, 2] -= xywh_np[:, 0]
                xywh_np[:, 3] -= xywh_np[:, 1]
                
                results[image_id.item()] = np.hstack([xywh_np, scores_np, depths_np, labels_t])
                
            if isVisualize == True:
                visualize_RGB(transform=transform_, fnt = fnt_, result=results[image_id.item()], image=image_vis,image_id= image_id, mode = dataloader.dataset.mode)
                # visualize_Heatmap(transform=transform_, image_depth = image_depth, image_id = image_id, mode = dataloader.dataset.mode)
                # visualize_FoVImage(transform, fnt, color, result, image, image_id, mode = dataloader.dataset.mode)
            
    return results

def visualize_RGB(transform, fnt, result, image, image_id, mode):
    import time
    tic = time.time()
    vis_ = (image - image.min()) / (image.max() - image.min()) # [0, 1] 범위로 정규화
    vis_ = (vis_ * 255).type(torch.uint8).squeeze() # uint8, 0 ~ 255로 변환
    vis_ = transform(vis_)
    draw1 = ImageDraw.Draw(vis_)

    for i, info in enumerate(result):
        box = info[:4]
        bbox = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
        score = info[4]
        depth = info[5]
        draw1.rectangle(bbox, width=2)
        draw1.text(((bbox[2])/2, (bbox[3])/2), f"{depth}", font=fnt, fill ="red")

    ToTensor()(vis_)
    vis_.save(f'result({mode})_visualization_R/model_inference_R_after_treat_resize_{image_id.item():06d}.jpg')
    print(f"{time.time() - tic}s")
    #################################################################################

def visualize_Heatmap(transform, image_depth, image_id, mode):
    import time
    tic = time.time()
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
    
    plt.savefig(f"result({mode})_visualization_Heat/depth_heatmap_{image_id.item():06d}.jpg", dpi=400) # heatmap 저장
    print(f"{time.time() - tic}s")

def visualize_FoVImage(transform, fnt, color, result):
    pass

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
    if not result_filename.endswith('.json'):
        result_filename += '.json'
    
    annotations = list()
    for i in range(len(results)):
        result = results[i]
        if result[:,:4].sum() == 0: # 빈예측 [] 일 때,
            annotation = {
                            'image_id': i,
                            'category_id': 0,
                            'bbox':[0, 0, 0, 0],
                            'area':0,
                            'score':0.0,
                            'depth':0.0 
                            }
            annotations.append(annotation)
        else:
            for ann in result:
                bbox = ann[:4]
                score = ann[4]
                depth = ann[5]
                label = ann[6]
                annotation = {
                            'image_id': i,
                            'category_id': label.item(),
                            'bbox':bbox.tolist(),
                            'area':0,
                            'score':score.item(),
                            'depth':depth.item() 
                            }
                annotations.append(annotation)
    annots = {'annotation':annotations}
           
    with open(result_filename, 'w') as f:
        json.dump(annots, f, indent=4)
    print(f"Dt result json path : {result_filename}")
    
    return result_filename
        
        
                
    
                
                
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

    # Load dataloader 
    print(f"evaluation with num_workers of {num_workers}")
    args = config.args
    batch_size = 1

    test_dataset = KAISTPed(args, condition="test")
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              collate_fn=test_dataset.collate_fn,
                                              pin_memory=True)
    
    results = val_epoch_potenit(model, test_loader, input_size, isVisualize = isVisualize)
    return results



def main_train():
    model_path = Path(arguments.model_path).stem.replace('.', '_') # 'checkpoint_ssd300_pth
    print(model_path)
    # Run inference to get detection results
    os.makedirs(arguments.result_dir, exist_ok=True) # '../result'
    result_filename = opj(arguments.result_dir,  f'{model_path}_TEST_det') # '../result/original_checkpoint_ssd300_pth_TEST_det'

    # Run inference
    results = run_inference_train(arguments.model_path, arguments.num_workers, isVisualize = arguments.isVisualize)
    ############### savinig with pickle ################
    import pickle
    with open('run_inference_train.pickle', 'wb') as f:
        pickle.dump(results, f)
    #####################################################
        
    import pickle 
    with open('run_inference_train.pickle', 'rb') as f:
        results = pickle.load(f)
        
    # Save results
    user_submission_file = save_results_potenit(results, result_filename)
    test_annotation_file = config.PATH.JSON_GT_FILE
    save_results_potenit(results, result_filename)

    # Eval results
    # (1) load dataset and results
    # test_annotation_file로 부터 G.T를 읽어드림 # potenitGt.anns
    potenitGt = COCO(test_annotation_file) # 'MDPI_test_solution.json'
    
    # detection result file로 부터 검출결과를 읽어드림. # potenitDt.anns
    potenitDt = potenitGt.loadRes(user_submission_file) # 'jobs/2024-07-03_05h23m_/Epoch040_test_det.txt'
    
    imgIds = sorted(potenitGt.getImgIds()) # 정렬된 image_id들
    
    coco_eval = COCOeval(potenitGt, potenitDt, 'bbox')
    
    # 각종 파라미터 설정
    coco_eval.params.imgIds = imgIds
    # params.catIds 설정하면 class 별 mAP 출력해주는듯

    # 수치 뽑는 3종셋트
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def main_test():
    try:
        epoch = int(arguments.model_path.split('.')[-1].split('0')[-1])
    except:
        import pdb;pdb.set_trace()
        
    model_path = Path(arguments.model_path).stem.replace('.', '_') # 'checkpoint_ssd300_pth
    print(model_path)

    # Run inference to get detection results
    os.makedirs(arguments.result_dir, exist_ok=True) # '../result'
    result_filename = os.path.join(arguments.result_dir,  f'{model_path}_TEST_det')

    # Run inference
    results = run_inference(arguments.model_path, arguments.num_workers, isVisualize=arguments.isVisualize)

    ############### savinig with pickle ################
    # import pickle
    # with open('run_inference.pickle', 'wb') as f:
    #     pickle.dump(results, f)
    #####################################################
        
    # import pickle 
    # with open('run_inference.pickle', 'rb') as f:
    #     results = pickle.load(f)
        
    # Save results
    user_submission_file = save_results_potenit(results, result_filename)
    test_annotation_file = config.PATH.JSON_GT_FILE

    unannotated_id = []
    
    # gt annotation을 불러옴
    potenitGt = COCO(test_annotation_file, unannotated_id) # 'MDPI_test_solution.json'
    # detection result file로 부터 검출결과를 읽어드림. # potenitDt.anns
    potenitDt = potenitGt.loadRes(user_submission_file, unannotated_id) # 'jobs/2024-07-03_05h23m_/Epoch040_test_det.txt'
        
    imgIds = sorted(potenitGt.getImgIds()) # 정렬된 image_id들
    
    coco_eval = COCOeval(potenitGt, potenitDt, 'bbox', arguments.result_dir, epoch)
    
    # 각종 파라미터 설정
    coco_eval.params.imgIds = imgIds
    # params.catIds 설정하면 class 별 mAP 출력해주는듯

    # 수치 뽑는 3종셋트
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # AR50
    iou_thr = 0.5
    iou_thr_ids = list(coco_eval.params.iouThrs).index(iou_thr)
    ar50 = coco_eval.eval['recall'][iou_thr_ids, :, 0, 2]
    ar50 = ar50[ar50 > -1].mean()
    log_path = os.path.join(arguments.result_dir, 'eval_results.txt')
    with open(log_path, 'a') as file:
        file.write(f'AR50 : {ar50:.2f}')
    file.close()
    print(f'AR50 : {ar50:.2f}')
    
    # depth 산출
    GT, results = depth_eval.parsed_result(test_annotation_file, user_submission_file, type='COCO')
    depth_eval.Depth_Average_Precision(GT=GT, results=results, result_dir=arguments.result_dir)
    
    
    

def test_during_train(model_path, result_dir, num_workers, epoch, isVisualize):

    print(model_path)

    # Run inference
    results = run_inference(model_path, num_workers, isVisualize=isVisualize)
    
    # checkpoint file name
    model_path = Path(model_path).stem.replace('.', '_') # 'checkpoint_ssd300_pth
    
    # Run inference to get detection results
    os.makedirs(result_dir, exist_ok=True) # '../result'
    result_filename = os.path.join(result_dir,  f'{model_path}_TEST_det')

    ############### savinig with pickle ################
    import pickle
    with open('run_inference.pickle', 'wb') as f:
        pickle.dump(results, f)
    #####################################################
        
    import pickle 
    with open('run_inference.pickle', 'rb') as f:
        results = pickle.load(f)
        
    # Save results
    user_submission_file = save_results_potenit(results, result_filename)
    test_annotation_file = config.PATH.JSON_GT_FILE

    unannotated_id = []
    
    # gt annotation을 불러옴
    potenitGt = COCO(test_annotation_file, unannotated_id) # 'MDPI_test_solution.json'
    # detection result file로 부터 검출결과를 읽어드림. # potenitDt.anns
    potenitDt = potenitGt.loadRes(user_submission_file, unannotated_id) # 'jobs/2024-07-03_05h23m_/Epoch040_test_det.txt'

    imgIds = sorted(potenitGt.getImgIds()) # 정렬된 image_id들
    
    coco_eval = COCOeval(potenitGt, potenitDt, 'bbox', result_dir, epoch)
    
    # 각종 파라미터 설정
    coco_eval.params.imgIds = imgIds

    # 수치 뽑는 3종셋트
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # AR50
    iou_thr = 0.5
    iou_thr_ids = list(cocoEval.params.iouThrs).index(iou_thr)
    ar50 = cocoEval.eval['recall'][iou_thr_ids, :, 0, 2]
    ar50 = ar50[ar50 > -1].mean()
    log_path = os.path.join(arguments.result_dir, 'eval_results.txt')
    with open(log_path, 'a') as file:
        file.write(f'AR50 : {ar50:.2f}')
    file.close()
    print(f'AR50 : {ar50:.2f}')
    
    # depth 산출
    GT, results = depth_eval.parsed_result(test_annotation_file, user_submission_file, type='COCO')
    depth_eval.Depth_Average_Precision(GT=GT, results=results, result_dir=result_dir)
    
    
    



if __name__ == '__main__':  
    ################################ eval_AR_mAP ###################################
    parser = argparse.ArgumentParser(description='Process some integers.')
        
    parser.add_argument('--model_path', type=str, default='/home/hscho/workspace/src/MLPD/src/jobs/2024-07-16_14h15m_/checkpoint_ssd300.pth.tar039',
                            help='Pretrained model for evaluation.')
    parser.add_argument('--result_dir', type=str, default='result',
                            help='Save result directory')
    # parser.add_argument('--vis', action='store_true', 
    #                         help='Visualizing the results')
    parser.add_argument('--isVisualize', action='store_true', help="visualize results")
    parser.add_argument('--mode', type=str, default='test')

    ####################################################################################
    parser.add_argument('--num_workers', type = int)

    arguments = parser.parse_args() 

    print(f'eval with {arguments.mode} mode')
    # if arguments.mode == "train":
    #     main_train()
    # else:
    #     main_test()
    epoch = int(arguments.model_path.split('.')[-1].split('0')[-1])
    print(epoch)
    test_during_train(model_path=arguments.model_path, result_dir=arguments.result_dir, num_workers=arguments.num_workers, epoch=epoch, isVisualize=False)

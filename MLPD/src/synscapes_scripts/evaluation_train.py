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

def run_inference_train(model_path: str) -> Dict:
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
    input_size = config.test.input_size # [512, 640]

    # # Load dataloader for Fusion Dead Zone experiment
    # FDZ = [FusionDeadZone(config.FDZ_case[fdz_case], tuple(input_size))]
    # config.test.img_transform.add(FDZ)

    args = config.args
    batch_size = config.test.batch_size * torch.cuda.device_count()
    train_dataset = KAISTPed(args, condition="train")
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              num_workers=12,
                                              collate_fn=train_dataset.collate_fn,
                                              pin_memory=True)
   
    results = val_visualize_epoch_potenit(model, train_loader, input_size)
    # results = val_epoch_potenit(model, train_loader, input_size)
    return results


def val_visualize_epoch_potenit(model: SSD300, dataloader: DataLoader, input_size: Tuple, min_score: float = 0.1) -> Dict:
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
    
        
    transform_ = ToPILImage()  
    fnt = ImageFont.load_default()
    color = (0,255,0)
    
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
            
            detections = model.detect_objects(predicted_locs, predicted_scores, predicted_depths,
                                                     min_score=min_score, max_overlap=0.425, top_k=200)

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
                # results[image_id.item() + 1] = np.hstack([xywh_np, scores_np])
                # (xmin, ymin, xmax, ymax)
                results[image_id.item() + 1] = np.hstack([xyxy_np, scores_np, depths_np])
            
    return results

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

def visualize_RGB(transform, fnt, color, result, image, image_id):

    # (xmin, ymin, xmax, ymax)
    import pdb;pdb.set_trace()
    for n in range(len(image)):  
        # import pdb;pdb.set_trace()

        vis_ = (image[n] - image[n].min()) / (image[n].max() - image[n].min()) # [0, 1] 범위로 정규화
        vis_ = (vis_ * 255).type(torch.uint8) # uint8, 0 ~ 255로 변환
        vis_ = transform(vis_)
        draw1 = ImageDraw.Draw(vis_)
 
        for i, info in enumerate(result):
            import pdb;pdb.set_trace()
            box = info[:4]
            score = info[4]
            depth = info[5]
            draw1.rectangle(box.tolist(), outline="red", width=2)
            draw1.text(((box[0]+box[2])/2, (box[1]+box[3])/2), f"{depth}", font=fnt, fill ="red")

        ToTensor()(vis_)
        vis_.save(f'result_visualization_R/model_inference_R_after_treat_resize_{image_id.item() + 1}.jpg')
        #################################################################################

def visualize_Heatmap(transform, image_depth, image_id):
    import pdb;pdb.set_trace()
    depth_ = transform_(image_depth)
    draw2 = ImageDraw.Draw(depth_)
            
    depth_heatmap =  (np.array(image_depth.cpu().squeeze())*255).astype("uint8")
    depth_heatmap_equalized1 = cv2.equalizeHist(depth_heatmap)
    
        # np.array(depth_heatmap_equalized.squeeze().cpu())
    sns.heatmap(np.array(depth_heatmap_equalized1), 
                cmap='RdYlGn_r', 
                cbar = False, 
                xticklabels = False, 
                yticklabels = False) # depthmap heatmap으로 시각화
    
    plt.savefig(f"result_visualization_Heat/depth_heatmap_{image_id.item() + 1}.jpg", dpi=400) # heatmap 저장

def visualize_FoVImage(transform, fnt, color, result):
    pass

def evaluate_potenit(test_annotation_file: str, user_submission_file: str, phase_codename: str = 'Multispectral'):
    """
    Parameters
    ----------
    test_annotations_file: str
        Path to test_annotation_file on the server
    user_submission_file: str
        Path to file submitted by the user
    phase_codename: str
        Phase to which submission is made
    
    """
    potenitGt = Potenit(test_annotation_file) # 'MDPI_test_solution.json'
    import pdb;pdb.set_trace()
    potenitDt = potenitGt.loadRes(user_submission_file) # 'jobs/2024-07-03_05h23m_/Epoch040_test_det.txt'
    
    imgIds = sorted(potenitGt.getImgIds()) # 정렬된 image_id들
    import pdb;pdb.set_trace()
    method = os.path.basename(user_submission_file).split('_')[0] # 'Epoch040'
    import pdb;pdb.set_trace()
    # kaistEval = KAISTPedEval(kaistGt, kaistDt, 'bbox', method)
    potenitEval = PotenitEval(potenitGt, potenitDt, 'bbox', method)
    import pdb;pdb.set_trace()
    # kaistEval.params.catIds = [1]
    potenitEval.params.catIds = [1]

    # eval_result = {
    #     'all': copy.deepcopy(kaistEval),
    #     'day': copy.deepcopy(kaistEval),
    #     'night': copy.deepcopy(kaistEval),
    # }
    eval_result = {
        'all': copy.deepcopy(potenitEval)
    }
    import pdb;pdb.set_trace()
    if 'all' in eval_result.keys():
        eval_result['all'].params.imgIds = imgIds # params.imgIds에 All에 해당하는 이미지라벨을 할당
        eval_result['all'].evaluate(0)
        eval_result['all'].accumulate()
        MR_all = eval_result['all'].summarize(0)
    
    if 'day' in eval_result.keys():
        eval_result['day'].params.imgIds = imgIds[:1455]
        eval_result['day'].evaluate(0)
        eval_result['day'].accumulate()
        MR_day = eval_result['day'].summarize(0)

    if 'night' in eval_result.keys():
        eval_result['night'].params.imgIds = imgIds[1455:]
        eval_result['night'].evaluate(0)
        eval_result['night'].accumulate()
        MR_night = eval_result['night'].summarize(0)

    recall_all = 1 - eval_result['all'].eval['yy'][0][-1]
    title_str = f'\n########## Method: {method} ##########\n'
    msg = title_str \
        + f'MR_all: {MR_all * 100:.2f}\n' \
        + f'MR_day: {MR_day * 100:.2f}\n' \
        + f'MR_night: {MR_night * 100:.2f}\n' \
        + f'recall_all: {recall_all * 100:.2f}\n' \
        + '#'*len(title_str) + '\n\n'
    print(msg)


def detect(original_image, original_lwir, detection, \
        min_score=0.5, max_overlap=0.425, top_k=200, \
        suppress=None, width=2):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """
    
    det_boxes = detection[:,1:5]

    small_object =  det_boxes[:, 3] < 55
  
    det_boxes[:,2] = det_boxes[:,0] + det_boxes[:,2]
    det_boxes[:,3] = det_boxes[:,1] + det_boxes[:,3] 
    det_scores = detection[:,5]
    det_labels = list()
    for i in range(len(detection)) : 
        det_labels.append(1.0)
    det_labels = np.array(det_labels)
    det_score_sup = det_scores < min_score    
    det_boxes = det_boxes[~det_score_sup]
    det_scores = det_scores[~det_score_sup]
    det_labels = det_labels[~det_score_sup]
    
    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels]

    # PIL from Tensor
    original_image = original_image.squeeze().permute(1, 2, 0)
    original_image = original_image.numpy() * 255
    original_lwir = original_lwir.squeeze().numpy() * 255
    original_image = Image.fromarray(original_image.astype(np.uint8))
    original_lwir = Image.fromarray(original_lwir.astype(np.uint8))


    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        new_image = Image.new('RGB',(2*original_image.size[0], original_image.size[1]))
        new_image.paste(original_image,(0,0))
        new_image.paste(original_lwir,(original_image.size[0],0))
        return new_image

    # Annotate
    annotated_image = original_image
    annotated_image_lwir = original_lwir
    draw = ImageDraw.Draw(annotated_image)
    draw_lwir = ImageDraw.Draw(annotated_image_lwir)
    font = ImageFont.truetype("./utils/calibril.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.shape[0]):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue
                
        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]], width=width)
        draw_lwir.rectangle(xy=box_location, outline=label_color_map[det_labels[i]], width=width)
        
        # Text       
        text_score_vis = str(det_scores[i].item())[:7]
        text_score_lwir = str(det_scores[i].item())[:7]
        
        text_size_vis = font.getsize(text_score_vis)
        text_size_lwir = font.getsize(text_score_lwir)

        text_location_vis = [box_location[0] + 2., box_location[1] - text_size_vis[1]]
        textbox_location_vis = [box_location[0], box_location[1] - text_size_vis[1], box_location[0] + text_size_vis[0] + 4.,box_location[1]]
        
        text_location_lwir = [box_location[0] + 2., box_location[1] - text_size_lwir[1]]
        textbox_location_lwir = [box_location[0], box_location[1] - text_size_lwir[1], box_location[0] + text_size_lwir[0] + 4.,box_location[1]]

        draw.rectangle(xy=textbox_location_vis, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location_vis, text='{:.4f}'.format(det_scores[i].item()), fill='white', font=font)
        
        draw_lwir.rectangle(xy=textbox_location_lwir, fill=label_color_map[det_labels[i]])
        draw_lwir.text(xy=text_location_lwir, text='{:.4f}'.format(det_scores[i].item()), fill='white', font=font)
    
    new_image = Image.new('RGB',(original_image.size[0], original_image.size[1]))
    new_image.paste(original_image,(0,0))
    new_image_lwir = Image.new('RGB',(original_image.size[0], original_image.size[1]))
    new_image_lwir.paste(original_lwir,(0,0))

    del draw
    del draw_lwir

    return new_image, new_image_lwir


def visualize(result_filename, vis_dir):
    import pdb;pdb.set_trace()
    data_list = list()
    for line in open(result_filename):
        data_list.append(line.strip().split(','))
    data_list = np.array(data_list)
    import pdb;pdb.set_trace()
    input_size = config.test.input_size
    
    # Load dataloader for Fusion Dead Zone experiment
    config.args.test.co_transform = Compose([
                                             Resize(input_size), \
                                             ToTensor()
                                            ])

    train_dataset = KAISTPed(config.args, condition="train")
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=1,
                                              num_workers=config.args.dataset.workers,
                                              collate_fn=test_dataset.collate_fn,
                                              pin_memory=True)
     f.write(f'{image_id},{x_min:.4f},{y_min:.4f},{x_max:.4f},{y_max:.4f},{score:.8f},{depth:.4f}\n')

    for idx, blob in enumerate(tqdm(train_loader, desc='Visualizing')): 
        
        import pdb;pdb.set_trace()
        
        image_vis, image_depth, _, _, _ = blob
        detection = data_list[data_list[:,0] == str(idx+1)].astype(float)
        vis, lwir = detect(image_vis, image_lwir, detection)

        # if fdz_case=='sidesblackout_a':
        #     new_images = Image.blend(vis, lwir, 0.5)
        # elif fdz_case=='sidesblackout_b':
        #     new_images = Image.blend(lwir, vis, 0.5)
        # elif fdz_case=='surroundingblackout':
        #     ## We emptied the center considering the visualization.
        #     ## In reality, the original image is used as an input.
        #     x = 120
        #     y = 96
        #     vv = np.array(vis)
        #     vv[y:-y, x:-x] = 0
        #     ##
        #     new_images = np.array(lwir) + vv
        #     new_images = Image.fromarray(new_images.astype(np.uint8))
        # elif fdz_case in ['blackout_r', 'blackout_t', 'original']:
        #     new_images = Image.new('RGB',(2*vis.size[0], vis.size[1]))
        #     new_images.paste(vis,(0,0))
        #     new_images.paste(lwir,(vis.size[0],0))

        new_images.save('./{}/{:06d}.jpg'.format(vis_dir, idx))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--model-path', required=True, type=str,
    #                     help='Pretrained model for evaluation.')
    parser.add_argument('--model-path', type=str, default='/home/hscho/workspace/ssd2.5d/MLPD-Multi-Label-Pedestrian-Detection/src/jobs/2024-07-05_09h40m_/checkpoint_ssd300.pth.tar039',
                         help='Pretrained model for evaluation.')
    parser.add_argument('--result-dir', type=str, default='../result',
                        help='Save result directory')
    parser.add_argument('--vis', action='store_true', 
                        help='Visualizing the results')
    
    arguments = parser.parse_args()

    print(arguments)

    model_path = Path(arguments.model_path).stem.replace('.', '_') # 'checkpoint_ssd300_pth

    # Run inference to get detection results
    os.makedirs(arguments.result_dir, exist_ok=True) # '../result'
    result_filename = opj(arguments.result_dir,  f'{model_path}_TEST_det') # '../result/checkpoint_ssd300_pth_TEST_det'

    # Run inference
    # results = run_inference_train(arguments.model_path)
    # import pickle
    # with open('run_inference.pickle', 'wb') as f:
    #     pickle.dump(results, f)
    import pdb;pdb.set_trace()
    import pickle 
    with open('run_inference.pickle', 'rb') as f:
        results = pickle.load(f)

        
    # Save results
    save_results_potenit(results, result_filename)


    # visualize image
    # import pdb;pdb.set_trace()
    # visualize_RGB(transform = transform_, fnt = fnt, color = color, result = results[image_id.item() + 1], image = image_vis, image_id = indices)
    # visualize_Heatmap(transform = transform_, image_depth = image_depth, image_id = indices)
    # # visualize_FoVImage(transform = transform_, fnt = fnt, color = color, result = results[image_id.item() + 1], image = image_vis)
    
    
    # Visualizing
    # if arguments.vis:
    if True:
        vis_dir = opj(arguments.result_dir, 'vis', model_path) # '../result/vis/checkpoint_ssd300_pth
        os.makedirs(vis_dir, exist_ok=True) 
        visualize(result_filename + '.txt', vis_dir)

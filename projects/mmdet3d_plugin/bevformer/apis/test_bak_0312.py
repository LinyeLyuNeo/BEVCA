# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import os.path as osp
import pickle
import shutil
import tempfile
import time
import pickle
from nuscenes.utils.data_classes import Box

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from mmdet.core import encode_mask_results


import mmcv
import numpy as np
import pycocotools.mask as mask_util

def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    # Load the annotation file that contains 3D boxes and labels
    ann_file = '/data/lyulinye/bevformer/data/nuscenes/nuscenes_infos_temporal_val.pkl'  # Path to your annotations file
    with open(ann_file, 'rb') as f:
        annotations = pickle.load(f)
        
    # 准备训练数据
    gt_bev_tensors = []
    pred_bev_tensors = []    
        
    for i, data in enumerate(data_loader):
        
        # get the sample_idx from the test dataset
        # Access the DataContainer object
        img_metas_container = data['img_metas'][0]
        # Access the data attribute of the DataContainer
        img_metas_data = img_metas_container.data
        # Access the first element of the data
        first_element = img_metas_data[0]
        sample_idx = first_element[0]['sample_idx']
        
        # Find the annotation entry corresponding to the sample_idx
        sample_info = next((info for info in annotations['infos'] if info['token'] == sample_idx), None)
        
        if sample_info is None:
            print(f"Sample Info Not Found for sample_idx: {sample_idx}")
        else:
            print(f"Sample Info Found for sample_idx: {sample_idx}")

        if sample_info is not None:
            print("Sample Info Found:")
            # Access ground truth boxes
            gt_boxes = sample_info['gt_boxes']
            # print(f"Ground Truth Boxes: {gt_boxes}")

            # Access ground truth names
            gt_names = sample_info['gt_names']
            # print(f"Ground Truth Names: {gt_names}")
        else:
            print("Sample Info Not Found for the given sample_idx")
            
        # Initialize the BEV tensor
        grid_size = 200
        

        # Define range for BEV tensor
        min_range = -51.2
        max_range = 51.2
        range_size = max_range - min_range
        
        # Class label to index mapping
        detection_class_to_index = {
            'barrier': 1,
            'bicycle': 2,
            'bus': 3,
            'car': 4,
            'construction_vehicle': 5,
            'motorcycle': 6,
            'pedestrian': 7,
            'traffic_cone': 8,
            'trailer': 9,
            'truck': 10
        }
        # Function to map world coordinates to BEV grid coordinates
        def world_to_grid_coordinates(x, y, min_range, range_size, grid_size):
            x_grid = int((x - min_range) / range_size * grid_size)
            y_grid = int((y - min_range) / range_size * grid_size)
            return x_grid, y_grid
        
        # Function to create BEV tensor from bounding boxes and labels
        def create_bev_tensor(bboxes, labels, grid_size, min_range, range_size):
            tensor = np.zeros((grid_size, grid_size), dtype=int)
            for i, bbox in enumerate(bboxes):
                x, y, z, length, width, height, yaw = bbox
                label = labels[i]
                # Get the grid coordinates for the center of the bounding box
                x_grid, y_grid = world_to_grid_coordinates(x, y, min_range, range_size, grid_size)
                
                # Get the class index from the label
                class_index = detection_class_to_index.get(label, 0)
                
                # You can mark multiple grid cells based on the bounding box size (length, width)
                # Here, we mark the center cell and expand based on the box size
                length_cells = int(length / range_size * grid_size)
                width_cells = int(width / range_size * grid_size)

                for dx in range(-length_cells // 2, length_cells // 2 + 1):
                    for dy in range(-width_cells // 2, width_cells // 2 + 1):
                        grid_x = min(max(x_grid + dx, 0), grid_size - 1)
                        grid_y = min(max(y_grid + dy, 0), grid_size - 1)
                        tensor[grid_x, grid_y] = class_index

            return tensor

        # Create the GT BEV tensor
        gt_bev_tensor = create_bev_tensor(gt_boxes, gt_names, grid_size, min_range, range_size)
        gt_bev_tensor = torch.tensor(gt_bev_tensor, dtype=torch.float32).to('cuda')  # 转换为张量并移动到 GPU
        gt_bev_tensors.append(gt_bev_tensor)
        
        
        
        with torch.no_grad():
            # getting the predctions of boxes_3d, scores_3d and labels_3d
            result, pred_bev_tensor = model(return_loss=False, rescale=True, **data)
            pred_bev_tensors.append(pred_bev_tensor)
            # encode mask results
            if isinstance(result, dict):
                if 'bbox_results' in result.keys():
                    bbox_result = result['bbox_results']
                    batch_size = len(result['bbox_results'])
                    bbox_results.extend(bbox_result)
                if 'mask_results' in result.keys() and result['mask_results'] is not None:
                    mask_result = custom_encode_mask_results(result['mask_results'])
                    mask_results.extend(mask_result)
                    have_mask = True
            else:
                batch_size = len(result)
                bbox_results.extend(result)

            #if isinstance(result[0], tuple):
            #    assert False, 'this code is for instance segmentation, which our code will not utilize.'
            #    result = [(bbox_results, encode_mask_results(mask_results))
            #              for bbox_results, mask_results in result]
        if rank == 0:
            
            for _ in range(batch_size * world_size):
                prog_bar.update()
                
     # 将数据转换为张量
    gt_bev_tensors = torch.stack(gt_bev_tensors)
    pred_bev_tensors = torch.stack(pred_bev_tensors)

    # 转换数据形状以适应 DataLoader
    gt_bev_tensors = gt_bev_tensors.permute(0, 2, 1)  # 形状变为 (N, 200, 200)
    # pred_bev_tensors = pred_bev_tensors.permute(0, 3, 1, 2)  # 形状变为 (N, 40000, 1, 256)

    # 定义数据集和数据加载器
    gt_encoder_dataset = TensorDataset(gt_bev_tensors, pred_bev_tensors)
    # gt_encoder_dataloader = DataLoader(gt_encoder_dataset, batch_size=1, shuffle=True)
    
    # 保存 gt_encoder_dataset 到文件
    torch.save(gt_encoder_dataset, 'gt_encoder_val_dataset_bev_50_dim_256.pth')
    
    
    
    
    # collect results from all ranks
    if gpu_collect:
        bbox_results = collect_results_gpu(bbox_results, len(dataset))
        if have_mask:
            mask_results = collect_results_gpu(mask_results, len(dataset))
        else:
            mask_results = None
    else:
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        tmpdir = tmpdir+'_mask' if tmpdir is not None else None
        if have_mask:
            mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir)
        else:
            mask_results = None

    if mask_results is None:
        return bbox_results
    return {'bbox_results': bbox_results, 'mask_results': mask_results}


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)
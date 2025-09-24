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
from torch.utils.data import DataLoader, TensorDataset, Dataset

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

class BEVCustomDataset(Dataset):
    def __init__(self, p_tensors, alpha_tensors):
        self.p_tensors = p_tensors
        self.alpha_tensors = alpha_tensors

    def __len__(self):
        return len(self.p_tensors)

    def __getitem__(self, idx):
        return self.p_tensors[idx], self.alpha_tensors[idx]

# def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
#     """Test model with multiple gpus.
#     This method tests model with multiple gpus and collects the results
#     under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
#     it encodes results to gpu tensors and use gpu communication for results
#     collection. On cpu mode it saves the results on different gpus to 'tmpdir'
#     and collects them by the rank 0 worker.
#     Args:
#         model (nn.Module): Model to be tested.
#         data_loader (nn.Dataloader): Pytorch data loader.
#         tmpdir (str): Path of directory to save the temporary results from
#             different gpus under cpu mode.
#         gpu_collect (bool): Option to use either gpu or cpu to collect results.
#     Returns:
#         list: The prediction results.
#     """
#     model.eval()
#     bbox_results = []
#     mask_results = []
#     dataset = data_loader.dataset
#     rank, world_size = get_dist_info()
#     if rank == 0:
#         prog_bar = mmcv.ProgressBar(len(dataset))
#     time.sleep(2)  # This line can prevent deadlock problem in some cases.
#     have_mask = False
#     # Load the annotation file that contains 3D boxes and labels
#     ann_file = '/data/lyulinye/bevformer/data/nuscenes/nuscenes_infos_temporal_val.pkl'  # Path to your annotations file
#     with open(ann_file, 'rb') as f:
#         annotations = pickle.load(f)
        
#     # Class label to index mapping
#     detection_class_to_index = {
#         'barrier': 1,
#         'bicycle': 2,
#         'bus': 3,
#         'car': 4,
#         'construction_vehicle': 5,
#         'motorcycle': 6,
#         'pedestrian': 7,
#         'traffic_cone': 8,
#         'trailer': 9,
#         'truck': 10
#     }
    
#     def world_to_grid_coordinates(x, y, min_range, range_size, grid_size):
#         x_grid = int((x - min_range) / range_size * grid_size)
#         y_grid = int((y - min_range) / range_size * grid_size)
#         return x_grid, y_grid

#     def crop(bev_tensor, x, y, z, length, width, height, yaw, min_range=-51.2, max_range=51.2, grid_size=200):
#         range_size = max_range - min_range
        
#         # Reshape BEV tensor from (40000, 1, 256) to (200, 200, 1, 256)
#         bev_tensor = bev_tensor.view(grid_size, grid_size, 1, -1)

#         # Convert world coordinates to grid coordinates
#         x_grid, y_grid = world_to_grid_coordinates(x, y, min_range, range_size, grid_size)

#         # Calculate the number of cells to crop based on the bounding box size
#         length_cells = int(length / range_size * grid_size)
#         width_cells = int(width / range_size * grid_size)

#         # Calculate the crop boundaries
#         x_min = max(x_grid - length_cells // 2, 0)
#         x_max = min(x_grid + length_cells // 2, grid_size - 1)
#         y_min = max(y_grid - width_cells // 2, 0)
#         y_max = min(y_grid + width_cells // 2, grid_size - 1)

#         # Crop the BEV tensor
#         cropped_bev_tensor = bev_tensor[x_min:x_max+1, y_min:y_max+1, :, :]

#         return cropped_bev_tensor, x_min, x_max, y_min, y_max
    
#     def average_and_reshape(cropped_bev_tensor):
#         # 对前两个维度进行平均操作
#         averaged_tensor = cropped_bev_tensor.mean(dim=(0, 1), keepdim=True)
        
#         # 将张量从 [1, 1, 1, 256] 转换为 [1, 256]
#         reshaped_tensor = averaged_tensor.view(1, 256)
        
#         return reshaped_tensor

#     # 初始化 p_tensors、cropped_bev_tensors 和 alpha_tensors
#     p_tensors = []
#     alpha_tensors = []
        
#     for i, data in enumerate(data_loader):  # for each frame of the data
        
#         with torch.no_grad():
#             # getting the predctions of boxes_3d, scores_3d and labels_3d
#             result, bev_tensor = model(return_loss=False, rescale=True, **data)
#             # pred_bev_tensors.append(pred_bev_tensor)
#             # encode mask results
#             if isinstance(result, dict):
#                 if 'bbox_results' in result.keys():
#                     bbox_result = result['bbox_results']
#                     batch_size = len(result['bbox_results'])
#                     bbox_results.extend(bbox_result)
#                 if 'mask_results' in result.keys() and result['mask_results'] is not None:
#                     mask_result = custom_encode_mask_results(result['mask_results'])
#                     mask_results.extend(mask_result)
#                     have_mask = True
#             else:
#                 batch_size = len(result)
#                 bbox_results.extend(result)
#         ## obtaining the ground truth boxes and labels
#         # get the sample_idx from the test dataset
#         # Access the DataContainer object
#         img_metas_container = data['img_metas'][0]
#         # Access the data attribute of the DataContainer
#         img_metas_data = img_metas_container.data
#         # Access the first element of the data
#         first_element = img_metas_data[0]
#         sample_idx = first_element[0]['sample_idx']
        
#         # Find the annotation entry corresponding to the sample_idx
#         sample_info = next((info for info in annotations['infos'] if info['token'] == sample_idx), None)
        
#         if sample_info is None:
#             print(f"Sample Info Not Found for sample_idx: {sample_idx}")
#         else:
#             print(f"Sample Info Found for sample_idx: {sample_idx}")

#         if sample_info is not None:
#             print("Sample Info Found:")
#             # Access ground truth boxes
#             gt_boxes = sample_info['gt_boxes']
#             # print(f"Ground Truth Boxes: {gt_boxes}")

#             # Access ground truth names
#             gt_names = sample_info['gt_names']
#             # print(f"Ground Truth Names: {gt_names}")
            
#             for i, bbox in enumerate(gt_boxes):
#                 x, y, z, length, width, height, yaw = bbox
#                 label = gt_names[i]
#                 # Get the class index from the label
#                 class_index = detection_class_to_index.get(label, 0)
                
#                 if class_index == 7: # only collect car/pedestrian label data
#                     # get the alpha_i from cropping and pooling operation from the BEV tensor 
                    
#                     cropped_bev_tensor, x_min, x_max, y_min, y_max = crop(bev_tensor, x, y, z, length, width, height, yaw)
                    
#                     # 对裁剪后的 BEV 张量进行平均操作并重新调整形状
#                     reshaped_tensor = average_and_reshape(cropped_bev_tensor)
                    
#                     # 创建 p_tensor
#                     p_tensor = torch.tensor([x_min, x_max, y_min, y_max], dtype=torch.float32).cuda()
                    
#                     # 将 p_tensor、cropped_bev_tensor 和 reshaped_tensor 添加到列表中
#                     p_tensors.append(p_tensor)
#                     alpha_tensors.append(reshaped_tensor)
                    
#         else:
#             print("Sample Info Not Found for the given sample_idx")
            
#         if rank == 0:
#             for _ in range(batch_size * world_size):
#                 prog_bar.update()
                
#     # 将 p_tensors、cropped_bev_tensors 和 alpha_tensors 转换为张量
#     p_tensors = torch.stack(p_tensors)
#     alpha_tensors = torch.stack(alpha_tensors)
    
#     # 创建自定义数据集
#     bev_custom_dataset = BEVCustomDataset(p_tensors, alpha_tensors)
    
#     # 保存数据集到文件
#     torch.save(bev_custom_dataset, 'pedestrian_gtencoder_custom_dataset.pth')
    
#     # collect results from all ranks
#     if gpu_collect:
#         bbox_results = collect_results_gpu(bbox_results, len(dataset))
#         if have_mask:
#             mask_results = collect_results_gpu(mask_results, len(dataset))
#         else:
#             mask_results = None
#     else:
#         bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
#         tmpdir = tmpdir+'_mask' if tmpdir is not None else None
#         if have_mask:
#             mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir)
#         else:
#             mask_results = None

#     if mask_results is None:
#         return bbox_results
#     return {'bbox_results': bbox_results, 'mask_results': mask_results}

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
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # result = model(return_loss=False, rescale=True, **data)
            # for bevformer
            result, bev_tensor = model(return_loss=False, rescale=True, **data)  
            # for bevformerv2
            # result = model(return_loss=False, rescale=True, **data)  
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

def custom_multi_gpu_test_v2(model, data_loader, tmpdir=None, gpu_collect=False):
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
    
    def world_to_grid_coordinates(x, y, min_range, range_size, grid_size):
        x_grid = int((x - min_range) / range_size * grid_size)
        y_grid = int((y - min_range) / range_size * grid_size)
        return x_grid, y_grid

    def crop(bev_tensor, x, y, z, length, width, height, yaw, min_range=-51.2, max_range=51.2, grid_size=200):
        range_size = max_range - min_range
        
        # Reshape BEV tensor from (40000, 1, 256) to (200, 200, 1, 256)
        bev_tensor = bev_tensor.view(grid_size, grid_size, 1, -1)

        # Convert world coordinates to grid coordinates
        x_grid, y_grid = world_to_grid_coordinates(x, y, min_range, range_size, grid_size)

        # Calculate the number of cells to crop based on the bounding box size
        length_cells = int(length / range_size * grid_size)
        width_cells = int(width / range_size * grid_size)

        # Calculate the crop boundaries
        x_min = max(x_grid - length_cells // 2, 0)
        x_max = min(x_grid + length_cells // 2, grid_size - 1)
        y_min = max(y_grid - width_cells // 2, 0)
        y_max = min(y_grid + width_cells // 2, grid_size - 1)

        # Crop the BEV tensor
        cropped_bev_tensor = bev_tensor[x_min:x_max+1, y_min:y_max+1, :, :]

        return cropped_bev_tensor, x_min, x_max, y_min, y_max
    
    def average_and_reshape(cropped_bev_tensor):
        # 对前两个维度进行平均操作
        averaged_tensor = cropped_bev_tensor.mean(dim=(0, 1), keepdim=True)
        
        # 将张量从 [1, 1, 1, 256] 转换为 [1, 256]
        reshaped_tensor = averaged_tensor.view(1, 256)
        
        return reshaped_tensor

    # 初始化 p_tensors、cropped_bev_tensors 和 alpha_tensors
    p_tensors = []
    alpha_tensors = []
        
    for i, data in enumerate(data_loader):  # for each frame of the data
        
        with torch.no_grad():
            # getting the predctions of boxes_3d, scores_3d and labels_3d
            result, bev_tensor = model(return_loss=False, rescale=True, **data)
            # pred_bev_tensors.append(pred_bev_tensor)
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
        ## obtaining the ground truth boxes and labels
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
            
            for i, bbox in enumerate(gt_boxes):
                x, y, z, length, width, height, yaw = bbox
                label = gt_names[i]
                # Get the class index from the label
                class_index = detection_class_to_index.get(label, 0)
                
                if class_index == 7: # only collect car/pedestrian label data
                    # get the alpha_i from cropping and pooling operation from the BEV tensor 
                    
                    cropped_bev_tensor, x_min, x_max, y_min, y_max = crop(bev_tensor, x, y, z, length, width, height, yaw)
                    
                    # 对裁剪后的 BEV 张量进行平均操作并重新调整形状
                    reshaped_tensor = average_and_reshape(cropped_bev_tensor)
                    
                    # 创建 p_tensor
                    p_tensor = torch.tensor([x_min, x_max, y_min, y_max], dtype=torch.float32).cuda()
                    
                    # 将 p_tensor、cropped_bev_tensor 和 reshaped_tensor 添加到列表中
                    p_tensors.append(p_tensor)
                    alpha_tensors.append(reshaped_tensor)
                    
        else:
            print("Sample Info Not Found for the given sample_idx")
            
        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()
                
    # 将 p_tensors、cropped_bev_tensors 和 alpha_tensors 转换为张量
    p_tensors = torch.stack(p_tensors)
    alpha_tensors = torch.stack(alpha_tensors)
    
    # 创建自定义数据集
    bev_custom_dataset = BEVCustomDataset(p_tensors, alpha_tensors)
    
    # 保存数据集到文件
    torch.save(bev_custom_dataset, 'pedestrian_gtencoder_custom_dataset.pth')
    
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
    if (tmpdir is None):
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
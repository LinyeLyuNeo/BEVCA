import torch.autograd
torch.autograd.set_detect_anomaly(True)  # 帮助定位梯度问题
import os

# 设置环境变量，避免分布式训练问题
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['LOCAL_RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29501'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
os.environ["CUDA_VISIBLE_DEVICES"] = '6'

import argparse
import logging
import torch
import numpy as np

import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import math
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    AmbientLights,
    TexturesVertex
)
import torch.nn.functional as F

import os

import time
from pathlib import Path
import multiprocessing as mp
import numpy as np
import torch.utils.data
import torch.nn as nn
import yaml
from distutils.version import LooseVersion
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image, ImageDraw
from models.yolo import Model
from utils.datasets_RAUCA import create_dataloader
from utils.general_RAUCA import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
     get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, set_logging, colorstr
from utils.google_utils import attempt_download
from utils.loss_RAUCA import ComputeLoss
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, de_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
import neural_renderer
from PIL import Image,ImageOps
from Image_Segmentation.network import U_Net
import torch.nn.functional as F
import kornia.augmentation as K
import torchvision.transforms as Tr
import kornia.geometry as KG
import torchvision.models as models
from neural_renderer.load_obj_differentiable_new import LoadTextures

import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from PIL import Image
import torchvision.transforms as transforms

import sys
import os

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
from nuscenes.nuscenes import NuScenes
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
logger = logging.getLogger(__name__)
import torch.nn.functional as F

def pad_tensor_to_multiple(tensor, divisor=32, pad_value=0):
    B, N, C, H, W = tensor.shape
    new_H = ((H + divisor - 1) // divisor) * divisor
    new_W = ((W + divisor - 1) // divisor) * divisor

    pad_bottom = new_H - H
    pad_right = new_W - W

    padded_tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode='constant', value=pad_value)
    return padded_tensor

def mix_image(image_optim, mask,origin_image):
    return (1 - mask) * origin_image + mask *torch.clamp(image_optim,0,1)

def flip_image(image_path, saved_location):
    """
    Flip or mirror the image

    @param image_path: The path to the image to edit
    @param saved_location: Path to save the flipped image
    """
    image_obj = Image.open(image_path)
    flipped_image = ImageOps.flip(image_obj)
    flipped_image.save(saved_location)

class ROA(torch.nn.Module):
    def __init__(self):
        super(ROA, self).__init__()
        self.randomAffine = K.RandomAffine(degrees=(-5.0, 5.0), translate=(0.1, 0.1),scale=(0.7,1.0),keepdim=True)
        self.colorJiggle = K.ColorJiggle(0.25, (0.75,1.5),p=0.5,keepdim=True)

    def forward(self, x):
        x = self.randomAffine(x)
        if not torch.isnan(x).any():
            x = self.colorJiggle(x)
        return x

def draw_red_origin(file_path):
    image = Image.open(file_path)

    width, height = image.size

    new_image = Image.new('RGBA', (width, height))
    draw = ImageDraw.Draw(new_image)

    center_x = width // 2
    center_y = height // 2

    radius = 3
    draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill=(255, 0, 0))
    print(new_image.size,image.convert('RGBA').size)
    result_image = Image.alpha_composite(image.convert('RGBA'), new_image)
    result_file_path = file_path
    result_image.save(result_file_path)
    return result_file_path

def loss_smooth(img, mask):
    print(f"img.shape:{img.shape}")
    print(f"mask.shape:{mask.shape}")
    s1 = torch.pow(img[:, :, 1:, :-1] - img[:, :, :-1, :-1], 2)
    s2 = torch.pow(img[:, :, :-1, 1:] - img[:, :, :-1, :-1], 2)
    mask = mask[:, :,:-1, :-1]
    return T * torch.sum(mask * (s1 + s2))

def loss_smooth_UVmap(img, mask):
    img = img.squeeze(0)
    mask = mask.squeeze(0)

    s1 = torch.pow(img[ 1:, :-1, :] - img[:-1, :-1, :], 2)
    s2 = torch.pow(img[ :-1, 1:, :] - img[ :-1, :-1, :], 2)
    mask = mask[ :-1,:-1, :]
    return T * torch.sum(mask * (s1 + s2))/2048.0/2048.0

def cal_texture(texture_param, texture_origin, texture_mask, texture_content=None, CONTENT=False,):
    if CONTENT:
        textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1)
    else:
        textures = 0.5 * (torch.nn.Tanh()(texture_param) + 1)
    return texture_origin * (1 - texture_mask) + texture_mask * textures

def convert_nuscenes_to_pytorch3d(cam_in_ann, device,fov,aspect_ratio):
    R_nu = cam_in_ann[:3, :3]
    T_nu = cam_in_ann[:3, 3]
    R_old = np.array([
    [-1,  0,  0],
    [0, -1,  0],
    [0,  0,  1]
    ])
    R_pytorch3d = R_nu
    T_pytorch3d = T_nu
    eye = torch.tensor(T_pytorch3d, dtype=torch.float32).unsqueeze(0).to(device)
    Rotation =torch.tensor(R_nu@R_old, dtype=torch.float32).unsqueeze(0).to(device)
    T = -torch.bmm(Rotation.transpose(1, 2), eye.unsqueeze(-1)).squeeze(-1)
    cameras = FoVPerspectiveCameras(
        device=device, R=Rotation, T=T, fov=fov,aspect_ratio=aspect_ratio, degrees=True
    )
    return cameras, Rotation, T

def normalize_multiview_images(images, mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False):
    B, N, C, H, W = images.shape

    images_reshaped = images.reshape(-1, C, H, W).clone()
    if to_rgb:
        if C == 3:
            images_reshaped = images_reshaped[:, [2, 1, 0], :, :]
    mean_tensor = torch.tensor(mean, dtype=images.dtype, device=images.device).view(1, C, 1, 1)
    std_tensor = torch.tensor(std, dtype=images.dtype, device=images.device).view(1, C, 1, 1)
    normalized_images = (images_reshaped - mean_tensor) / std_tensor
    normalized_images = normalized_images.reshape(B, N, C, H, W)
    return normalized_images

mapping_matric=np.array([
    [0,1,0],
    [0,0,1],
    [1,0,0]
])

matrix1=np.array([
    [1,0,0],
    [0,-1,0],
    [0,0,1]

])
def create_mask(r, theta, r_limit_value, theta_limit_value):
    # Define the mask shape (200, 200, 1)
    mask = np.zeros((200, 200, 1), dtype=np.uint8)

    # Define the origin of the matrix (100, 100)
    origin_x, origin_y = 100, 100

    # Convert theta to radians if it's in degrees
    theta = math.radians(theta)  # assuming theta is in degrees, convert to radians

    # Loop over all pixels in the matrix
    for x in range(200):
        for y in range(200):
            # Calculate the distance (r) and angle (theta) from the origin
            dx = x - origin_x
            dy = y - origin_y
            r_calc = math.sqrt(dx**2 + dy**2)
            theta_calc = math.atan2(dy, dx)

            theta_calc_deg = math.degrees(theta_calc)

            theta_calc_deg = 90 - theta_calc_deg   # Adjust the angle

            # If it's negative, wrap it into the positive range
            if theta_calc_deg < 0:
                theta_calc_deg += 360
            # Ensure grid_theta_deg is between 0 and 360 degrees
            theta_calc_deg = theta_calc_deg % 360

            # Check if the pixel (x, y) is within the specified r and theta limits
            if r_limit_value[0] <= r_calc <= r_limit_value[1] and \
               (theta_limit_value[0] <= theta_calc_deg <= theta_limit_value[1]):
                mask[x, y] = 1  # Mark this pixel as part of the desired region

    return mask
# Define a function to calculate the theta limits
def calculate_theta_limits(grid_theta_deg, theta_limit_value):
    # Calculate lower and upper limits
    theta_lower_limit = grid_theta_deg - theta_limit_value
    theta_upper_limit = grid_theta_deg + theta_limit_value

    # Wrap the values within the range (0, 360)
    theta_lower_limit = theta_lower_limit % 360
    theta_upper_limit = theta_upper_limit % 360

    # If the lower limit is greater than the upper limit (due to wrapping), swap them
    if theta_lower_limit > theta_upper_limit:
        # Wrap theta_upper_limit around 360
        theta_upper_limit += 360

    return theta_lower_limit, theta_upper_limit

def mean_of_non_zero(tensor):
    """
    Calculate the mean of non-zero values in the given tensor while retaining the grad_fn.

    Args:
        tensor (torch.Tensor): Input tensor, which may have a grad_fn.

    Returns:
        torch.Tensor: Mean of the non-zero values, retains the grad_fn if input tensor has it.
    """
    # Mask out zero values using masked_select, which keeps the grad_fn intact
    non_zero_values = tensor.masked_select(tensor != 0)

    # Check if there are non-zero values to avoid division by zero
    count_non_zero = non_zero_values.numel()

    if count_non_zero > 0:
        sum_non_zero = non_zero_values.sum()  # Sum of non-zero values, retains grad_fn
        mean_non_zero = sum_non_zero / count_non_zero  # Calculate mean
        return mean_non_zero
    else:
        # Return a tensor with 0 and retain grad_fn if necessary
        return torch.mean(tensor)

def train(hyp, opt, device):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # ---------------------------------#
    # -------BEV Model loading---------#
    # ---------------------------------#

    bev_cfg = Config.fromfile(opt.bev_config)
    if opt.cfg_options is not None:
        bev_cfg.merge_from_dict(opt.cfg_options)

    if bev_cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**bev_cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(bev_cfg, 'plugin'):
        if bev_cfg.plugin:
            import importlib
            if hasattr(bev_cfg, 'plugin_dir'):
                plugin_dir = bev_cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(opt.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if bev_cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # set tf32
    if bev_cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    bev_cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(bev_cfg.data.test, dict):
        bev_cfg.data.test.test_mode = True
        samples_per_gpu = bev_cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            bev_cfg.data.test.pipeline = replace_ImageToTensor(
                bev_cfg.data.test.pipeline)
    elif isinstance(bev_cfg.data.test, list):
        for ds_cfg in bev_cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in bev_cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in bev_cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if opt.launcher == 'none':
        distributed = False
    else:
        distributed = False

    # set random seeds
    if opt.seed is not None:
        set_random_seed(opt.seed, deterministic=opt.deterministic)

    # build the config  for RGB with car
    dataset = build_dataset(bev_cfg.data.test)
    cfg_rgb_test_withCar = bev_cfg.data.test.copy()
    cfg_rgb_test_withCar['data_root'] = 'dataset_withcar'
    cfg_rgb_test_withCar['ann_file'] = 'dataset_withcar/nuscenes_infos_temporal_val.pkl'

    # load the annotations for RGB with car
    with open(cfg_rgb_test_withCar['ann_file'], 'rb') as f:
        data_with_car_annotations = pickle.load(f)

    # build the config for RGB without car
    cfg_rgb_test_withoutCar = bev_cfg.data.test.copy()
    cfg_rgb_test_withoutCar['data_root'] = 'dataset_withoutcar'
    cfg_rgb_test_withoutCar['ann_file'] = 'dataset_withoutcar/nuscenes_infos_temporal_val.pkl'

    # build the config for the mask of RGB with car
    cfg_rgb_test_withCar_mask = bev_cfg.data.test.copy()
    cfg_rgb_test_withCar_mask['data_root'] = 'dataset_mask_withcar'
    cfg_rgb_test_withCar_mask['ann_file'] = '/dataset_mask_withcar/nuscenes_infos_temporal_val.pkl'

    # build the config for adversarial texture with car
    cfg_rgb_test_withCar_test = bev_cfg.data.test.copy()
    cfg_rgb_test_withCar_test['data_root'] = 'dataset_withoutcar'
    cfg_rgb_test_withCar_test['ann_file'] = 'dataset_withoutcar/nuscenes_infos_temporal_val.pkl'
    dataset_with_car = build_dataset(cfg_rgb_test_withCar)
    dataset_without_car = build_dataset(cfg_rgb_test_withoutCar)
    dataset_with_car_mask = build_dataset(cfg_rgb_test_withCar_mask)
    dataset_with_car_test = build_dataset(cfg_rgb_test_withCar_test)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=bev_cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=bev_cfg.data.nonshuffler_sampler,
    )
    data_loader_with_car = build_dataloader(
        dataset_with_car,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=bev_cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=bev_cfg.data.nonshuffler_sampler,
    )
    data_loader_without_car = build_dataloader(
        dataset_without_car,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=bev_cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=bev_cfg.data.nonshuffler_sampler,
    )

    data_loader_with_car_mask = build_dataloader(
        dataset_with_car_mask,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=bev_cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=bev_cfg.data.nonshuffler_sampler,
    )

    data_loader_with_car_test = build_dataloader(
        dataset_with_car_test,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=bev_cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=bev_cfg.data.nonshuffler_sampler,
    )

    # build the bev model and load checkpoint
    bev_cfg.model.train_cfg = None
    bev_model = build_model(bev_cfg.model, test_cfg=bev_cfg.get('test_cfg'))
    fp16_cfg = bev_cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(bev_model)
    checkpoint = load_checkpoint(bev_model, opt.bev_checkpoint, map_location='cpu')
    if opt.fuse_conv_bn:
        bev_model = fuse_conv_bn(bev_model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        bev_model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        bev_model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        bev_model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        bev_model.PALETTE = dataset.PALETTE

    if not distributed:
        # assert False
        # model = MMDataParallel(model, device_ids=[0])
        # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
        bev_model = MMDataParallel(
            bev_model.cuda(),
            device_ids=[torch.cuda.current_device()]
        )
        bev_model.eval()
    else:
        bev_model = MMDistributedDataParallel(
            bev_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=True)
        bev_model.eval()

        time.sleep(2)  # This line can prevent deadlock problem in some cases.

    # load the rgb with car nuScenes object
    datapath = 'dataset_withcar'
    nusc = NuScenes(version='v1.0-mini', dataroot=datapath, verbose=True)

    # load the mask of rgb with car nuScenes object
    mask_datapath = 'dataset_mask_withcar'
    mask_nusc = NuScenes(version='v1.0-mini', dataroot=mask_datapath, verbose=True)

    # ---------------------------------#
    # -------Load 3D model-------------#
    # ---------------------------------#

    ## Pytorch3D Loading
    objfile='car_pytorch3d_last_E2E_output_different/car_pytorch3d_last_E2E_output_forward_-z_up_y/pytorch3d_Etron.obj'
    device = 'cuda'
    verts, faces, aux = load_obj(objfile)
    tex_maps = aux.texture_images
    image_origin = None
    if tex_maps is not None and len(tex_maps) > 0:
        verts_uvs = aux.verts_uvs.to(device)  # (V, 2)
        faces_uvs = faces.textures_idx.to(device)  # (F, 3)
        image_origin = list(tex_maps.values())[0].to(device)[None]
        print(f"image_origin.shape:{image_origin.shape}")
        tex = TexturesUV(
            verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image_origin
        )

    mask_image_dir="car_pytorch3d_last_E2E_output_different/car_pytorch3d_last_E2E_output_forward_-z_up_y/mask.png"
    mask_image = Image.open(mask_image_dir)#.convert("L")

    mask_image = (np.transpose(np.array(mask_image)[:,:,:3],(0,1,2))/255).astype('uint8')
    mask_image = torch.from_numpy(mask_image).to(device).unsqueeze(0)

    image_origin = image_origin.clone().detach()

    mesh = Meshes(
        verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=tex
    )
    R, T = look_at_view_transform(2.7, 0, 180)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    os.makedirs('rendered_images', exist_ok=True)

    view_angle_dict={
        "CAM_FRONT": 70,
        "CAM_FRONT_RIGHT": 70,
        "CAM_FRONT_LEFT": 70,
        "CAM_BACK_RIGHT": 70,
        "CAM_BACK_LEFT": 70,
        "CAM_BACK": 110,
    }

    # random initialize image_optim
    image_optim = torch.rand_like(image_origin).to(device).requires_grad_(True)

    # Setup Optimizer
    optim = torch.optim.Adam([image_optim], lr=opt.lr)
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    # ---------------------------------#
    # ------------Training-------------#
    # ---------------------------------#
    model_nsr=U_Net()

    # load the NSR model and load checkpoint
    saved_state_dict = torch.load('NRP_weights_no_car_paint/car/model_nsr_l39.pth')

    # setup sensors order
    sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    new_state_dict = {}
    for k, v in saved_state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    saved_state_dict = new_state_dict
    model_nsr.load_state_dict(saved_state_dict)
    model_nsr.to(device)
    model_ROA=ROA()
    model_ROA.to(device)
    model_ROA.eval()
    epoch_start=1+opt.continueFrom

    blur = 0
    raster_settings = RasterizationSettings(
        image_size=[1600, 1600],
        blur_radius=blur,
        faces_per_pixel=1,
        bin_size=0,
    )
    lights = AmbientLights(device=device)
    renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
    shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )
    renderer.to(device)
    renderer.eval()

    epoch_adv_loss = []
    epoch_masked_bev_loss = []
    epoch_smooth_loss = []
    epoch_total_loss = []

    for epoch in range(epoch_start, epochs+1):  # epoch ------------------------------------------------------------------

        print(f"Epoch {epoch}/{epochs}")

        model_nsr.eval()

        image_optim_in = mix_image(image_optim, mask_image, image_origin)

        tex = TexturesUV(
            verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image_optim_in
            )
        mesh.textures = tex

        for i, (data_with_car, data_without_car, data_with_car_mask, data_with_car_test) in enumerate(tqdm(zip(data_loader_with_car, data_loader_without_car, data_loader_with_car_mask, data_loader_with_car_test), desc="Processing batches", total=len(data_loader_with_car))):

            print(f"Starting to process sample {i}")

            sample_idx = data_with_car['img_metas'][0].data[0][0]['sample_idx']

            data_with_car_sample_info = next((info for info in data_with_car_annotations['infos'] if info['token'] == sample_idx), None)

            if data_with_car_sample_info is not None and 'gt_boxes' in data_with_car_sample_info:
                data_with_car_gt_boxes = data_with_car_sample_info['gt_boxes']
                data_with_car_gt_names = data_with_car_sample_info['gt_names']

            # get the current sample from data with car
            current_sample = nusc.get('sample', sample_idx)

            mask_sample_idx = data_with_car_mask['img_metas'][0].data[0][0]['sample_idx']

            # get the current sample from the mask of data with car
            mask_current_sample = mask_nusc.get('sample', mask_sample_idx)

            # initialize the multiview images
            multiview_images = []

            # check if there are any vehicle annotations in the current sample
            has_vehicle_annotations = False
            for ann_token in current_sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                if 'vehicle' in ann['category_name']:
                    has_vehicle_annotations = True
                    break

            if not has_vehicle_annotations:
                print(f"Warning: No vehicle annotations found in sample {sample_idx}, skipping this sample")
                continue  # Skip the entire sample processing

            valid_views = 0  # Count valid views

            # without_car_multiview_images = []

            for sensor in sensors:
                fov=view_angle_dict[sensor]
                cam_data = nusc.get('sample_data', current_sample['data'][sensor])
                filename = cam_data["filename"]
                bg_img_path = os.path.normpath(os.path.join(datapath, filename))
                bg_img_np = cv2.imread(bg_img_path, cv2.IMREAD_UNCHANGED)

                if bg_img_np is None:
                    raise ValueError(f"cannot load background image: {filename}")

                bg_img_rgb = cv2.cvtColor(bg_img_np[:,:,:3], cv2.COLOR_BGR2RGB)

                mask_cam_data = mask_nusc.get('sample_data', mask_current_sample['data'][sensor])
                mask_filename = mask_cam_data["filename"]
                mask_img_path = os.path.normpath(os.path.join(mask_datapath, mask_filename))
                mask_img_np = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

                if mask_img_np is None:
                    raise ValueError(f"cannot load mask image: {mask_filename}")

                _, mask_img_binary = cv2.threshold(mask_img_np, 127, 255, cv2.THRESH_BINARY)

                mask_img_rgb = np.stack([mask_img_binary, mask_img_binary, mask_img_binary], axis=2) / 255.0

                car_ref_rgb = bg_img_rgb * mask_img_rgb

                # Resize car_ref_rgb from (1600, 1600, 3) to (1, 3, 640, 640) for NSR model
                car_ref_rgb_tensor = torch.from_numpy(car_ref_rgb).to(device, non_blocking=True).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                car_ref_rgb_resized = F.interpolate(car_ref_rgb_tensor, size=(640, 640), mode='bilinear', align_corners=False)

                out_tensor = model_nsr(car_ref_rgb_resized)
                sig = nn.Sigmoid()
                out_tensor=sig(out_tensor)

                tensor1 = out_tensor[:,0:3, :, :]
                tensor2 = out_tensor[:,3:6, :, :]
                # Save the rendered image as a PNG file
                output_dir = "rendered_images"
                os.makedirs(output_dir, exist_ok=True)

                ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
                calibrated_sensor = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                vehicle_annotations = []
                for ann_token in current_sample['anns']:
                    ann = nusc.get('sample_annotation', ann_token)
                    if 'vehicle' in ann['category_name']:
                        vehicle_annotations.append(ann)

                if not vehicle_annotations:
                    print(f"Warning: No vehicle annotations found in sensor {sensor} for sample {sample_idx}, skipping this view")
                    continue

                print(f"find {len(vehicle_annotations)}  vehicle class annotations")
                sample_annotation=vehicle_annotations[0]
                renderer_vehicle_tranlation = sample_annotation["translation"]
                renderer_vehicle_rotation = sample_annotation["rotation"]

                global_from_ego = transform_matrix(ego_pose['translation'], Quaternion(ego_pose["rotation"]), inverse=False)
                ego_from_sensor = transform_matrix(calibrated_sensor["translation"], Quaternion(calibrated_sensor["rotation"]), inverse=False)

                ego_from_sensor[0:3, 0:3] = ego_from_sensor[0:3, 0:3]

                target_from_global = transform_matrix(renderer_vehicle_tranlation, Quaternion(renderer_vehicle_rotation), inverse=True)

                pose = target_from_global.dot(global_from_ego.dot(ego_from_sensor))
                cameras, Rotation, T = convert_nuscenes_to_pytorch3d(pose, device,fov,1)

                imgs_pred = renderer(mesh, cameras=cameras)

                # obtain the rendered image from pytorch3d
                imgs_rgb = imgs_pred[0,...,:3].clone()

                # Reshape imgs_rgb from shape (900, 900, 3) to torch tensor with shape (1, 3, 640, 640)
                imgs_rgb_tensor = imgs_rgb.permute(2, 0, 1).unsqueeze(0)  # Convert to (1, 3, 900, 900)
                imgs_rgb_resized = F.interpolate(imgs_rgb_tensor, size=(640, 640), mode='bilinear', align_corners=False)

                tensor3 = torch.clamp(imgs_rgb_resized*tensor1+tensor2,0,1)

                tensor3_resized = F.interpolate(tensor3, size=(1600, 1600), mode='bilinear', align_corners=False)
                tensor3_reshaped = tensor3_resized.squeeze(0).permute(1, 2, 0)

                # save the pytorch3d output
                output_filename = f"pytorch3d_{i}_{sensor}.png"
                output_path = os.path.join(output_dir, output_filename)
                imgs_rgb_np = imgs_rgb.detach().cpu().numpy() * 255
                imgs_rgb_np = imgs_rgb_np.astype(np.uint8)
                Image.fromarray(imgs_rgb_np).save(output_path)

                bg_img_tensor = torch.from_numpy(bg_img_rgb).to(device).float() / 255.0

                imgs_mask_3d = torch.from_numpy(mask_img_rgb).to(device).float()

                imgs_rgb_combined = (1 - imgs_mask_3d) * bg_img_tensor + imgs_mask_3d * tensor3_reshaped

                aspect_ratio = 16 / 9

                # Calculate the new height for image1 to match the aspect ratio
                new_height = int(imgs_rgb_combined.shape[1] / aspect_ratio)

                # Crop image1 to the new height, centered vertically
                top = (imgs_rgb_combined.shape[0] - new_height) // 2
                bottom = top + new_height
                cropped_imgs_rgb_combined = imgs_rgb_combined[top:bottom, :, :]

                resized_imgs_rgb_combined = cropped_imgs_rgb_combined

                # Save the combined image as a PNG file
                combined_output_filename = f"final_combined_pytorch3d_{i}_{sensor}.png"
                combined_output_path = os.path.join(output_dir, combined_output_filename)
                imgs_rgb_combined_np = resized_imgs_rgb_combined.detach().cpu().numpy() * 255
                imgs_rgb_combined_np = imgs_rgb_combined_np.astype(np.uint8)
                Image.fromarray(imgs_rgb_combined_np).save(combined_output_path)
                # shaping from H, W, C to C, H, W
                imgs_rgb_in = resized_imgs_rgb_combined.permute(2, 0, 1)

                imgs_bgr_in = imgs_rgb_in[[2, 1, 0], :, :]

                multiview_images.append(imgs_bgr_in)
                # without_car_multiview_images.append(without_car_bg_img_in)

                if multiview_images and len(multiview_images) > valid_views:
                    valid_views += 1

            if valid_views < len(sensors):
                print(f"Warning: Sample {sample_idx} does not have enough valid views (only {valid_views}/{len(sensors)}), skipping this sample")
                continue

            stacked_multiview_images = torch.stack(multiview_images, dim=0)

            final_mutliview_images = stacked_multiview_images.unsqueeze(0)

            data_with_car_test_copy = {}

            for key in data_with_car_test:
                if isinstance(data_with_car_test[key], list):
                    data_with_car_test_copy[key] = []
                    for item in data_with_car_test[key]:
                        if hasattr(item, 'clone'):
                            data_with_car_test_copy[key].append(item.clone())
                        else:
                            data_with_car_test_copy[key].append(item)
                else:
                    data_with_car_test_copy[key] = data_with_car_test[key]

            normalized_images = normalize_multiview_images(final_mutliview_images * 255, mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
            print(f"Normalized multiview shape: {normalized_images.shape}")

            padded_normalized_images = pad_tensor_to_multiple(normalized_images, divisor=32, pad_value=0)

            print(f"Padded multiview shape: {padded_normalized_images.shape}")

            # data_with_car_copy['img'][0].data[0] = normalized_images.clone()
            data_with_car_test_copy['img'][0].data[0] = padded_normalized_images.clone()

            # Visualize final_multiview_images
            final_multiview_images_np = final_mutliview_images.squeeze(0).permute(0, 2, 3, 1).detach().cpu().numpy() * 255
            final_multiview_images_np = final_multiview_images_np.astype(np.uint8)

            # Create a figure with 2 rows and 3 columns
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            sensor_order = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

            for idx, ax in enumerate(axes.flat):
                if idx < len(sensor_order):
                    ax.imshow(final_multiview_images_np[idx])
                    ax.set_title(sensor_order[idx])
                    ax.axis('off')

            # Save the visualization as a PNG file
            visualization_dir = os.path.join(log_dir, "multiview_visualizations")
            os.makedirs(visualization_dir, exist_ok=True)
            visualization_output_path = os.path.join(visualization_dir, f"multiview_rendered_imgs_sample_{i}.png")
            plt.savefig(visualization_output_path)
            plt.close()

            num_views = normalized_images.shape[1]
            if hasattr(data_with_car_test_copy['img_metas'][0], 'data'):

                for b in range(len(data_with_car_test_copy['img_metas'])):
                    data_with_car_test_copy['img_metas'][b].data[0] = data_with_car_test_copy['img_metas'][b].data[0][:num_views]

            print(f"Shape of the multiview image tensor for data_with_car: {data_with_car_test_copy['img'][0].data[0].shape}")

            print(f"Shape of the transformed multiview image tensor for data_without_car: {data_without_car['img'][0].data[0].shape}")
            # # Forward pass for data without car

            # with torch.no_grad():
            result_with_car, bev_tensor_with_car = bev_model(return_loss=False, rescale=True, **data_with_car_test_copy)

            with torch.no_grad():

                result_without_car, bev_tensor_without_car = bev_model(return_loss=False, rescale=True, **data_without_car)

            bev_difference = torch.abs(bev_tensor_with_car - bev_tensor_without_car)

            def process_bev_tensor(tensor):

                processed = tensor.view(200, 200, 1, 256).mean(dim=-1, keepdim=False).clone()
                return processed.detach().cpu().numpy()

            bev_difference_np = process_bev_tensor(bev_difference)

            threshold_bev_value = 0.3

            new_threshold_bev_value = 0.25

            bev_difference = bev_difference.view(200, 200, 1, 256).mean(dim=-1, keepdim=False).clone()

            thresholded_bev_difference = torch.where(bev_difference > threshold_bev_value, bev_difference, torch.zeros_like(bev_difference))

            thresholded_bev_difference_np = np.where(bev_difference_np > threshold_bev_value, bev_difference_np, 0)

            # visualize the gt_boxes of the car
            print(f"Ground truth bounding boxes for sample {sample_idx}: {data_with_car_gt_boxes}")
            print(f"Ground truth class names for sample {sample_idx}: {data_with_car_gt_names}")

            # Visualize the ground truth bounding boxes in BEV view
            bev_gt_visual_dir = os.path.join(log_dir, "bev_gt_visual")
            os.makedirs(bev_gt_visual_dir, exist_ok=True)

            # Create a blank BEV grid
            bev_gt_grid = np.zeros((200, 200))

            # Perception range and resolution
            perception_range = [-51.2, 51.2]
            resolution = 0.512

            # Visualize the BEV grid with ground truth bounding boxes
            plt.figure(figsize=(10, 8))
            plt.imshow(bev_gt_grid, cmap='viridis')
            plt.colorbar(label='Ground Truth Intensity')
            plt.title('Ground Truth Bounding Boxes in BEV View')
            # Mark the center of the bounding boxes with a red circle
            for box, name in zip(data_with_car_gt_boxes, data_with_car_gt_names):
                if name == "car":
                    x, y = box[:2]
                    grid_x = int((x - perception_range[0]) / resolution)
                    grid_y = int((y - perception_range[0]) / resolution)
                    if 0 <= grid_x < 200 and 0 <= grid_y < 200:
                        plt.plot(grid_x, grid_y, 'ro')  # Red circle notation

            # Mark the ego vehicle with an 'x' notation
            ego_x, ego_y = 0.0, 0.0  # Assuming ego vehicle is at (0, 0) in world coordinates
            ego_grid_x = int((ego_x - perception_range[0]) / resolution)
            ego_grid_y = int((ego_y - perception_range[0]) / resolution)
            if 0 <= ego_grid_x < 200 and 0 <= ego_grid_y < 200:
                plt.plot(ego_grid_x, ego_grid_y, 'bx')  # Blue 'x' notation

            # calcuate the distance between the ego vehicle and the car
            grid_r = math.sqrt((grid_x - ego_grid_x)**2 + (grid_y - ego_grid_y)**2)

            # Calculate the change in coordinates
            delta_x = grid_x - ego_grid_x
            delta_y = grid_y - ego_grid_y

            # Calculate grid_theta (the angle between the points)
            grid_theta_rad = math.atan2(delta_y, delta_x)
            grid_theta_deg = math.degrees(grid_theta_rad)

            # If it's negative, wrap it into the positive range
            if grid_theta_deg < 0:
                grid_theta_deg += 360
            # Ensure grid_theta_deg is between 0 and 360 degrees
            grid_theta_deg = grid_theta_deg % 360

            print(grid_r)
            print(grid_theta_deg)

            r_limit_value = 15
            theta_limit_value = 15

            # Define the lower and upper limits based on grid_r and the threshold
            r_lower_limit = 0
            r_upper_limit = grid_r + r_limit_value

            r_limit_values = (r_lower_limit, r_upper_limit)

            theta_lower_limit, theta_upper_limit = calculate_theta_limits(grid_theta_deg, theta_limit_value)

            theta_limit_values = (theta_lower_limit, theta_upper_limit)

            mask = create_mask(grid_r, grid_theta_deg, r_limit_values, theta_limit_values)

            mask_visual_dir = os.path.join(log_dir, "mask_visual")
            os.makedirs(mask_visual_dir, exist_ok=True)
            # Convert the numpy array to a PIL Image (assuming it's a binary mask)
            mask_visual = Image.fromarray(mask.squeeze() * 255)  # Multiply by 255 for visibility
            mask_visual_output_filename = f"mask_visual_sample_{i}.png"
            mask_visual_output_path = os.path.join(mask_visual_dir, mask_visual_output_filename)

                # Save the image as a PNG
            mask_visual.save(mask_visual_output_path)

            # Assuming mask is a numpy ndarray containing binary values (0s and 1s)
            mask_tensor = torch.tensor(mask, dtype=torch.uint8)  # Use torch.uint8 for binary values

            # Now, move the tensor to the device (CPU or GPU)
            mask_tensor = mask_tensor.to(device)

            masked_thresholded_bev_difference = torch.where(mask_tensor, thresholded_bev_difference, torch.zeros_like(thresholded_bev_difference))

            # masked_bev_loss = mean_of_non_zero(bev_difference)

            masked_bev_loss = mean_of_non_zero(masked_thresholded_bev_difference)

            # masked_bev_loss = mean_of_non_zero(new_masked_thresholded_bev_difference)

            # masked_bev_loss = mean_of_non_zero(combined_masked_thresholded_bev_difference)

            print(f"Masked BEV Loss: {masked_bev_loss.item()}")

            masked_thresholded_bev_difference_np = masked_thresholded_bev_difference.detach().cpu().numpy()
            # new_masked_thresholded_bev_difference_np = new_masked_thresholded_bev_difference.detach().cpu().numpy()
            # combined_masked_thresholded_bev_difference_np = combined_masked_thresholded_bev_difference.detach().cpu().numpy()

            def create_visualization_dir(log_dir, dir_name):

                visual_dir = os.path.join(log_dir, dir_name)
                os.makedirs(visual_dir, exist_ok=True)
                return visual_dir

            def save_image_with_plt(data, log_dir, dir_name, filename, cmap='viridis', vmin=None, vmax=None):

                visual_dir = create_visualization_dir(log_dir, dir_name)
                plt.figure(figsize=(10, 8))
                ax = plt.gca()
                plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
                plt.colorbar(label='Difference Intensity')
                output_path = os.path.join(visual_dir, filename)
                plt.savefig(output_path)
                plt.close()

            def save_bev_tensor_visualization(bev_tensor, log_dir, dir_name, filename):

                bev_tensor_view = bev_tensor.view(200, 200, 1, 256).mean(dim=-1, keepdim=False).clone()
                bev_tensor_np = bev_tensor_view.detach().cpu().numpy()

                visual_dir = create_visualization_dir(log_dir, dir_name)
                plt.figure(figsize=(10, 8))
                ax = plt.gca()
                plt.imshow(bev_tensor_np, cmap='viridis')
                plt.colorbar(label='BEV Tensor Intensity')
                output_path = os.path.join(visual_dir, filename)
                plt.savefig(output_path)
                plt.close()

            save_image_with_plt(
                data=masked_thresholded_bev_difference_np,
                log_dir=log_dir,
                dir_name="masked_thresholded_bev_diff_visual",
                filename=f"masked_thresholded_bev_diff_sample_{i}.png",
                cmap='viridis'
            )

            # Save the visualization as a PNG file
            bev_gt_output_filename = f"bev_gt_boxes_sample_{i}.png"
            bev_gt_output_path = os.path.join(bev_gt_visual_dir, bev_gt_output_filename)
            plt.savefig(bev_gt_output_path)
            plt.close()

            plt.figure(figsize=(10, 8))
            ax = plt.gca()
            plt.imshow(thresholded_bev_difference_np, cmap='viridis')
            plt.colorbar(label='Thresholded and Normalized Difference Intensity')
            thresholded_bev_diff_output_path = os.path.join(
                create_visualization_dir(log_dir, "thresholded_bev_diff_visual"),
                f"thresholded_bev_diff_sample_{i}.png"
            )
            plt.savefig(thresholded_bev_diff_output_path)
            plt.close()

            save_bev_tensor_visualization(bev_tensor_without_car, log_dir, "bev_tensor_without_car_visual",
                                        f"bev_tensor_without_car_{i}.png")

            save_bev_tensor_visualization(bev_tensor_with_car, log_dir, "bev_tensor_with_car_visual",
                                        f"bev_tensor_with_car_{i}.png")

            save_image_with_plt(bev_difference_np, log_dir, "bev_diff_visual",
                                f"raw_bev_diff_sample_{i}.png", cmap='viridis', vmin=0, vmax=1)

            # Calculate the loss value based on bev_difference using Mean Squared Error (MSE)
            adv_loss = torch.mean(torch.square(bev_difference))
            print(f"BEV Adv Loss: {adv_loss.item()}")

            smooth_loss = loss_smooth_UVmap(image_optim, mask_image)
            print(f"Smooth Loss: {smooth_loss.item()}")

            λ_smooth = 10000

            loss = masked_bev_loss +  λ_smooth * smooth_loss

            epoch_masked_bev_loss.append(masked_bev_loss.item())
            epoch_adv_loss.append(adv_loss.item())
            epoch_smooth_loss.append(smooth_loss.item())
            epoch_total_loss.append(loss.item())

            print(f"Total Loss: {loss.item()}")

            print(f"image_optim before step:{torch.sum(image_optim)},device:{device}")

            optim.zero_grad()
            loss.backward(retain_graph=False) #retain_graph=True
            print(f"image_optim grad: {torch.sum(image_optim.grad)}")
            optim.step()

            print(f"image_optim after step:{torch.sum(image_optim)},device:{device}")

            image_optim_in = mix_image(image_optim, mask_image, image_origin)

            tex = TexturesUV(
                verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image_optim_in
                )
            mesh.textures = tex
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

        if len(epoch_adv_loss) > 0:
            avg_adv_loss = sum(epoch_adv_loss) / len(epoch_adv_loss)
            avg_masked_bev_loss = sum(epoch_masked_bev_loss) / len(epoch_adv_loss)
            avg_smooth_loss = sum(epoch_smooth_loss) / len(epoch_smooth_loss)
            avg_total_loss = sum(epoch_total_loss) / len(epoch_total_loss)

            if tb_writer:
                tb_writer.add_scalar("Loss/adv_loss", avg_adv_loss, epoch)
                tb_writer.add_scalar("Loss/masked_bev_loss", avg_masked_bev_loss, epoch)
                tb_writer.add_scalar("Loss/smooth_loss", avg_smooth_loss, epoch)
                tb_writer.add_scalar("Loss/total_loss", avg_total_loss, epoch)

                current_lr = optim.param_groups[0]['lr']
                tb_writer.add_scalar("Learning_rate", current_lr, epoch)

                logger.info(f"Epoch {epoch} - Avg Adv Loss: {avg_adv_loss:.6f}, Avg Smooth Loss: {avg_smooth_loss:.6f}, Avg Total Loss: {avg_total_loss:.6f}")

        if epoch % 1 == 0:
            image_optim_in_np = (255 * image_optim_in.squeeze(0).detach().cpu().numpy()).astype('uint8')
            output_path = os.path.join(log_dir, f'texture_{epoch}.png')
            Image.fromarray(image_optim_in_np).save(output_path)
    tb_writer.close()
    torch.cuda.empty_cache()
    # return results
    return

log_dir = ""
def make_log_dir(logs):
    global log_dir
    dir_name = ""
    for key in logs.keys():
        dir_name += str(key) + '-' + str(logs[key]) + '+'
    # dir_name = 'logs_FCA_no_car_paint_withROA_5_5/' + dir_name
    dir_name = 'logs/' + dir_name
    print(dir_name)
    if not (os.path.exists(dir_name)):
        os.makedirs(dir_name)
    log_dir = dir_name

if __name__ == '__main__':
    print(f"logger{logger}")
    parser = argparse.ArgumentParser()
    # hyperparameter for training adversarial camouflage
    # ------------------------------------#
    parser.add_argument('--weights', type=str, default='yolov3_9_5.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/carla.yaml', help='data.yaml path')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for texture_param')
    parser.add_argument('--obj_file', type=str, default='car_asset_E2E/pytorch3d_Etron.obj', help='3d car model obj')
    parser.add_argument('--faces', type=str, default='car_assets/exterior_face.txt',
                        help='exterior_face file  (exterior_face, all_faces)')
    parser.add_argument('--datapath', type=str, default='car_train_total_no_paper/adversarialtrain',
                            help='data path')
    parser.add_argument('--patchInitial', type=str, default='random',
                        help='data path')
    parser.add_argument('--device', default='6', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--lamb", type=float, default=1e-4) #lambda
    parser.add_argument("--d1", type=float, default=0.9)
    parser.add_argument("--d2", type=float, default=0.1)
    parser.add_argument("--t", type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=20)

    # ------------------------------------#
    parser.add_argument('--bev_config', default='./projects/configs/bevformer/bevformer_base.py', help='test config file path')
    parser.add_argument('--bev_checkpoint', default='ckpts/bevformer_r101_dcn_24ep.pth', help='checkpoint file')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        default='bbox',
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    #add
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--classes', nargs='+', type=int, default=[2],
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--continueFrom', type=int, default=0, help='continue from which epoch')
    parser.add_argument('--texturesize', type=int, default=6, help='continue from which epoch')
    opt = parser.parse_args()

    T = opt.t
    D1 = opt.d1
    D2 = opt.d2
    lamb = opt.lamb
    LR = opt.lr
    Dataset=opt.datapath.split('/')[-1]
    PatchInitial=opt.patchInitial
    logs = {
        'date': time.strftime('%Y-%m-%d-%H-%M-%S'),
        'epoch': opt.epochs,
        'texturesize':opt.texturesize,
        'patchInitialWay':PatchInitial,
        'batch_size': opt.batch_size,
        'UV_map': "2048",
        'loss_type': 'bev_mask',
        'λ_smooth': 10000
    }

    make_log_dir(logs)

    texture_dir_name = ''
    for key, value in logs.items():
        texture_dir_name+= f"{key}-{str(value)}+"

    # Set DDP variables

    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    print('WORLD_SIZE' in os.environ)
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        check_git_status()

    # Resume
    # wandb_run = check_wandb_resume(opt)
    # if opt.resume and not wandb_run:  # resume an interrupted run   ``
    if opt.resume:  # resume an interrupted run   ``
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f) )  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = \
            '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve))
    opt.total_batch_size=opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    print(f"device:{device}")
    if opt.local_rank != -1:
        msg = 'is not compatible with YOLOv3 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo')
    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps
    # Train
    logger.info(opt)

    tb_writer = None  # init loggers
    if opt.global_rank in [-1, 0]:
        prefix = colorstr('tensorboard: ')
        logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
        tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
    train(hyp, opt, device)


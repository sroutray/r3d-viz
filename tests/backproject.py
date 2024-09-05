import torch
import numpy as np
import scipy
import random
from torch_scatter import scatter_mean
from torch.nn import functional as F
import tests.libs.pointops2.functions.pointops as pointops
from volumentations.augmentations.functional import rotate_around_axis, scale
import math

# from mask2former_video.utils.util_3d import plot_only_3d
# import matplotlib.pyplot as plt

import ipdb
st = ipdb.set_trace


def elastic_distortion(pointcloud, granularity, magnitude, scannet_pc=None):
    """Apply elastic distortion on sparse coordinate space.

    pointcloud: numpy array of (number of points, at least 3 spatial dims)
    granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: noise multiplier
  """

    if scannet_pc is not None:
        pc = scannet_pc
    else:
        pc = pointcloud

    blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
    blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
    blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
    coords = pc[:, :3]
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
        noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(
            coords_min - granularity,
            coords_min + granularity * (noise_dim - 2),
            noise_dim,
        )
    ]
    interp = scipy.interpolate.RegularGridInterpolator(
        ax, noise, bounds_error=0, fill_value=0
    )
    pointcloud[:, :3] = pointcloud[:, :3] + interp(pointcloud[:, :3]) * magnitude
    
    if scannet_pc is not None:
        scannet_pc[:, :3] = scannet_pc + interp(scannet_pc) * magnitude
    return pointcloud, scannet_pc


def rotate_axis_custom(pc, rotation_limit, axis, scannet_pc=None):
    angle = random.uniform(rotation_limit[0], rotation_limit[1])
    if scannet_pc is not None:
        center_point = scannet_pc[:, :3].mean(axis=0).astype(scannet_pc[:, :3].dtype)
        scannet_pc = rotate_around_axis(scannet_pc, axis, angle, center_point)
    else:
        center_point = pc[:, :3].mean(axis=0).astype(pc[:, :3].dtype)

    pc = rotate_around_axis(pc, axis, angle, center_point)

    return pc, scannet_pc


def rotation_augmentations(pc, scannet_pc=None):
    axis = np.array([0, 0, 1])
    rotation_limit = (-3.14, 3.14)
    pc, scannet_pc = rotate_axis_custom(pc, rotation_limit, axis, scannet_pc)

    axis = np.array([0, 1, 0])
    rotation_limit = (-0.1308, 0.1308)
    pc, scannet_pc = rotate_axis_custom(pc, rotation_limit, axis, scannet_pc)

    axis = np.array([1, 0, 0])
    rotation_limit = (-0.1308, 0.1308)
    pc, scannet_pc = rotate_axis_custom(pc, rotation_limit, axis, scannet_pc)

    return pc, scannet_pc

def scale_augmentations(pc, scannet_pc=None):
    scale_ = []
    limit = [0.9, 1.1]
    for _ in range(3):
        scale_.append(random.uniform(limit[0], limit[1]))
    
    pc = scale(pc, scale_).astype(pc.dtype)

    if scannet_pc is not None:
        scannet_pc = scale(scannet_pc, scale_).astype(scannet_pc.dtype)

    return pc, scannet_pc


def downsample_depth_map(depth_map, H_new, W_new):
    """
    Downsamples a batch of depth maps by taking a random value within each patch.

    Args:
        depth_map: torch.Tensor of shape (batch_size, 3, H, W) representing the batch of depth maps
        H_new: int, the new height of the downsampled depth maps
        W_new: int, the new width of the downsampled depth maps

    Returns:
        A torch.Tensor of shape (batch_size, 3, H_new, W_new) representing the downsampled depth maps.
    """
    # Determine the patch size based on the desired downsampled size
    patch_size = max(depth_map.shape[2] // H_new, depth_map.shape[3] // W_new)

    # Pad the depth maps to make them evenly divisible by patch_size
    pad_H = depth_map.shape[2] % patch_size
    pad_W = depth_map.shape[3] % patch_size
    depth_map = torch.nn.functional.pad(depth_map, (0, pad_W, 0, pad_H))

    # Rearrange the depth maps into patches
    B, _, H, W = depth_map.shape
    patches = torch.nn.functional.unfold(
        depth_map, kernel_size=patch_size,
        stride=patch_size).reshape(B, 3, patch_size**2, H_new, W_new)
    
    # Compute the number of elements in each patch
    num_elements = (patch_size ** 2)

    # Sample a random index within each patch
    rand_indices = torch.randint(high=num_elements, size=(B, 1, 1, H_new, W_new)).to(patches)

    # Gather the values at the random indices for each patch
    downsampled_depth_maps = torch.gather(patches, dim=2, index=rand_indices.repeat(1, 3, 1, 1, 1).long()).squeeze(2)
    return downsampled_depth_maps


def augment_depth(xyz, scannet_pc=None):
    """
    xyz: B X V X H X W X 3
    """
    B, V, H, W, _ = xyz.shape
    
    # convert to a pointcloud
    xyz = xyz.reshape(B, V*H*W, 3)

    # mean center
    mean = xyz.mean(1, keepdims=True)
    xyz -= mean

    if scannet_pc is not None:
        scannet_pc = [scannet_pc[i] - mean[i] for i in range(B)]
    
    # # add unform noise between xyz.min(1) and xyz.max(1)
    noise = torch.from_numpy(
        np.random.uniform(xyz.min(1, keepdims=True)[0].cpu().numpy(),
         xyz.max(1, keepdims=True)[0].cpu().numpy())).to(xyz) / 2.0

    xyz += noise
    if scannet_pc is not None:
        scannet_pc = [scannet_pc[i] + noise[i] for i in range(B)]

    # Add uniform noise between xyz.min(1) and xyz.max(1)
    # st()
    # min_xyz, _ = torch.min(xyz, dim=1, keepdim=True)
    # max_xyz, _ = torch.max(xyz, dim=1, keepdim=True)
    # noise = torch.rand_like(xyz, device=xyz.device) * (max_xyz - min_xyz) + min_xyz
    # xyz += noise / 2.0

    # if scannet_pc is not None:
    #     scannet_pc = [scannet_pc[i] + noise[i] / 2.0 for i in range(B)]

    for i in (0, 1):
        if np.random.rand() > 0.5:
            xyz_max = xyz[..., i].max(dim=1, keepdim=True)[0]
            xyz[..., i] = xyz_max - xyz[..., i]

            if scannet_pc is not None:
                for j in range(B):
                    scannet_pc[j][:, i] = xyz_max[j] - scannet_pc[j][:, i]
                # scannet_pc = [xyz_max[j] - scannet_pc[j, :, i] for j in range(B)]
                # scannet_pc[..., i] = xyz_max - scannet_pc[..., i]

    # # add elastic distortion
    if np.random.rand() < 0.95:
        for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
            all_xyz, all_scannet_pc = [], []
            for i in range(B):
                xyz_, scannet_pc_ = elastic_distortion(
                    xyz[i].cpu().numpy(), granularity, magnitude, scannet_pc[i].cpu().numpy()
                )
                all_xyz.append(xyz_)
                all_scannet_pc.append(torch.from_numpy(scannet_pc_).to(xyz))
            xyz = torch.from_numpy(np.stack(all_xyz)).to(xyz)
            scannet_pc = all_scannet_pc

    xyz = xyz.reshape(B, V, H, W, 3)
    return xyz, scannet_pc


def augment_depth_numpy(xyz, scannet_pc=None, do_flipping=True, do_elastic_distortion=True):
    """
    xyz: B X V X H X W X 3
    """
    B, V, H, W, _ = xyz.shape
    assert B == 1

    # convert to a pointcloud
    xyz = xyz.reshape(B, V*H*W, 3)
    
    # mean center
    if scannet_pc is not None:
        mean = scannet_pc.mean(1, keepdims=True)
        scannet_pc -= mean
    else:
        mean = xyz.mean(1, keepdims=True)

    xyz -= mean
    
    # # add unform noise between xyz.min(1) and xyz.max(1)
    if scannet_pc is not None:
        noise = np.random.uniform(scannet_pc.min(1, keepdims=True),
            scannet_pc.max(1, keepdims=True)) / 2.0
        scannet_pc += noise
    else:
        noise = np.random.uniform(xyz.min(1, keepdims=True),
            xyz.max(1, keepdims=True)) / 2.0

    xyz += noise

    if do_flipping:
        for i in (0, 1):
            if np.random.rand() > 0.5:
                if scannet_pc is not None:
                    xyz_max = scannet_pc[..., i].max(1, keepdims=True)
                    scannet_pc[..., i] = xyz_max - scannet_pc[..., i]
                else:
                    xyz_max = xyz[..., i].max(1, keepdims=True)
                xyz[..., i] = xyz_max - xyz[..., i]

    if scannet_pc is not None:
        scannet_pc = scannet_pc[0]

    xyz = xyz[0]

    # # add elastic distortion
    if do_elastic_distortion:
        if np.random.rand() < 0.95:
            for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
                xyz, scannet_pc = elastic_distortion(
                    xyz, granularity, magnitude, scannet_pc
                )

    xyz, scannet_pc = rotation_augmentations(xyz, scannet_pc)
    xyz, scannet_pc = scale_augmentations(xyz, scannet_pc)

    xyz = xyz.reshape(B, V, H, W, 3)
    return xyz, scannet_pc

  
def unproject(intrinsics, poses, depths, mask_valid=True):
    """
    Inputs:
        intrinsics: B X V X 3 X 3
        poses: B X V X 4 X 4 (torch.tensor)
        depths: B X V X H X W (torch.tensor)
    
    Outputs:
        world_coords: B X V X H X W X 3 (all valid 3D points)
        valid: B X V X H X W (bool to indicate valid points)
                can be used to index into RGB images
                to get N X 3 valid RGB values
    """
    B, V, H, W = depths.shape
    fx, fy, px, py = intrinsics[..., 0, 0][..., None], intrinsics[..., 1, 1][..., None], intrinsics[..., 0, 2][..., None], intrinsics[..., 1, 2][..., None]

    y = torch.arange(0, H).to(depths.device)
    x = torch.arange(0, W).to(depths.device)
    y, x = torch.meshgrid(y, x)

    x = x[None, None].repeat(B, V, 1, 1).flatten(2)
    y = y[None, None].repeat(B, V, 1, 1).flatten(2)
    z = depths.flatten(2)
    x = (x - px) * z / fx
    y = (y - py) * z / fy
    cam_coords = torch.stack([
        x, y, z, torch.ones_like(x)
    ], -1)

    world_coords = (poses @ cam_coords.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
    world_coords = world_coords[..., :3] / world_coords[..., 3][..., None]

    world_coords = world_coords.reshape(B, V, H, W, 3)

    if mask_valid:
        world_coords[depths == 0] = -10
    return world_coords

def backproject_depth(depths, poses, intrinsics=None, mask_valid=True):
    B, V, H, W = depths.shape
    if intrinsics is None:
        print("CAUTION!!! USING DEFAULT INTRINSICS oF SCANNET!! MIGHT BE BAD!!")
        intrinsics = torch.from_numpy(
            get_scannet_intrinsic([H, W])).reshape(1, 1, 3, 3).repeat(B, V, 1, 1).cuda().to(depths.dtype)
    xyz = unproject(intrinsics, poses, depths, mask_valid=mask_valid)
    return xyz


def interpolate_depth(xyz, multi_scale_features, augment=False, method="nearest"):
    multi_scale_xyz = []
    B, V, H, W, _ = xyz.shape
    for feat in multi_scale_features:
        h, w = feat.shape[2:]
        xyz_ = torch.nn.functional.interpolate(
            xyz.reshape(B*V, H, W, 3).permute(0, 3, 1, 2), size=(h, w),
            mode=method).permute(0, 2, 3, 1).reshape(B, V, h, w, 3)
        multi_scale_xyz.append(xyz_.float())    
    return multi_scale_xyz


def backprojector(
    multi_scale_features, depths, poses,
    intrinsics=None, augment=False,
    return_original_xyz=False, 
    method='nearest', scannet_pc=None, mask_valid=True):
    """
    Inputs:
        multi_scale_features: list
            [B*V, 256, 15, 20], [B*V, 256, 30, 40], [B*V, 256, 60, 80]
        depths: tensor [B, 5, 480, 640]
        poses: tensor [B, 5, 4, 4]
        mask_features: [B, 5, 256, 120, 160]
        intrinsics: tensor [B, 5, 4, 4]

    Outputs:
        list: []
            B, V, H, W, 3
    """
    xyz = backproject_depth(depths, poses, intrinsics, mask_valid=mask_valid)
    
    if augment:
        new_xyz, scannet_pc = augment_depth(xyz, scannet_pc)
    else:
        new_xyz = xyz

    multi_scale_xyz = interpolate_depth(
        new_xyz, multi_scale_features,
        method=method)

    if return_original_xyz:
        print('returning original xyz')
        multi_scale_xyz_original = interpolate_depth(
            xyz, multi_scale_features, augment=False, method=method)
        return multi_scale_xyz, multi_scale_xyz_original
    
    return multi_scale_xyz, scannet_pc



def backprojector_dataloader(
    multi_scale_features, depths, poses,
    intrinsics=None, augment=False,
    method='nearest', scannet_pc=None, padding=None, do_flipping=True,
    mask_valid=True, do_elastic_distortion=True, mean_center=False):
    """
    Inputs:
        multi_scale_features: list
            [B*V, 256, 15, 20], [B*V, 256, 30, 40], [B*V, 256, 60, 80]
        depths: tensor [B, 5, 480, 640]
        poses: tensor [B, 5, 4, 4]
        mask_features: [B, 5, 256, 120, 160]
        intrinsics: tensor [B, 5, 4, 4]

    Outputs:
        list: []
            B, V, H, W, 3
    """
    xyz = backproject_depth(
        depths[None], poses[None], intrinsics[None], mask_valid=mask_valid)
    
    if mean_center:
        if scannet_pc is not None:
            mean = scannet_pc.mean(1, keepdims=True)
            scannet_pc -= mean
        else:
            mean = xyz.mean(1, keepdims=True)
        xyz -= mean
        
    if augment:
        new_xyz, scannet_pc = augment_depth_numpy(
            xyz.numpy(),
            scannet_pc[None].numpy() if scannet_pc is not None else None,
            do_flipping=do_flipping, 
            do_elastic_distortion=do_elastic_distortion)
        new_xyz = torch.from_numpy(new_xyz)
        if scannet_pc is not None:
            scannet_pc = torch.from_numpy(scannet_pc)

    else:
        new_xyz = xyz

    if padding is not None:
        # Investigate what to pad with
        new_xyz = F.pad(new_xyz.permute(0, 1, 4, 2, 3), (0, padding[1], 0, padding[0]), mode='constant', value=0).permute(0, 1, 3, 4, 2)

    multi_scale_xyz = interpolate_depth(
        new_xyz, multi_scale_features,
        method=method)

    multi_scale_xyz = [xyz.squeeze(0) for xyz in multi_scale_xyz]
    
    return multi_scale_xyz, scannet_pc, new_xyz


def multiscsale_voxelize(multi_scale_xyz, voxel_size):
    """
    Inputs: 
        multi_scale_xyz: list of tensors [B, V, H, W, 3]
        voxel_size: list of floats of len multi_scale_xyz

    Outputs:
        N=V*H*W
        multi_scale_unqiue_idx list of tensors [B, N]
        multi_scale_p2v: list of tensors [B, N]
        multi_scale_padding_mask: list of tensors [B, N]
    """
    multi_scale_p2v = []
    assert len(multi_scale_xyz) == len(voxel_size)
    for i, xyz in enumerate(multi_scale_xyz):
        B, V, H, W, _ = xyz.shape
        xyz = xyz.reshape(B, V*H*W, 3)
        point_to_voxel = voxelization(xyz, voxel_size[i])
        multi_scale_p2v.append(point_to_voxel)
    return multi_scale_p2v


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert len(arr.shape) == 3
    arr -= arr.min(1, keepdims=True)[0].to(torch.long)
    arr_max = arr.max(1, keepdims=True)[0].to(torch.long) + 1

    keys = torch.zeros(arr.shape[0], arr.shape[1], dtype=torch.long).to(arr.device)

    # Fortran style indexing
    for j in range(arr.shape[2] - 1):
        keys += arr[..., j]
        keys *= arr_max[..., j + 1]
    keys += arr[..., -1]
    return keys

def voxelization(xyz, voxel_size):
    """
    Inputs:
        xyz: tensor [B, N, 3]
        voxel_size: float
    Outputs: 
        point_to_voxel_all: tensor [B, N], is the mapping from original point cloud to voxel
    """
    B, N, _ = xyz.shape
    xyz = xyz / voxel_size
    xyz = torch.round(xyz).long()
    xyz = xyz - xyz.min(1, keepdim=True)[0]

    keys = ravel_hash_vec(xyz)

    point_to_voxel = torch.stack(
        [torch.unique(keys[b], return_inverse=True)[1] for b in range(B)], 0)
    return point_to_voxel


def prepare_feats_for_pointops(xyz, shape, feats=None, voxelize=False, p2v=None):
    bs, v = shape

    if len(xyz.shape) == 5:
        b, v, h, w, _ = xyz.shape
        if feats is not None:
            feats = feats.reshape(bs, v, feats.shape[1], h, w).permute(0, 1, 3, 4, 2).flatten(1, 3) # B, VHW, F
        xyz = xyz.reshape(bs, v, h, w, 3).flatten(1, 3) # B, VHW, 3

    if voxelize:
        if feats is not None:
            feats = torch.cat(
                [scatter_mean(feats[b], p2v[b], dim=0) for b in range(len(feats))]) # bn, F
        xyz = torch.cat(
            [scatter_mean(xyz[b], p2v[b], dim=0) for b in range(len(xyz))])
        batch_offset = ((p2v).max(1)[0] + 1).cumsum(0).to(torch.int32)
    else:
        # queryandgroup expects N, F and N, 3 with additional batch offset
        xyz = xyz.flatten(0, 1).contiguous()
        if feats is not None:
            feats = feats.flatten(0, 1).contiguous()
        batch_offset = (torch.arange(bs, dtype=torch.int32, device=xyz.device) + 1) * v * h * w

    return feats, xyz, batch_offset
    

def voxel_map_to_source(voxel_map, poin2voxel):
    """
    Input:
        voxel_map (B, N1, C)
        point2voxel (B, N)
    Output:
        src_new (B, N, C)
    """
    bs, n, c = voxel_map.shape
    src_new = torch.stack([voxel_map[i, poin2voxel[i]] for i in range(bs)])
    return src_new


def interpolate_feats_3d(
    source_feats, source_xyz,
    source_p2v, target_xyz,
    target_p2v, shape, num_neighbors=3, voxelize=False, 
    return_voxelized=False
    ):
    """
    Inputs:
        source_feats: tensor [B*V, C, H1, W1] or B, N, C
        source_xyz: tensor [B, V, H1, W1, 3] or B, N, 3
        source_p2v: tensor [B, N1]
        target_xyz: tensor [B, V, H2, W2, 3] or B, N2, 3
        target_p2v: tensor [B, N2]
    Outputs:
        target_feats: tensor [BV, C, H2, W2] or B, C, N2
    """

    source_feats_pops, source_xyz_pops, source_batch_offset = prepare_feats_for_pointops(
        xyz=source_xyz, shape=shape, feats=source_feats,
        voxelize=voxelize, p2v=source_p2v
    )
    _, target_xyz_pops, target_batch_offset = prepare_feats_for_pointops(
        xyz=target_xyz, shape=shape, feats=None,
        voxelize=voxelize, p2v=target_p2v)
    target_feats = pointops.interpolation(
        source_xyz_pops, target_xyz_pops, source_feats_pops,
        source_batch_offset, target_batch_offset, k=num_neighbors) # bn, C

    # undo voxelization
    if voxelize and not return_voxelized:
        out_new = []
        idx = 0
        for i, b in enumerate(target_batch_offset):
            out_new.append(target_feats[idx:b][target_p2v[i]])
            idx = b
        output = torch.stack(out_new, 0)
        if len(target_xyz.shape) == 5:
            bs, v, h, w, _ = target_xyz.shape
            output = output.reshape(bs, v, h, w, -1).flatten(0, 1).permute(0, 3, 1, 2) # BV, C, H, W
        else:
            output = output.permute(0, 2, 1)

    else:
        # just batch it
        if len(target_batch_offset) != 1:
            max_batch_size = (target_batch_offset[1:] - target_batch_offset[:-1]).max()
            max_batch_size = max(max_batch_size, target_batch_offset[0])
        else:    
            max_batch_size = target_batch_offset[0]
        output = torch.zeros(
            len(target_batch_offset), max_batch_size, target_feats.shape[1], device=target_feats.device)
        idx = 0
        for i, b in enumerate(target_batch_offset):
            output[i, :b - idx] = target_feats[idx:b]
            idx = b
        output = output.permute(0, 2, 1)
       
    return output


def get_scannet_intrinsic(image_size, intrinsic_file=None):
    scannet_intrinsic = np.array([[577.871,   0.       , 319.5, 0.],
                                  [  0.       , 577.871, 239.5, 0.],
                                  [  0.       ,   0.       ,   1., 0. ],
                                  [0., 0., 0., 1.0]
                                ])
    scannet_intrinsic[0] /= 480 / image_size[0]
    scannet_intrinsic[1] /= 640 / image_size[1]
    return scannet_intrinsic

def get_ai2thor_intrinsics(image_size, intrinsic_file=None):
    fov = 90
    hfov = float(fov) * np.pi / 180.
    H, W = image_size[:2]
    if H != W:
        assert False, "Ai2thor only supports square images"
    intrinsics = np.array([
            [(W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., (H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
    intrinsics[0,2] = W/2.
    intrinsics[1,2] = H/2.
    return intrinsics


def get_replica_intrinsics(image_size, intrinsic_file=None):
    H, W = image_size[:2]

    hfov = 90
    # the pin-hole camera has the same value for fx and fy
    fx = W / 2.0 / math.tan(math.radians(hfov / 2.0))
    # self.fy = self.H / 2.0 / math.tan(math.radians(self.yhov / 2.0))
    fy = fx
    cx = (W-1.0) / 2.0
    cy = (H-1.0) / 2.0

    intrinsics = np.array([
        [fx, 0., cx, 0.],
        [0., fy, cy, 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ])
    return intrinsics


def get_s3dis_intrinsics(image_size, intrinsic_file=None):
    H, W = image_size[:2]
    
    intrinsics = np.loadtxt(intrinsic_file).reshape(3, 3).astype(np.float32)
    intrinsics_ = np.eye(4).astype(np.float32)
    intrinsics_[:3, :3] = intrinsics    
    intrinsics = intrinsics_
    
    intrinsics[0] /= 1080 / H
    intrinsics[1] /= 1080 / W
    
    return intrinsics


def get_matterport_intrinsics(image_size, intrinsic_file=None):
    H, W = image_size[:2]
    
    intrinsics = np.loadtxt(intrinsic_file).reshape(3, 3).astype(np.float32)
    intrinsics_ = np.eye(4).astype(np.float32)
    intrinsics_[:3, :3] = intrinsics    
    intrinsics = intrinsics_
    
    intrinsics[0] /= 512 / H
    intrinsics[1] /= 640 / W
    
    return intrinsics
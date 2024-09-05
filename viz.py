import json
import os
from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R
from torch import Tensor

import liblzfse
import wandb
from jaxtyping import Bool, Float
from plyfile import PlyData
from tests import colors
from tests.backproject import backprojector

wandb.init(project="r3d-viz")
# pointcloud = torch.cat([world_coords, color_valid*255.0], dim=1).cpu().numpy()
# wandb.log({"point_cloud": wandb.Object3D(pointcloud)})


Colormaps = Literal[
    "default", "turbo", "viridis", "magma", "inferno", "cividis", "gray", "pca"
]


@dataclass(frozen=True)
class ColormapOptions:
    """Options for colormap"""

    colormap: Colormaps = "default"
    """ The colormap to use """
    normalize: bool = False
    """ Whether to normalize the input tensor image """
    colormap_min: float = 0
    """ Minimum value for the output colormap """
    colormap_max: float = 1
    """ Maximum value for the output colormap """
    invert: bool = False
    """ Whether to invert the output colormap """


def apply_depth_colormap(
    depth: Float[Tensor, "*bs 1"],
    accumulation: Optional[Float[Tensor, "*bs 1"]] = None,
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    colormap_options: ColormapOptions = ColormapOptions(),
) -> Float[Tensor, "*bs rgb=3"]:
    """Converts a depth image to color for easier analysis.

    Args:
        depth: Depth image.
        accumulation: Ray accumulation used for masking vis.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.
        colormap: Colormap to apply.

    Returns:
        Colored depth image with colors in [0, 1]
    """

    near_plane = (
        near_plane if near_plane is not None else float(torch.min(depth))
    )
    far_plane = far_plane if far_plane is not None else float(torch.max(depth))

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)
    # depth = torch.nan_to_num(depth, nan=0.0) # TODO(ethan): remove this

    colored_image = apply_colormap(depth, colormap_options=colormap_options)

    if accumulation is not None:
        colored_image = colored_image * accumulation + (1 - accumulation)

    return colored_image


def apply_colormap(
    image: Float[Tensor, "*bs channels"],
    colormap_options: ColormapOptions = ColormapOptions(),
    eps: float = 1e-9,
) -> Float[Tensor, "*bs rgb=3"]:
    """
    Applies a colormap to a tensor image.
    If single channel, applies a colormap to the image.
    If 3 channel, treats the channels as RGB.
    If more than 3 channel, applies a PCA reduction on the dimensions to 3 channels

    Args:
        image: Input tensor image.
        eps: Epsilon value for numerical stability.

    Returns:
        Tensor with the colormap applied.
    """

    # default for rgb images
    if image.shape[-1] == 3:
        return image

    # rendering depth outputs
    if image.shape[-1] == 1 and torch.is_floating_point(image):
        output = image
        if colormap_options.normalize:
            output = output - torch.min(output)
            output = output / (torch.max(output) + eps)
        output = (
            output
            * (colormap_options.colormap_max - colormap_options.colormap_min)
            + colormap_options.colormap_min
        )
        output = torch.clip(output, 0, 1)
        if colormap_options.invert:
            output = 1 - output
        return apply_float_colormap(output, colormap=colormap_options.colormap)

    # rendering boolean outputs
    if image.dtype == torch.bool:
        return apply_boolean_colormap(image)

    if image.shape[-1] > 3:
        return apply_pca_colormap(image)

    raise NotImplementedError


def apply_boolean_colormap(
    image: Bool[Tensor, "*bs 1"],
    true_color: Float[Tensor, "*bs rgb=3"] = colors.WHITE,
    false_color: Float[Tensor, "*bs rgb=3"] = colors.BLACK,
) -> Float[Tensor, "*bs rgb=3"]:
    """Converts a depth image to color for easier analysis.

    Args:
        image: Boolean image.
        true_color: Color to use for True.
        false_color: Color to use for False.

    Returns:
        Colored boolean image
    """

    colored_image = torch.ones(image.shape[:-1] + (3,))
    colored_image[image[..., 0], :] = true_color
    colored_image[~image[..., 0], :] = false_color
    return colored_image


def apply_pca_colormap(
    image: Float[Tensor, "*bs dim"],
    pca_mat: Optional[Float[Tensor, "dim rgb=3"]] = None,
    ignore_zeros=True,
) -> Float[Tensor, "*bs rgb=3"]:
    """Convert feature image to 3-channel RGB via PCA. The first three principle
    components are used for the color channels, with outlier rejection per-channel

    Args:
        image: image of arbitrary vectors
        pca_mat: an optional argument of the PCA matrix, shape (dim, 3)
        ignore_zeros: whether to ignore zero values in the input image (they won't affect the PCA computation)

    Returns:
        Tensor: Colored image
    """
    original_shape = image.shape
    image = image.view(-1, image.shape[-1])
    if ignore_zeros:
        valids = (image.abs().amax(dim=-1)) > 0
    else:
        valids = torch.ones(image.shape[0], dtype=torch.bool)

    if pca_mat is None:
        _, _, pca_mat = torch.pca_lowrank(image[valids, :], q=3, niter=20)
    assert pca_mat is not None
    image = torch.matmul(image, pca_mat[..., :3])
    d = torch.abs(
        image[valids, :] - torch.median(image[valids, :], dim=0).values
    )
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    m = 2.0  # this is a hyperparam controlling how many std dev outside for outliers
    rins = image[valids, :][s[:, 0] < m, 0]
    gins = image[valids, :][s[:, 1] < m, 1]
    bins = image[valids, :][s[:, 2] < m, 2]

    image[valids, 0] -= rins.min()
    image[valids, 1] -= gins.min()
    image[valids, 2] -= bins.min()

    image[valids, 0] /= rins.max() - rins.min()
    image[valids, 1] /= gins.max() - gins.min()
    image[valids, 2] /= bins.max() - bins.min()

    image = torch.clamp(image, 0, 1)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return image.view(*original_shape[:-1], 3)


def apply_float_colormap(
    image: Float[Tensor, "*bs 1"], colormap: Colormaps = "viridis"
) -> Float[Tensor, "*bs rgb=3"]:
    """Convert single channel to a color image.

    Args:
        image: Single channel image.
        colormap: Colormap for image.

    Returns:
        Tensor: Colored image with colors in [0, 1]
    """
    if colormap == "default":
        colormap = "turbo"

    image = torch.nan_to_num(image, 0)
    if colormap == "gray":
        return image.repeat(1, 1, 3)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return torch.tensor(
        matplotlib.colormaps[colormap].colors, device=image.device
    )[image_long[..., 0]]


def load_depth(filepath: os.path, size: tuple[int, int]) -> np.ndarray:
    with open(filepath, "rb") as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)

    depth_img = depth_img.reshape(size)  # For a FaceID camera 3D Video
    # depth_img = depth_img.reshape((256, 192))  # For a LiDAR 3D Video

    return depth_img


def load_img(
    filename: os.path, *, use_rgb: bool = True, **kwargs
) -> np.ndarray:
    img = cv2.imread(filename, **kwargs)
    if use_rgb and img.shape[-1] >= 3:
        # Take care of RGBA case when flipping.
        img = np.concatenate([img[..., 2::-1], img[..., 3:]], axis=-1)
    return img


def load_ply(filepath):
    # import ipdb; ipdb.set_trace()
    with open(filepath, "rb") as f:
        plydata = PlyData.read(f)
    data = plydata.elements[0].data
    coords = np.array([data["x"], data["y"], data["z"]], dtype=np.float32).T
    feats = None
    labels = None
    if ({"red", "green", "blue"} - set(data.dtype.names)) == set():
        feats = np.array(
            [data["red"], data["green"], data["blue"]], dtype=np.uint8
        ).T
    if "label" in data.dtype.names:
        labels = np.array(data["label"], dtype=np.uint32)
    return coords, feats, labels


def get_mat_from_pose(pose):
    matrix = np.eye(4)
    if len(pose) == 6:
        px, py, pz, r, p, y = pose
        matrix[:3, :3] = R.from_euler(
            "xyz", [r, p, y], degrees=False
        ).as_matrix()
    else:
        qx, qy, qz, qw, px, py, pz = pose
        matrix[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    matrix[:3, -1] = [px, py, pz]
    return matrix


def plot_3d(xdata, ydata, zdata, color=None, b_min=2, b_max=8, view=(45, 45)):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=200)
    ax.view_init(view[0], view[1])

    # ax.set_xlim(b_min, b_max)
    # ax.set_ylim(b_min, b_max)
    # ax.set_zlim(b_min, b_max)
    ax.set_xlim(xdata.min(), xdata.max())
    ax.set_ylim(ydata.min(), ydata.max())
    ax.set_zlim(zdata.min(), zdata.max())

    ax.scatter3D(xdata, ydata, zdata, c=color, cmap="rgb", s=0.1)


def save_log(depth_frame, rgb_frame, pcd_frame, pcd_unproj, near_plane = None, far_plane = None):
    depth_frame = depth_frame.unsqueeze(-1)
    depth_img = apply_depth_colormap(depth_frame, near_plane=near_plane, far_plane=far_plane)
    log_dict = {
        "depth": wandb.Image(depth_img.permute(2, 0, 1)),
        "rgb": wandb.Image(rgb_frame.permute(2, 0, 1)),
        "pcd": wandb.Object3D(pcd_frame),
        "pcd_unproj": wandb.Object3D(pcd_unproj)
    }
    wandb.log(log_dict)


def process_video(video_path: os.path, log: bool = False, f_skip: int = 1):
    # Read metadata
    with open(os.path.join(video_path, "metadata.json"), "r") as f:
        # Load JSON data from file
        metadata = json.load(f)

    fps = metadata["fps"]
    num_frames = len(metadata["poses"])
    num_viz_frames = len(metadata["poses"][0::f_skip])
    depth_h = metadata["dh"]
    depth_w = metadata["dw"]
    img_h = metadata["h"]
    img_w = metadata["w"]
    frame_poses = metadata["poses"][0::f_skip]
    frame_pose_mats = np.array(
        [get_mat_from_pose(pose) for pose in frame_poses]
    )
    # frame_pose_mats = np.array(
    #     [np.linalg.inv(get_mat_from_pose(pose)) for pose in frame_poses]
    # )
    cam_int = np.asarray(metadata["K"]).reshape((3, 3)).T
    frame_cam_int = metadata["perFrameIntrinsicCoeffs"][0::f_skip]
    frame_cam_int = [
        np.asarray([K[0], 0, 0, 0, K[1], 0, K[2], K[3], 1]).reshape((3, 3)).T
        for K in frame_cam_int
    ]
    # cam_int = np.linalg.inv(cam_int)

    depth_frames = np.zeros(
        (num_viz_frames, depth_h, depth_w), dtype=np.float32
    )
    rgb_frames = np.zeros(
        (num_viz_frames, depth_h, depth_w, 3), dtype=np.uint8
    )
    viz_frame_idx = 0
    for frame_idx in range(0, num_frames, f_skip):
        # Load depth image
        depth_frames[viz_frame_idx] = load_depth(
            os.path.join(video_path, "rgbd", f"{frame_idx}.depth"),
            (depth_h, depth_w),
        )
        rgb_frame = load_img(
            os.path.join(video_path, "rgbd", f"{frame_idx}.jpg"),
            flags=cv2.IMREAD_UNCHANGED,
        )
        rgb_frame = cv2.resize(
            rgb_frame, (depth_w, depth_h), interpolation=cv2.INTER_AREA
        )
        rgb_frames[viz_frame_idx] = rgb_frame
        viz_frame_idx += 1

    # Convert to tensors
    depth_frames = torch.tensor(depth_frames)
    near_plane = float(torch.min(depth_frames))
    far_plane = float(torch.max(depth_frames))
    rgb_frames = torch.tensor(rgb_frames / 255.0)
    frame_pose_mats = torch.tensor(frame_pose_mats)

    # Load point cloud
    print("Initial pose:", metadata["initPose"])
    print("Initial pose mat:", get_mat_from_pose(metadata["initPose"]))

    viz_frame_idx = 0
    pcd_unproj_all = torch.zeros((num_viz_frames*depth_h*depth_w, 6), dtype=torch.float64)
    for frame_idx in range(0, num_frames, f_skip):
        depth = depth_frames[viz_frame_idx]
        feats = [torch.randn((1, 3, depth_h, depth_w))]
        pose = frame_pose_mats[viz_frame_idx]
        cam_int_hom = torch.eye(4, dtype=torch.float64)
        cam_int_hom[:3, :3] = torch.tensor(frame_cam_int[viz_frame_idx])
        # Adjust for resizing
        cam_int_hom[0] /= img_h / depth_h
        cam_int_hom[1] /= img_w / depth_w 
        world_coords = backprojector(
            feats,
            depth[None, None],
            # torch.eye(4, dtype=torch.float64)[None, None],
            pose[None, None],
            intrinsics=cam_int_hom[None, None],
        )[0]
        world_coords = world_coords[0][:, 0]
        world_coords = world_coords.reshape(-1, 3)
        x_data, y_data, z_data = world_coords.cpu().unbind(-1)

        color_valid = rgb_frames[viz_frame_idx].reshape(-1, 3)
        pcd_unproj = np.concatenate([world_coords.numpy(), 255.0 * color_valid.numpy()], 1)
        pcd_unproj_all[viz_frame_idx*depth_h*depth_w:(viz_frame_idx+1)*depth_h*depth_w] = torch.tensor(pcd_unproj, dtype=torch.float64)
        plot_3d(
            x_data.numpy(),
            z_data.numpy(),
            y_data.numpy(),
            color=color_valid.cpu().numpy(),
            view=(45,-45)
        )

        plt.savefig("sensor_depth.png")
        plt.close()

        pcd_coords, pcd_colors, _ = load_ply(
            os.path.join(video_path, "pcd", f"{frame_idx:07d}.ply")
        )
        pcd_coords = torch.tensor(pcd_coords, dtype=torch.float64)
        pcd_colors = torch.tensor(pcd_colors)
        # Transform point cloud to world frame
        pcd_coords = torch.matmul(pcd_coords, pose[:3, :3].T)

        pcd_all = torch.cat([pcd_coords, pcd_colors], dim=1)
        plot_3d(
            pcd_coords[:, 0].numpy(),
            pcd_coords[:, 2].numpy(),
            pcd_coords[:, 1].numpy(),
            color=pcd_colors.numpy() / 255.0,
        )
        plt.savefig("pcd.png")
        plt.close()
        # import ipdb; ipdb.set_trace()
        if log:
            save_log(depth, rgb_frames[viz_frame_idx], pcd_all.numpy(), pcd_unproj, near_plane, far_plane)
        viz_frame_idx += 1

    wandb.log({"pcd_unproj_all": wandb.Object3D(pcd_unproj_all.numpy())})

data_root = "tests/test_vids"
# wandb.init(project='r3d-viz')

for vid in os.listdir(data_root):
    vid_path = os.path.join(data_root, vid)
    print(f"Processing: {vid}")
    process_video(vid_path, log=True, f_skip=15)

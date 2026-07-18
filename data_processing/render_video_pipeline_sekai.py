"""
csv contains two kinds of crowdDensity
To solve the problem of reflection when multiple ellipsoids intersect, the core reason is that the ellipsoid is too large. Just use --ellipsoid_scale_factor.
Complete video processing and 3D rendering pipeline - full version
===========================================

Process overview:
1. Use Grounded SAM2 to process the video and obtain the object mask
2. Use the depth, intrinsics, and extrinsics in the npz file to obtain the point cloud of each frame
3. Use the object mask and point cloud of each frame in the first two steps to fit the 3D gaussian and calculate the 3D bounding box.
4. Insert the ellipsoid corresponding to the 3D gaussian into the point cloud of the first frame to construct a dynamic scene
5. Insert the four prisms corresponding to the 3D bounding box into the point cloud of the first frame to build a dynamic scene
6. Extract the center point of each frame object and connect it into a 3D trajectory line segment to construct a trajectory scene.
7-15. Render three scenes in three modes (background + foreground, foreground only, background only)
16. Generate 3D Gaussian projection to 2D visualization video
17. Output auxiliary video (background mask, object id, etc.)
Solution: The combination of gaussian projection and bg does not consider aplha transparency and depth, and mask does not consider depth either.
"""

import logging
import argparse
import os
import shutil
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from time import time

import torch
import numpy as np
import cv2
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from kornia.geometry.depth import depth_to_3d_v2
from scipy.stats import chi2

from pytorch3d.structures import Pointclouds, Meshes, join_meshes_as_batch
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    HardPhongShader,
    TexturesVertex,
    PointLights,
)
from pytorch3d.utils import ico_sphere
from torchvision.io import write_video, read_video
import torchvision.transforms.functional as TF

from grounded_sam2_video_run_demo import run_video_segmentation, load_models_and_init

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================================
# Auxiliary function: Calculation of ellipsoid scaling factors from probabilities
# ============================================================================

def compute_ellipsoid_scale_factor_from_probability(probability: float, df: int = 3) -> float:
    """
    Compute ellipsoid scaling factor (ellipsoid_scale_factor) from Gaussian probabilities
    
    For a 3D Gaussian distribution, the square of the Mahalanobis distance follows the chi-squared distribution,
    The degrees of freedom are 3 (3 dimensions).
    
    Args:
        probability: Gaussian probability (between 0-1), for example 0.97 means that 97% of the probability mass is within the ellipsoid
        df: degrees of freedom, 3 for 3D Gaussian
    
    Returns:
        scale_factor: scaling factor, used to scale the ellipsoid semi-axis
        
    Example:
        scale_factor = compute_ellipsoid_scale_factor_from_probability(0.97)  # ≈ 2.991
        scale_factor = compute_ellipsoid_scale_factor_from_probability(0.95)  # ≈ 2.795
        scale_factor = compute_ellipsoid_scale_factor_from_probability(0.99)  # ≈ 3.368
    """
    return np.sqrt(chi2.ppf(probability, df=df))


# ============================================================================
# Parameter analysis
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Complete video processing and 3D rendering pipeline"
    )
    parser.add_argument(
        '--csv_path', type=str,
        default="/path/to/filtered_sekai_by_grounded_sam2.csv",
        help='CSV file with video and npz paths'
    )
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument(
        '--video_root', type=str,
        default="/path/to/sekai_videos_root/",
        help='Root directory where videos and npz files are stored'
    )
    parser.add_argument(
        '--output_root', type=str, default='outputs/complete_pipeline',
        help='Root directory for outputs'
    )
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--num_parts', type=int, default=1, help='Number of parts to split data')
    parser.add_argument('--part_idx', type=int, default=0, help='Part index to process')
    parser.add_argument(
        '--gaussian_mask_threshold', type=float, default=0.003,
        help='Threshold for 2D gaussian mask'
    )
    parser.add_argument(
        '--obb_scale_factor', type=float, default=2.0,
        help='OBB scale factor (1.0=68.3%%, 2.0=95.4%%, 3.0=99.7%% coverage)'
    )
    parser.add_argument('--point_size', type=float, default=0.005, help='Point size for rendering')
    parser.add_argument(
        '--ellipsoid_samples', type=int, default=2000,
        help='[Deprecated when using mesh] Number of points to sample from ellipsoid'
    )
    parser.add_argument('--ellipsoid_subdiv', type=int, default=3, help='Icosphere subdivisions for ellipsoid mesh (2-4 is typical)')
    parser.add_argument(
        '--ellipsoid_scale_factor', 
        type=float, 
        default=None,
        help='Scale factor for ellipsoid half-axes. If None, will be computed from --ellipsoid_probability. '
             'For 3D Gaussian: 2.991≈97%%, 2.795≈95%%, 3.368≈99%% probability mass inside ellipsoid.'
    )
    parser.add_argument(
        '--ellipsoid_probability',
        type=float,
        default=0.8,
        help='Gaussian probability mass inside ellipsoid (0-1). Used to compute ellipsoid_scale_factor if not explicitly set. Default: 0.97 (97%%)'
    )
    parser.add_argument(
        '--trajectory_line_thickness', type=int, default=3,
        help='Thickness of trajectory lines in pixels'
    )
    parser.add_argument(
        '--trajectory_radius', type=float, default=0.03,
        help='Base radius for calculating trajectory line thickness (multiplied by 100)'
    )
    parser.add_argument(
        '--max_frames', type=int, default=81,
        help='Maximum number of frames to process per video'
    )
    parser.add_argument('--fps', type=int, default=10, help='Output video FPS')
    parser.add_argument(
        '--render_batch_size', type=int, default=27,
        help='Batch size for rendering (higher values can speed up rendering but use more memory)'
    )
    parser.add_argument(
        '--use_fp16', action='store_true',
        help='Use FP16 (half precision) for rendering to speed up (may slightly affect quality)'
    )
    parser.add_argument(
        '--pin_memory', action='store_true',
        help='Use pinned memory for faster GPU transfers'
    )
    parser.add_argument(
        '--mask_erode_kernel', type=int, default=5,
        help='Kernel size for mask erosion to remove boundary noise (0 to disable)'
    )
    parser.add_argument(
        '--mask_erode_iterations', type=int, default=1,
        help='Number of erosion iterations for mask boundary noise removal'
    )
    parser.add_argument(
        '--trajectory_saturation_factor', type=float, default=1.3,
        help='Saturation enhancement factor for trajectory videos (1.0=no change, >1=more saturated)'
    )
    parser.add_argument(
        '--trajectory_contrast_factor', type=float, default=1.1,
        help='Contrast enhancement factor for trajectory videos (1.0=no change, >1=more contrast)'
    )
    return parser.parse_args()


# ============================================================================
# Step 1: Grounded SAM2 Segmentation
# ============================================================================

def run_grounded_sam2_segmentation(
    video_path: str,
    output_dir: Path,
    grounding_model,
    video_predictor,
    image_predictor,
    device: str,
    max_frames: int = 81,
    text_prompt: str = "person . car ."
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Step 1: Use Grounded SAM2 to segment the video
    
    Returns:
        video_segments: {frame_idx: {obj_id: mask (H,W)}}
    """
    logger.info("=" * 80)
    logger.info("Step 1: Run Grounded SAM2 object segmentation")
    logger.info("=" * 80)
    
    save_tracking_results_dir = output_dir / "annotated_frames"
    temp_frame_dir = output_dir / "custom_video_frames_temp"
    tracking_video = output_dir / "output_tracking_demo.mp4"
    
    video_segments = run_video_segmentation(
        video_path=video_path,
        grounding_model=grounding_model,
        video_predictor=video_predictor,
        image_predictor=image_predictor,
        device=device,
        text_prompt=text_prompt,
        visualize=True,
        save_tracking_results_dir=str(save_tracking_results_dir),
        output_video_path=str(tracking_video),
        temp_frame_dir=str(temp_frame_dir),
        max_frames=max_frames
    )
    
    logger.info(f"✓ Segmentation complete, processed {len(video_segments)} frames")
    return video_segments


# ============================================================================
# Step 2: Point cloud generation
# ============================================================================

def get_point_cloud_from_depth_cuda(
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor
) -> torch.Tensor:
    """
    Step 2: Generate point cloud in world coordinate system from depth map (GPU accelerated version)
    
    Args:
        depth: (H, W) depth map
        intrinsic: (3, 3) camera intrinsic parameters
        extrinsic: (4, 4) camera external parameters (world to camera)
    
    Returns:
        pts3d_world: (H*W, 3) point cloud in world coordinate system
    """
    # Generate 3D points in the camera coordinate system
    pts3d_cam = depth_to_3d_v2(depth, intrinsic, normalize_points=False).reshape(-1, 3)
    
    # Convert to world coordinate system
    c2w = torch.linalg.inv(extrinsic)
    pts3d_hom = torch.cat([pts3d_cam, torch.ones(len(pts3d_cam), 1, device=pts3d_cam.device)], dim=1)
    pts3d_world = (c2w @ pts3d_hom.T).T[:, :3]
    
    return pts3d_world


def extract_object_point_clouds_gpu(
    point_cloud: torch.Tensor,
    masks: Dict[int, np.ndarray],
    depth: torch.Tensor,
    device: str = 'cuda',
    erode_kernel_size: int = 5,
    erode_iterations: int = 1
) -> Dict[int, torch.Tensor]:
    """
    Extract the point cloud of each object according to the mask from the complete point cloud (GPU accelerated version)
    
    Args:
        point_cloud: (H*W, 3) complete point cloud
        masks: {object_id: (H, W) mask}
        depth: (H, W) depth map
        device: device
        erode_kernel_size: Kernel size for corrosion operation (0 means no corrosion)
        erode_iterations: Number of corrosion iterations
    
    Returns:
        object_pcs: {object_id: (N_obj, 3) point cloud}
    """
    H, W = depth.shape
    object_pcs = {}
    
    if isinstance(point_cloud, np.ndarray):
        pc_gpu = torch.from_numpy(point_cloud).float().to(device)
    else:
        pc_gpu = point_cloud
    
    for obj_id, mask in masks.items():
        # Uniform conversion to numpy array for processing
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask
        
        # Processing mask dimensions
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        if mask_np.shape != (H, W):
            # Make sure it is uint8 type for resize
            mask_uint8 = mask_np.astype(np.uint8) if mask_np.dtype != np.uint8 else mask_np
            mask_np = cv2.resize(
                mask_uint8, (W, H),
                interpolation=cv2.INTER_NEAREST
            )
        
        # Corrode mask to remove boundary noise
        if erode_kernel_size > 0 and erode_iterations > 0:
            # Make sure mask is of type uint8
            mask_uint8 = mask_np.astype(np.uint8) if mask_np.dtype != np.uint8 else mask_np
            
            # Create a corrosion kernel
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (erode_kernel_size, erode_kernel_size)
            )
            
            # Perform corrosion operations
            mask_eroded = cv2.erode(mask_uint8, kernel, iterations=erode_iterations)
            
            # Check if there are enough pixels left after erosion
            if np.sum(mask_eroded) > 10:  # Keep at least 10 pixels
                mask_np = mask_eroded
            # else: keep the original mask to avoid completely disappearing
        
        # Convert to boolean type and extract point cloud
        mask_flat = mask_np.reshape(-1).astype(bool)
        pc = pc_gpu[mask_flat]
        
        if len(pc) > 0:
            object_pcs[obj_id] = pc
    
    return object_pcs


def get_background_pointcloud_from_masks(
    masks: Dict[int, np.ndarray],
    pc: torch.Tensor,
    first_frame_rgb_tensor: torch.Tensor,
    H: int, W: int,
    device: str = 'cuda',
    kernel_size: int = 9,
    iterations: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get background point cloud and color from masks (inflate foreground mask to ensure full coverage)
    
    Returns:
        bg_pts: (N_bg, 3) background point cloud
        bg_colors: (N_bg, 3) background color (0-255)
        bg_mask: (H, W) background mask
    """
    # Merge all foreground masks
    combined = torch.zeros((H, W), dtype=torch.bool, device=device)
    for _, m in masks.items():
        mt = m
        if mt.ndim == 3:
            mt = mt[0]
        if mt.shape != (H, W):
            mt_np = cv2.resize(
                mt.cpu().numpy().astype(np.uint8), (W, H),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            mt = torch.from_numpy(mt_np).to(device)
        combined |= mt
    
    # Inflate foreground mask
    if kernel_size > 1:
        arr = combined.cpu().numpy().astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        arr = cv2.dilate(arr, kernel, iterations=iterations)
        fg_mask = torch.from_numpy(arr.astype(bool)).to(device)
    else:
        fg_mask = combined
    
    # background = ~foreground
    bg_mask = ~fg_mask
    bg_pts = pc[bg_mask.reshape(-1)]
    bg_colors = first_frame_rgb_tensor[bg_mask]
    
    return bg_pts, bg_colors, bg_mask


# ============================================================================
# Step 3: 3D Gaussian fitting and OBB calculation
# ============================================================================

def fit_3d_gaussian_gpu(
    points: torch.Tensor,
    use_consistent_shape: bool = False,
    reference_covariance: Optional[torch.Tensor] = None,
    device: str = 'cuda'
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Step 3: Fit 3D Gaussian distribution (GPU accelerated version)
    
    Args:
        points: (N, 3) point cloud
        use_consistent_shape: Whether to keep the shape consistent (using the reference covariance matrix)
        reference_covariance: (3, 3) reference covariance matrix
    
    Returns:
        mean: (3,) mean
        cov: (3, 3) covariance matrix
    """
    if len(points) < 4:
        return None, None
    
    if isinstance(points, np.ndarray):
        pts = torch.from_numpy(points).float().to(device)
    else:
        pts = points
    
    mean = torch.mean(pts, dim=0)
    centered = pts - mean
    cov = (centered.T @ centered) / (pts.size(0) - 1)
    
    # Keep the shape of the reference covariance (only update position and orientation)
    if use_consistent_shape and reference_covariance is not None:
        if isinstance(reference_covariance, np.ndarray):
            ref_cov = torch.from_numpy(reference_covariance).float().to(device)
        else:
            ref_cov = reference_covariance.to(device)
        
        # Using the eigenvalues ​​of the reference covariance, the eigenvectors of the current covariance
        # ref_eigenvalues, _ = torch.linalg.eigh(ref_cov)
        # _, curr_eigenvectors = torch.linalg.eigh(cov)
        # cov = curr_eigenvectors @ torch.diag(ref_eigenvalues) @ curr_eigenvectors.T
        cov = ref_cov
    return mean, cov


def compute_obb_from_gaussian(
    mean: torch.Tensor,
    cov: torch.Tensor,
    scale_factor: float = 2.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Step 3: Calculate OBB directly from 3D Gaussian parameters (GPU version)
    
    Args:
        mean: (3,) Gaussian mean
        cov: (3, 3) covariance matrix
        scale_factor: scaling factor (1.0=68.3%, 2.0=95.4%, 3.0=99.7% coverage)
    
    Returns:
        center: (3,) OBB center point
        extents: (3,) The half length of the three axes of OBB
        rotation: (3, 3) OBB rotation matrix
    """
    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean).float()
    if isinstance(cov, np.ndarray):
        cov = torch.from_numpy(cov).float()
    
    center = mean
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    
    # Sort by feature value size
    idx = torch.argsort(eigenvalues, descending=True)
    rotation = eigenvectors[:, idx]
    sorted_eigenvalues = eigenvalues[idx]
    
    # Calculate extent (half length)
    extents = scale_factor * torch.sqrt(torch.clamp(sorted_eigenvalues, min=1e-8))
    
    return center, extents, rotation


# ============================================================================
# Visualization function of 3D Gaussian projection to 2D
# ============================================================================

def compute_probability_density_map_gpu(
    means_3d: torch.Tensor,
    covs_3d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    image_size: Tuple[int, int],
    device='cuda'
) -> torch.Tensor:
    """
    GPU version of probability density map calculation - projecting 3D Gaussian to 2D image plane
    
    Args:
        means_3d: (N, 3) 3D Gaussian mean
        covs_3d: (N, 3, 3) 3D covariance matrix
        K: (3, 3) camera intrinsic parameter matrix
        R: (3, 3) rotation matrix (world to camera)
        t: (3, 1) or (3,) translation vector
        image_size: (width, height)
        device: 'cuda' or 'cpu'
    
    Returns:
        density_map: (height, width) probability density map
    """
    width, height = image_size
    
    # Create pixel coordinate grid on GPU
    u_coords = torch.arange(width, device=device, dtype=torch.float32)
    v_coords = torch.arange(height, device=device, dtype=torch.float32)
    u_grid, v_grid = torch.meshgrid(u_coords, v_coords, indexing='xy')
    pixel_coords = torch.stack([u_grid, v_grid], dim=-1)  # (H, W, 2)
    
    # Initialize density map
    density_map = torch.zeros((height, width), device=device, dtype=torch.float32)
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    num_gaussians = means_3d.shape[0]
    
    for i in range(num_gaussians):
        mean_3d_world = means_3d[i]
        cov_3d_world = covs_3d[i]
        
        # Convert to camera coordinate system
        if t.dim() == 2:
            t_vec = t.squeeze()
        else:
            t_vec = t
        mean_3d_cam = R @ mean_3d_world + t_vec
        cov_3d_cam = R @ cov_3d_world @ R.T
        
        x, y, z = mean_3d_cam[0], mean_3d_cam[1], mean_3d_cam[2]
        
        # Skip points behind camera
        if z <= 0:
            continue
            
        # Calculate the Jacobian matrix
        J = torch.tensor([
            [fx / z, 0, -fx * x / (z * z)],
            [0, fy / z, -fy * y / (z * z)]
        ], device=device, dtype=torch.float32)
        
        # 2D parameters after projection
        mean_2d = torch.tensor([
            fx * x / z + cx,
            fy * y / z + cy
        ], device=device, dtype=torch.float32)
        
        cov_2d = J @ cov_3d_cam @ J.T
        cov_2d += torch.eye(2, device=device) * 1e-6  # regularization
        
        try:
            # Multivariate normal distribution PDF calculation on GPU
            diff = pixel_coords - mean_2d  # (H, W, 2)
            diff_flat = diff.reshape(-1, 2)  # (H*W, 2)
            
            # Compute the inverse covariance matrix
            cov_inv = torch.linalg.inv(cov_2d)
            det_cov = torch.det(cov_2d)
            
            # Calculate Mahalanobis distance
            mahal_dist = torch.sum((diff_flat @ cov_inv) * diff_flat, dim=1)  # (H*W,)
            
            # Calculate PDF
            coeff = 1.0 / (2 * torch.pi * torch.sqrt(det_cov))
            pdf_values = coeff * torch.exp(-0.5 * mahal_dist)
            pdf_map = pdf_values.reshape(height, width)
            
            density_map += pdf_map
            
        except Exception as e:
            logger.warning(f"Skipping Gaussian {i} due to error: {e}")
            continue
            
    return density_map


def project_gaussian_to_2d_gpu(
    mean: torch.Tensor,
    cov: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    image_size: Tuple[int, int],
    device='cuda'
) -> Tuple[torch.Tensor, float]:
    """
    GPU version of single Gaussian projection
    
    Args:
        mean: (3,) 3D Gaussian mean
        cov: (3, 3) 3D covariance matrix
        K: (3, 3) camera internal parameters
        R: (3, 3) rotation matrix
        t: (3, 1) or (3,) translation vector
        image_size: (width, height)
        device: 'cuda' or 'cpu'
    
    Returns:
        density: (height, width) density map
        z_depth: float, the depth of the Gaussian center in the camera coordinate system
    """
    # Make sure the input is on the GPU
    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean).float().to(device)
    if isinstance(cov, np.ndarray):
        cov = torch.from_numpy(cov).float().to(device)
    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).float().to(device)
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R).float().to(device)
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t).float().to(device)
    
    means_3d = mean.unsqueeze(0)
    covs_3d = cov.unsqueeze(0)
    
    # Calculate mean z in the camera coordinate system
    if t.dim() == 2:
        t_vec = t.squeeze()
    else:
        t_vec = t
    mean_3d_cam = R @ mean + t_vec
    
    density = compute_probability_density_map_gpu(
        means_3d, covs_3d, K, R, t, image_size, device
    )
    density = torch.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    
    return density, mean_3d_cam[2].item()


def colorize_density_maps_alpha(
    density_maps: List[torch.Tensor],
    ids: List[int],
    obj_id_to_color_idx: Dict[int, int],
    H: int,
    W: int,
    threshold: float = 0.1,
    background: str = 'white',
    gray_level: int = 200,
    soft: bool = True,
    alpha_gamma: float = 1.0,
    order: str = 'input',
    strength_mode: str = 'sum',
    solid: bool = False,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Shade a set of density maps, using alpha blending
    
    Args:
        density_maps: List of (H, W) density map
        ids: List of object IDs
        obj_id_to_color_idx: mapping of object ID to color index (will be updated in place)
        H, W: image size
        threshold: density threshold
        background: background color 'white'|'gray'|'black' or (R,G,B) tuple
        gray_level: brightness of gray background (0-255)
        soft: True=soft threshold (smooth transparency), False=hard threshold
        alpha_gamma: non-linear adjustment of transparency (>1 is thinner, <1 is denser)
        order: drawing order 'input'|'strength'|'id'
        strength_mode: 'sum'|'max', used for order='strength'
        solid: True=solid color overlay, False=alpha blending
        device: 'cuda' or 'cpu'
    
    Returns:
        torch.uint8 (H, W, 3) RGB image
    """
    assert len(density_maps) == len(ids), "density_maps and ids must have the same length"
    
    if len(density_maps) == 0:
        # Return to blank background
        if isinstance(background, str):
            bg = background.lower()
            if bg == 'white':
                return torch.ones((H, W, 3), dtype=torch.uint8, device=device) * 255
            elif bg == 'gray':
                return torch.ones((H, W, 3), dtype=torch.uint8, device=device) * gray_level
            else:
                return torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
        else:
            r, g, b = background
            return torch.ones((H, W, 3), dtype=torch.uint8, device=device) * torch.tensor([r, g, b], device=device)
    
    dtype = torch.float32

    # Background color (0~1)
    if isinstance(background, str):
        bg = background.lower()
        if bg == 'white':
            bg_rgb = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)
        elif bg == 'gray':
            g = float(max(0, min(255, gray_level))) / 255.0
            bg_rgb = torch.tensor([g, g, g], dtype=dtype, device=device)
        elif bg == 'black':
            bg_rgb = torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device=device)
        else:
            bg_rgb = torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device=device)
    else:
        r, g, b = background
        bg_rgb = torch.tensor([r/255.0, g/255.0, b/255.0], dtype=dtype, device=device)

    # Initialize basemap as background
    composite = torch.ones((H, W, 3), dtype=dtype, device=device) * bg_rgb

    # Select stacking order
    idxs = list(range(len(density_maps)))
    if order == 'strength':
        def strength(t: torch.Tensor):
            m = t.max()
            if m <= 0:
                return 0.0
            t = t / (m + 1e-8)
            return (t.sum().item() if strength_mode == 'sum' else t.max().item())
        idxs.sort(key=lambda i: strength(density_maps[i]))
    elif order == 'id':
        idxs.sort(key=lambda i: ids[i])

    for i in idxs:
        prob = density_maps[i]
        obj_id = ids[i]

        # Size/device alignment
        assert prob.shape[-2:] == (H, W), f"prob shape {prob.shape} must match (H, W)={H, W}"
        prob = prob.to(device=device, dtype=dtype)

        # Use a unified color assignment function (consistent with trajectories)
        color_rgb = get_object_color(obj_id, obj_id_to_color_idx, device, return_float=True)

        # Normalize probability to [0,1]
        pmax = prob.max()
        if pmax > 0:
            prob = prob / (pmax + 1e-8)
        else:
            continue

        if solid:
            # solid color coverage
            mask = (prob > threshold)
            if mask.any():
                composite[mask] = color_rgb
        else:
            # Transparency (alpha)
            if soft:
                alpha = (prob - threshold) / (1.0 - threshold + 1e-8)
                alpha = alpha.clamp_(min=0.0, max=1.0)
            else:
                alpha = (prob > threshold).to(dtype) * prob

            # nonlinear stretching
            if alpha_gamma != 1.0:
                alpha = alpha.clamp(min=0.0, max=1.0).pow_(alpha_gamma)

            # Alpha composition: C_out = C_obj * a + C_in * (1 - a)
            a3 = alpha.unsqueeze(-1)  # (H, W, 1)
            composite = color_rgb * a3 + composite * (1.0 - a3)

    # Convert to uint8
    out = (composite.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return out


# ============================================================================
# Step 4 & 5: Ellipsoid and prism point cloud generation
# ============================================================================

def get_obb_corners_3d(
    center: torch.Tensor,
    extents: torch.Tensor,
    rotation: torch.Tensor
) -> torch.Tensor:
    """Get the 8 corner points of OBB"""
    local_corners = torch.tensor([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ], dtype=center.dtype, device=center.device) * extents
    
    world_corners = (rotation @ local_corners.T).T + center
    return world_corners


def sample_ellipsoid_points_gpu(
    mean: torch.Tensor,
    cov: torch.Tensor,
    num_samples: int = 2000,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Step 4: Generate ellipsoid point cloud from 3D Gaussian sampling points
    
    Args:
        mean: (3,) mean
        cov: (3, 3) covariance matrix
        num_samples: number of sampling points
    
    Returns:
        points: (num_samples, 3) Ellipsoid point cloud
    """
    if isinstance(mean, np.ndarray):
        mean_np = mean
        cov_np = cov
    else:
        mean_np = mean.cpu().numpy()
        cov_np = cov.cpu().numpy()
    
    # Sampling using numpy
    points_np = np.random.multivariate_normal(mean_np, cov_np, num_samples)
    points = torch.from_numpy(points_np).float().to(device)
    
    return points

def create_bbox_edge_points(
    center: torch.Tensor,
    extents: torch.Tensor,
    rotation: torch.Tensor,
    points_per_edge: int = 30
) -> torch.Tensor:
    """
    Step 5: Create the border point cloud of the OBB quadrilateral prism
    
    Returns:
        edge_points: (N, 3) edge point cloud
    """
    # Get 8 corner points
    corners = get_obb_corners_3d(center, extents, rotation)
    
    # Define 12 edges (vertex index pairs)
    edges = [
        # 4 sides on the bottom
        (0, 1), (1, 2), (2, 3), (3, 0),
        # 4 sides on top
        (4, 5), (5, 6), (6, 7), (7, 4),
        # 4 vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    
    edge_points_list = []
    for i, j in edges:
        # Linearly interpolate between two vertices
        t = torch.linspace(0, 1, points_per_edge, device=center.device)
        points = corners[i].unsqueeze(0) + t.unsqueeze(1) * (corners[j] - corners[i]).unsqueeze(0)
        edge_points_list.append(points)
    
    edge_points = torch.cat(edge_points_list, dim=0)
    return edge_points


def get_object_color(obj_id: int, obj_id_to_color_idx: Dict[int, int], device: str = 'cuda', 
                     return_float: bool = False) -> torch.Tensor:
    """
    Assign a color to the object
    
    Args:
        obj_id: object ID
        obj_id_to_color_idx: mapping of object ID to color index
        device: device
        return_float: Returns float32 (0-1) if True, otherwise returns uint8 (0-255)
    
    Returns:
        color: (3,) RGB color
    """
    cmap = plt.get_cmap('tab20')  # Use plt.get_cmap uniformly for consistency
    color_idx = obj_id_to_color_idx.get(obj_id, 0)
    color_rgb = cmap(color_idx % 20)[:3]  # 0~1
    
    if return_float:
        return torch.tensor(color_rgb, dtype=torch.float32, device=device)
    else:
        color_255 = torch.tensor([c * 255 for c in color_rgb], dtype=torch.uint8, device=device)
        return color_255


# ============================================================================
# Step 6: 2D trajectory projection and rendering
# ============================================================================

def project_3d_point_to_2d(
    point_3d: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor
) -> Tuple[torch.Tensor, float]:
    """
    Project 3D points to 2D image coordinates
    
    Args:
        point_3d: (3,) 3D point in world coordinate system
        intrinsic: (3, 3) camera intrinsic parameters
        extrinsic: (4, 4) camera external parameters (world to camera)
        
    Returns:
        point_2d: (2,) image coordinates [x, y]
        depth: float depth in camera coordinate system
    """
    # Convert to homogeneous coordinates
    point_3d_hom = torch.cat([point_3d, torch.ones(1, device=point_3d.device)])
    
    # World coordinate system -> camera coordinate system
    point_cam_hom = extrinsic @ point_3d_hom
    point_cam = point_cam_hom[:3]
    depth = point_cam[2].item()
    
    # Camera coordinate system -> Image coordinate system
    point_img_hom = intrinsic @ point_cam
    point_2d = point_img_hom[:2] / point_img_hom[2]
    
    return point_2d, depth


def draw_trajectory_on_image(
    image: torch.Tensor,
    depth_map: torch.Tensor,
    center_points_per_frame: Dict[int, Dict[int, torch.Tensor]],
    obj_id_to_color_idx: Dict[int, int],
    current_frame: int,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    line_thickness: int = 3,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Draw the trajectory points of the current frame on the 2D image (no historical trajectory lines are drawn)
    
    Args:
        image: (H, W, 3) background image (RGB, 0-255)
        depth_map: (H, W) background depth map
        center_points_per_frame: {frame_idx: {obj_id: center_point (3,)}}
        obj_id_to_color_idx: mapping of object ID to color index
        current_frame: current frame index
        intrinsic: (3, 3) camera intrinsic parameters
        extrinsic: (4, 4) camera external parameters
        line_thickness: the radius of the track point (the diameter of the circle will be 2×line_thickness)
        
    Returns:
        trajectory_image: (H, W, 3) image with trajectory points
        trajectory_depth: (H, W) trajectory point depth map
        trajectory_mask: (H, W) trajectory point mask (bool)
    """
    H, W = image.shape[:2]
    trajectory_image = image.clone()
    trajectory_depth = torch.zeros_like(depth_map)
    trajectory_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    
    # Draw the track points of the current frame for each object
    if current_frame not in center_points_per_frame:
        return trajectory_image, trajectory_depth, trajectory_mask
    
    for obj_id in obj_id_to_color_idx.keys():
        # Only process objects that exist in the current frame
        if obj_id not in center_points_per_frame[current_frame]:
            continue
            
        obj_color = get_object_color(obj_id, obj_id_to_color_idx, device)
        center_3d = center_points_per_frame[current_frame][obj_id]
        
        # Project to current camera perspective
        point_2d, depth = project_3d_point_to_2d(center_3d, intrinsic, extrinsic)
        
        # Check if the point is within the image range and the depth is valid
        if (0 <= point_2d[0] < W and 0 <= point_2d[1] < H and depth > 0):
            # Draw colored dots directly on the image
            trajectory_image, trajectory_depth = draw_point_marker(
                trajectory_image, trajectory_depth, 
                point_2d, depth, obj_color, line_thickness, device
            )
            trajectory_mask = trajectory_depth > 0
    
    return trajectory_image, trajectory_depth, trajectory_mask


def draw_accumulated_trajectory(
    image: torch.Tensor,
    depth_map: torch.Tensor,
    center_points_per_frame: Dict[int, Dict[int, torch.Tensor]],
    obj_id_to_color_idx: Dict[int, int],
    current_frame: int,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    line_thickness: int = 3,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Draw a cumulative trajectory line from frame 0 to the current frame on a 2D image
    
    Args:
        image: (H, W, 3) background image (RGB, 0-255)
        depth_map: (H, W) background depth map
        center_points_per_frame: {frame_idx: {obj_id: center_point (3,)}}
        obj_id_to_color_idx: mapping of object ID to color index
        current_frame: current frame index
        intrinsic: (3, 3) camera intrinsic parameters
        extrinsic: (4, 4) camera external parameters
        line_thickness: track line thickness (used for both point radius and line segment width to ensure consistent visualization size)
        
    Returns:
        trajectory_image: (H, W, 3) image with accumulated trajectory lines
        trajectory_depth: (H, W) trajectory line depth map
        trajectory_mask: (H, W) trajectory mask (bool)
    """
    H, W = image.shape[:2]
    trajectory_image = image.clone()
    trajectory_depth = torch.zeros_like(depth_map)
    trajectory_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    
    # Draw cumulative trajectories for each object
    for obj_id in obj_id_to_color_idx.keys():
        obj_color = get_object_color(obj_id, obj_id_to_color_idx, device)
        
        # Collect all center points and corresponding projection points of the object (from frame 0 to the current frame)
        trajectory_points_2d = []
        trajectory_depths = []
        
        for frame_idx in range(current_frame + 1):
            if (frame_idx in center_points_per_frame and 
                obj_id in center_points_per_frame[frame_idx]):
                center_3d = center_points_per_frame[frame_idx][obj_id]
                
                # Project to current camera perspective
                point_2d, depth = project_3d_point_to_2d(center_3d, intrinsic, extrinsic)
                
                # Check if the point is within the image range and the depth is valid
                if (0 <= point_2d[0] < W and 0 <= point_2d[1] < H and depth > 0):
                    trajectory_points_2d.append(point_2d)
                    trajectory_depths.append(depth)
        
        # Draw trajectory segments (at least 2 points required)
        if len(trajectory_points_2d) >= 2:
            for i in range(len(trajectory_points_2d) - 1):
                start_2d = trajectory_points_2d[i]
                end_2d = trajectory_points_2d[i + 1]
                start_depth = trajectory_depths[i]
                end_depth = trajectory_depths[i + 1]
                
                # Draw colored line segments directly on the image
                # Note: The thickness of cv2.line is the line width, and the radius of cv2.circle is the radius.
                # To keep track point size consistent, use the same line_thickness value
                trajectory_image, trajectory_depth = draw_line_with_depth(
                    trajectory_image, trajectory_depth,
                    start_2d, end_2d, start_depth, end_depth,
                    obj_color, max(1, line_thickness), device
                )
            trajectory_mask = trajectory_depth > 0
        
        # If there is only 1 point (first frame), draw the starting point marker
        elif len(trajectory_points_2d) == 1:
            point_2d = trajectory_points_2d[0]
            point_depth = trajectory_depths[0]
            
            # Draw colored dots directly on the image
            trajectory_image, trajectory_depth = draw_point_marker(
                trajectory_image, trajectory_depth,
                point_2d, point_depth, obj_color, line_thickness, device
            )
            trajectory_mask = trajectory_depth > 0
    
    return trajectory_image, trajectory_depth, trajectory_mask


def draw_point_marker(
    image: torch.Tensor,
    depth_map: torch.Tensor,
    point_2d: torch.Tensor,
    depth: float,
    color: torch.Tensor,
    radius: int = 3,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Draw colored dot markers directly on the image
    
    Args:
        image: (H, W, 3) input image (RGB, 0-255)
        depth_map: (H, W) depth map
        point_2d: (2,) 2D coordinates of point [x, y]
        depth: depth of point
        color: (3,) RGB color (0-255)
        radius: point radius (note: point diameter = 2 × radius)
        
    Returns:
        image: (H, W, 3) updated image
        depth_map: (H, W) updated depth map
    """
    # Convert to numpy for plotting
    image_np = image.cpu().numpy()
    depth_np = depth_map.cpu().numpy()
    # Note: image_np is in RGB format, color is also in RGB format, you can use it directly
    # OpenCV will draw in channel order without BGR conversion.
    color_tuple = (int(color[0].item()), int(color[1].item()), int(color[2].item()))  # RGB
    center = (int(point_2d[0].item()), int(point_2d[1].item()))
    
    # Draw colored dots directly on the image
    cv2.circle(image_np, center, radius, color_tuple, -1)
    
    # Create a mask to update the depth map
    H, W = depth_map.shape
    mask_np = np.zeros((H, W), dtype=np.float32)
    cv2.circle(mask_np, center, radius, 1.0, -1)
    mask = mask_np > 0.01
    depth_np[mask] = float(depth)
    
    # Convert back to torch tensor
    image = torch.from_numpy(image_np).to(device)
    depth_map = torch.from_numpy(depth_np).to(device)

    return image, depth_map


def draw_line_with_depth(
    image: torch.Tensor,
    depth_map: torch.Tensor,
    start_2d: torch.Tensor,
    end_2d: torch.Tensor,
    start_depth: float,
    end_depth: float,
    color: torch.Tensor,
    thickness: int = 3,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Draw colored line segments with depth directly on the image
    
    Args:
        image: (H, W, 3) input image (RGB, 0-255)
        depth_map: (H, W) depth map
        start_2d: (2,) 2D coordinates of the starting point [x, y]
        end_2d: (2,) 2D coordinates of the end point [x, y]
        start_depth: starting point depth
        end_depth: end point depth
        color: (3,) RGB color (0-255)
        thickness: line segment thickness (line segment width, corresponding to the diameter of the dot)
        
    Returns:
        image: (H, W, 3) updated image
        depth_map: (H, W) updated depth map
    """
    # Convert to numpy for plotting
    image_np = image.cpu().numpy()
    depth_np = depth_map.cpu().numpy()
    # Note: image_np is in RGB format, color is also in RGB format, you can use it directly
    # OpenCV will draw in channel order without BGR conversion.
    color_tuple = (int(color[0].item()), int(color[1].item()), int(color[2].item()))  # RGB
    start_pt = (int(start_2d[0].item()), int(start_2d[1].item()))
    end_pt = (int(end_2d[0].item()), int(end_2d[1].item()))
    
    # Draw colored lines directly on the image
    cv2.line(image_np, start_pt, end_pt, color_tuple, thickness, lineType=cv2.LINE_AA)
    
    # Create a mask to update the depth map (needs to calculate depth interpolation)
    H, W = depth_map.shape
    mask_np = np.zeros((H, W), dtype=np.float32)
    cv2.line(mask_np, start_pt, end_pt, 1.0, thickness, lineType=cv2.LINE_AA)
    mask = mask_np > 0.01
    
    # Calculate the depth of each point on the line segment (linear interpolation)
    if np.any(mask):
        y_coords, x_coords = np.where(mask)
        
        # Calculate the distance ratio of each point to the starting point and the ending point
        start_x, start_y = start_2d[0].item(), start_2d[1].item()
        end_x, end_y = end_2d[0].item(), end_2d[1].item()
        
        # total length of line segment
        total_length = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
        
        if total_length > 0:
            # Calculate the distance from each point to the starting point
            distances = np.sqrt((x_coords - start_x) ** 2 + (y_coords - start_y) ** 2)
            
            # Calculate interpolation scale (0 to 1)
            ratios = np.clip(distances / total_length, 0, 1)
            
            # Depth linear interpolation
            interpolated_depths = start_depth * (1 - ratios) + end_depth * ratios
            
            # Update depth
            depth_np[y_coords, x_coords] = interpolated_depths
        else:
            # If the start and end points coincide, use the start point depth
            depth_np[mask] = float(start_depth)
    
    # Convert back to torch tensor
    image = torch.from_numpy(image_np).to(device)
    depth_map = torch.from_numpy(depth_np).to(device)

    return image, depth_map


def render_trajectory_sequence(
    background_points: torch.Tensor,
    background_colors: torch.Tensor,
    center_points_per_frame: Dict[int, Dict[int, torch.Tensor]],
    obj_id_to_color_idx: Dict[int, int],
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_size: Tuple[int, int],
    mode: str = 'full',
    point_size: float = 0.01,
    line_thickness: int = 3,
    saturation_factor: float = 1.3,
    contrast_factor: float = 1.1,
    device: str = 'cuda'
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor],
           List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Render track sequence
    
    Args:
        background_points: background point cloud
        background_colors: background colors
        center_points_per_frame: center point of each frame
        obj_id_to_color_idx: mapping of object ID to color index
        intrinsics: camera internal parameters
        extrinsics: camera external parameters
        image_size: image size
        mode: 'full' (background + track), 'foreground' (track only), 'background' (background only)
        point_size: point size
        line_thickness: line thickness
        saturation_factor: saturation enhancement factor (1.0=unchanged, >1=enhanced, 1.2-1.5 recommended)
        contrast_factor: contrast enhancement factor (1.0=unchanged, >1=enhanced, recommended 1.1-1.3)
        device: device
        
    Returns:
        point_rgb_frames: List of (H, W, 3) RGB of the current point in each frame
        point_depth_frames: List of (H, W) Depth of the current point in each frame
        point_mask_frames: List of (H, W) Mask of the current point in each frame
        full_traj_rgb_frames: List of (H, W, 3) RGB of accumulated trajectory lines for each frame
        full_traj_depth_frames: List of (H, W) The depth of the accumulated trajectory line in each frame
        full_traj_mask_frames: List of (H, W) Mask of the accumulated trajectory line in each frame
    """
    num_frames = len(intrinsics)
    
    # Output of current point
    point_rgb_frames = []
    point_depth_frames = []
    point_mask_frames = []
    
    # Output of accumulated trajectory lines
    full_traj_rgb_frames = []
    full_traj_depth_frames = []
    full_traj_mask_frames = []
    
    H, W = image_size
    
    for t in tqdm(range(num_frames), desc=f"Rendering trajectory {mode}"):
        K = intrinsics[t]
        T = extrinsics[t]
        
        # render background
        if mode in ('full', 'background') and background_points is not None:
            bg_rgb, bg_depth, bg_mask = render_point_cloud_pytorch3d(
                background_points, background_colors, K, T, image_size, point_size
            )
        else:
            bg_rgb = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
            bg_depth = torch.zeros((H, W), dtype=torch.float32, device=device)
            bg_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
        
        # 1. Draw the points of the current frame (excluding historical trajectories)
        if mode in ('full', 'foreground'):
            point_traj_rgb, point_traj_depth, point_traj_mask = draw_trajectory_on_image(
                bg_rgb, bg_depth, center_points_per_frame, obj_id_to_color_idx,
                t, K, T, line_thickness, device
            )
        else:
            point_traj_rgb = bg_rgb
            point_traj_depth = bg_depth
            point_traj_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
        
        # 2. Draw the cumulative trajectory line from frame 0 to the current frame
        if mode in ('full', 'foreground'):
            accum_traj_rgb, accum_traj_depth, accum_traj_mask = draw_accumulated_trajectory(
                bg_rgb, bg_depth, center_points_per_frame, obj_id_to_color_idx,
                t, K, T, line_thickness, device
            )
        else:
            accum_traj_rgb = bg_rgb
            accum_traj_depth = bg_depth
            accum_traj_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
        
        # Process the output of the current point according to mode
        if mode == 'foreground':
            # Track points only: black background + points
            point_rgb = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
            point_rgb[point_traj_mask] = point_traj_rgb[point_traj_mask]
            point_depth = point_traj_depth
            point_mask = point_traj_mask
        elif mode == 'background':
            # background only
            point_rgb = bg_rgb
            point_depth = bg_depth  
            point_mask = bg_mask
        else:
            # Complete scene (background + points)
            point_rgb = point_traj_rgb
            point_depth = bg_depth.clone()
            valid_point_depth = point_traj_mask & (point_traj_depth > 0)
            point_depth[valid_point_depth] = point_traj_depth[valid_point_depth]
            point_mask = bg_mask | point_traj_mask
        
        # Process the output of accumulated trajectories according to mode
        if mode == 'foreground':
            # Track lines only: black background + track
            full_traj_rgb = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
            full_traj_rgb[accum_traj_mask] = accum_traj_rgb[accum_traj_mask]
            full_traj_depth = accum_traj_depth
            full_traj_mask = accum_traj_mask
        elif mode == 'background':
            # background only
            full_traj_rgb = bg_rgb
            full_traj_depth = bg_depth  
            full_traj_mask = bg_mask
        else:
            # Complete scene (background + trajectory lines)
            full_traj_rgb = accum_traj_rgb
            full_traj_depth = bg_depth.clone()
            valid_traj_depth = accum_traj_mask & (accum_traj_depth > 0)
            full_traj_depth[valid_traj_depth] = accum_traj_depth[valid_traj_depth]
            full_traj_mask = bg_mask | accum_traj_mask
        
        # Add to output list
        point_rgb_frames.append(point_rgb)
        point_depth_frames.append(point_depth)
        point_mask_frames.append(point_mask)
        
        full_traj_rgb_frames.append(full_traj_rgb)
        full_traj_depth_frames.append(full_traj_depth)
        full_traj_mask_frames.append(full_traj_mask)
    

    # Apply Gaussian blur to RGB videos with saturation enhancement
    def _gaussian_blur_and_enhance_video_rgb(
        frames: List[torch.Tensor], 
        sat_factor: float = 1.3,
        cont_factor: float = 1.1
    ) -> List[torch.Tensor]:
        """
        Apply slight Gaussian blur to RGB video and enhance saturation
        
        Args:
            frames: List of (H, W, 3) uint8 RGB frames
            sat_factor: saturation enhancement factor (1.0=unchanged, >1 enhanced, <1 reduced)
            cont_factor: Contrast enhancement factor (1.0=unchanged, >1 enhanced)
        """
        if not frames:
            return frames
        H, W = frames[0].shape[:2]
        
        # Use smaller kernel_size to reduce blur
        kernel_size = max(3, min(H, W) // 32)  # Changed to //32 (previously //16)
        if kernel_size % 2 == 0:
            kernel_size += 1
        # Decrease the sigma value to reduce blur
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.5  # Changed to +0.5 (was +0.8)

        # Stack to (B, C, H, W), normalize to [0,1]
        video = torch.stack([f.permute(2, 0, 1).float() / 255.0 for f in frames], dim=0)
        video_blurred = TF.gaussian_blur(video, kernel_size=kernel_size, sigma=sigma)
        
        # Enhance saturation and contrast
        if sat_factor != 1.0 or cont_factor != 1.0:
            # 1. Calculate the average brightness (grayscale) of each pixel
            luminance = video_blurred.mean(dim=1, keepdim=True)  # (B, 1, H, W)
            
            # 2. Enhance the degree of color deviation from grayscale (increase saturation)
            video_enhanced = luminance + (video_blurred - luminance) * sat_factor
            
            # 3. Increase contrast
            mean_val = video_enhanced.mean(dim=(2, 3), keepdim=True)
            video_enhanced = mean_val + (video_enhanced - mean_val) * cont_factor
            
            video_blurred = video_enhanced
        
        # Back to (H, W, 3) uint8 per frame
        return [
            (video_blurred[i].permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
            for i in range(video_blurred.shape[0])
        ]

    # Apply blur and enhancement to trajectory RGB frames
    point_rgb_frames = _gaussian_blur_and_enhance_video_rgb(
        point_rgb_frames, sat_factor=saturation_factor, cont_factor=contrast_factor
    )
    full_traj_rgb_frames = _gaussian_blur_and_enhance_video_rgb(
        full_traj_rgb_frames, sat_factor=saturation_factor, cont_factor=contrast_factor
    )


    return (point_rgb_frames, point_depth_frames, point_mask_frames,
            full_traj_rgb_frames, full_traj_depth_frames, full_traj_mask_frames)


def generate_trajectory_id_video(
    center_points_per_frame: Dict[int, Dict[int, torch.Tensor]],
    obj_id_to_color_idx: Dict[int, int],
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_size: Tuple[int, int],
    line_thickness: int = 3,
    device: str = 'cuda'
) -> List[torch.Tensor]:
    """
    Generate a trajectory ID video: the trajectory of each object is marked with a different color, and the background is black
    
    Returns:
        id_frames: List of (H, W, 3) RGB frames
    """
    num_frames = len(intrinsics)
    id_frames = []
    H, W = image_size
    
    for t in range(num_frames):
        # Initialize to black background
        id_map = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
        
        K = intrinsics[t]
        T = extrinsics[t]
        
        # Draw track ID for each object
        for obj_id in obj_id_to_color_idx.keys():
            # Use a unified color assignment function (consistent with other videos)
            color_255 = get_object_color(obj_id, obj_id_to_color_idx, device)
            
            # Collect all center points and corresponding projection points of the object (from frame 0 to the current frame)
            trajectory_points_2d = []
            
            for frame_idx in range(t + 1):
                if (frame_idx in center_points_per_frame and 
                    obj_id in center_points_per_frame[frame_idx]):
                    center_3d = center_points_per_frame[frame_idx][obj_id]
                    
                    # Project to current camera perspective
                    point_2d, depth = project_3d_point_to_2d(center_3d, K, T)
                    
                    # Check if the point is within the image range and the depth is valid
                    if (0 <= point_2d[0] < W and 0 <= point_2d[1] < H and depth > 0):
                        trajectory_points_2d.append(point_2d)
            
            # Draw trajectory segments
            if len(trajectory_points_2d) >= 2:
                for i in range(len(trajectory_points_2d) - 1):
                    start_2d = trajectory_points_2d[i]
                    end_2d = trajectory_points_2d[i + 1]
                    
                    # Draw line segments using OpenCV
                    start_np = start_2d.cpu().numpy().astype(int)
                    end_np = end_2d.cpu().numpy().astype(int)
                    
                    # Convert to numpy plot
                    id_map_np = id_map.cpu().numpy()
                    color_np = color_255.cpu().numpy()
                    
                    cv2.line(id_map_np, tuple(start_np), tuple(end_np), 
                            color_np.tolist(), line_thickness)
                    
                    # Convert back to torch
                    id_map = torch.from_numpy(id_map_np).to(device)
        
        id_frames.append(id_map)
    
    return id_frames


# ============================================================================
# Step 7-15: Rendering system (point cloud + mesh)
# ============================================================================

def _build_cameras_from_extrinsics(K: torch.Tensor, T_world_camera: torch.Tensor, image_size: Tuple[int, int]):
    """Convert external parameters from COLMAP style to PyTorch3D cameras and return PerspectiveCameras."""
    H, W = image_size
    device = K.device
    
    # To save the original dtype, linalg.inv requires FP32
    original_dtype = T_world_camera.dtype
    use_fp32_for_inv = original_dtype == torch.float16
    
    if use_fp32_for_inv:
        K = K.float()
        T_world_camera = T_world_camera.float()

    # Convert camera coordinate system: COLMAP RDF -> PyTorch3D LUF
    c2w = torch.linalg.inv(T_world_camera)
    c2w[:3, :2] *= -1
    w2c = torch.linalg.inv(c2w)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    focal_length = torch.tensor([[fx.item(), fy.item()]], device=device)
    principal_point = torch.tensor([[cx.item(), cy.item()]], device=device)

    R_camera = w2c[:3, :3].permute(1, 0)
    T_camera = w2c[:3, 3]
    
    # Convert back to original dtype (if needed)
    if use_fp32_for_inv:
        focal_length = focal_length.to(original_dtype)
        principal_point = principal_point.to(original_dtype)
        R_camera = R_camera.to(original_dtype)
        T_camera = T_camera.to(original_dtype)

    cameras = PerspectiveCameras(
        device=device,
        focal_length=focal_length,
        principal_point=principal_point,
        R=R_camera.unsqueeze(0),
        T=T_camera.unsqueeze(0),
        in_ndc=False,
        image_size=((H, W),)
    )
    return cameras


def _build_cameras_from_extrinsics_batch(Ks: torch.Tensor, Ts: torch.Tensor, image_size: Tuple[int, int]):
    """Build cameras in batches.
    Args:
        Ks: (B, 3, 3) internal parameter matrix
        Ts: (B, 4, 4) external parameter matrix
        image_size: (H, W)
    Returns:
        PerspectiveCameras with batch size B
    """
    H, W = image_size
    device = Ks.device
    B = Ks.shape[0]
    
    # To save the original dtype, linalg.inv requires FP32
    original_dtype = Ts.dtype
    use_fp32_for_inv = original_dtype == torch.float16
    
    if use_fp32_for_inv:
        Ks = Ks.float()
        Ts = Ts.float()

    # Convert camera coordinate systems in batches
    c2ws = torch.linalg.inv(Ts)
    c2ws[:, :3, :2] *= -1
    w2cs = torch.linalg.inv(c2ws)

    fx = Ks[:, 0, 0]
    fy = Ks[:, 1, 1]
    cx = Ks[:, 0, 2]
    cy = Ks[:, 1, 2]

    focal_length = torch.stack([fx, fy], dim=1)  # (B, 2)
    principal_point = torch.stack([cx, cy], dim=1)  # (B, 2)

    R_cameras = w2cs[:, :3, :3].permute(0, 2, 1)  # (B, 3, 3)
    T_cameras = w2cs[:, :3, 3]  # (B, 3)
    
    # Convert back to original dtype (if needed)
    if use_fp32_for_inv:
        focal_length = focal_length.to(original_dtype)
        principal_point = principal_point.to(original_dtype)
        R_cameras = R_cameras.to(original_dtype)
        T_cameras = T_cameras.to(original_dtype)

    cameras = PerspectiveCameras(
        device=device,
        focal_length=focal_length,
        principal_point=principal_point,
        R=R_cameras,
        T=T_cameras,
        in_ndc=False,
        image_size=[(H, W)] * B
    )
    return cameras

def render_point_cloud_pytorch3d(
    points_3d: torch.Tensor,
    colors: torch.Tensor,
    K: torch.Tensor,
    T_world_camera: torch.Tensor,
    image_size: Tuple[int, int],
    point_size: float = 0.01,
    background_color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Render point cloud using PyTorch3D, returning RGB and depth map
    
    Returns:
        rendered_rgb: (H, W, 3) rendered RGB image (0-255 uint8)
        depth_map: (H, W) depth map (float32)
        mask: (H, W) point cloud mask (bool)
    """
    H, W = image_size
    device = points_3d.device
    
    if len(points_3d) == 0:
        return (
            torch.full((H, W, 3), int(background_color[0] * 255), dtype=torch.uint8, device=device),
            torch.zeros((H, W), dtype=torch.float32, device=device),
            torch.zeros((H, W), dtype=torch.bool, device=device)
        )
    
    # camera
    cameras = _build_cameras_from_extrinsics(K, T_world_camera, image_size)

    # Create point cloud objects
    point_cloud = Pointclouds(
        points=[points_3d],
        features=[colors.float() / 255.0]
    )
    
    # Rasterization settings
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=point_size,
        points_per_pixel=10,
        bin_size=0
    )
    
    # Create renderer
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    compositor = AlphaCompositor(background_color=background_color)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)
    
    # rendering
    rendered_image = renderer(point_cloud)[0]
    rendered_rgb = rendered_image[..., :3] * 255
    rendered_rgb = torch.clamp(rendered_rgb, 0, 255).to(torch.uint8)
    
    # get depth
    fragments = rasterizer(point_cloud)
    mask = (fragments.idx[0, ..., 0] != -1)
    depth_map = fragments.zbuf[0][..., 0]
    depth_map[~mask] = 0.0
    
    return rendered_rgb, depth_map, mask


def render_point_cloud_pytorch3d_batch(
    points_3d: torch.Tensor,
    colors: torch.Tensor,
    Ks: torch.Tensor,
    Ts: torch.Tensor,
    image_size: Tuple[int, int],
    point_size: float = 0.01,
    background_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    use_fp16: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch rendering point cloud (optimized version + FP16 support)
    
    Args:
        points_3d: (N, 3) point cloud coordinates (shared by all frames)
        colors: (N, 3) point cloud colors (shared by all frames)
        Ks: (B, 3, 3) internal parameter matrix
        Ts: (B, 4, 4) external parameter matrix
        image_size: (H, W)
        point_size: point size
        background_color: background color
        use_fp16: whether to use FP16 acceleration
    
    Returns:
        rendered_rgb: (B, H, W, 3) rendered RGB image (0-255 uint8)
        depth_maps: (B, H, W) depth map (float32)
        masks: (B, H, W) point cloud mask (bool)
    """
    H, W = image_size
    device = points_3d.device
    B = Ks.shape[0]
    
    if len(points_3d) == 0:
        return (
            torch.full((B, H, W, 3), int(background_color[0] * 255), dtype=torch.uint8, device=device),
            torch.zeros((B, H, W), dtype=torch.float32, device=device),
            torch.zeros((B, H, W), dtype=torch.bool, device=device)
        )
    
    # FP16 Optimization: Convert input data types
    compute_dtype = torch.float16 if use_fp16 else torch.float32
    
    # Building batch cameras - using computational precision
    cameras = _build_cameras_from_extrinsics_batch(
        Ks.to(compute_dtype), Ts.to(compute_dtype), image_size
    )

    # Optimization: Pre-normalize colors and ensure on GPU
    colors_normalized = colors.to(compute_dtype) / 255.0
    points_3d_compute = points_3d.to(compute_dtype)
    
    # Create point cloud object (repeat B times) - use list comprehension to avoid repeated creation
    point_cloud = Pointclouds(
        points=[points_3d_compute for _ in range(B)],
        features=[colors_normalized for _ in range(B)]
    )
    
    # Rasterization settings - optimize bin_size and points_per_pixel
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=point_size,
        points_per_pixel=8,  # Decreased from 10 to 8 for faster speed
        bin_size=128,  # Let PyTorch3D automatically choose the optimal value
    )
    
    # Create renderer
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    compositor = AlphaCompositor(background_color=background_color)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)
    
    # Batch rendering - use torch.no_grad() to reduce memory usage
    with torch.no_grad():
        # FP16 automatic mixed precision
        if use_fp16 and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                rendered_images = renderer(point_cloud)  # (B, H, W, 4)
        else:
            rendered_images = renderer(point_cloud)  # (B, H, W, 4)
        
        # Transfer back to FP32 for post-processing
        rendered_rgb = rendered_images[..., :3].float() * 255
        rendered_rgb = torch.clamp(rendered_rgb, 0, 255).to(torch.uint8)
        
        # Get depth - reuse rasterizer to avoid double calculations
        fragments = rasterizer(point_cloud)
        masks = (fragments.idx[..., 0] != -1)  # (B, H, W)
        depth_maps = fragments.zbuf[..., 0].float().clone()  # (B, H, W) - Make sure it is FP32
        depth_maps[~masks] = 0.0
    
    return rendered_rgb, depth_maps, masks


def make_ellipsoid_mesh(mean: torch.Tensor, cov: torch.Tensor, scale_factor: float = 2.0, subdivisions: int = 3,
                        color_rgb255: Optional[torch.Tensor] = None, device: str = 'cuda') -> Meshes:
    """Construct an ellipsoid Mesh from 3D Gaussians (mean, cov)."""
    device = mean.device if isinstance(mean, torch.Tensor) else torch.device(device)
    sphere = ico_sphere(subdivisions, device=device)  # unit sphere at origin
    verts = sphere.verts_list()[0]  # (V,3)
    faces = sphere.faces_list()[0]  # (F,3)

    if isinstance(mean, np.ndarray):
        mean_t = torch.from_numpy(mean).float().to(device)
    else:
        mean_t = mean.to(device).float()
    if isinstance(cov, np.ndarray):
        cov_t = torch.from_numpy(cov).float().to(device)
    else:
        cov_t = cov.to(device).float()

    # eigendecomposition
    evals, evecs = torch.linalg.eigh(cov_t)
    evals = torch.clamp(evals, min=1e-8)
    axes = scale_factor * torch.sqrt(evals)
    # Transform: x = mean + R * diag(axes) * u
    M = evecs @ torch.diag(axes)  # (3,3)
    verts_world = verts @ M.T + mean_t  # (V,3)

    # colors
    if color_rgb255 is None:
        color_rgb255 = torch.tensor([200, 60, 60], dtype=torch.uint8, device=device)
    colors = (color_rgb255.float() / 255.0).expand_as(verts_world)
    textures = TexturesVertex(verts_features=colors.unsqueeze(0))

    return Meshes(verts=[verts_world], faces=[faces], textures=textures)


def make_rectangular_prism_mesh(center: torch.Tensor, extents: torch.Tensor, rotation: torch.Tensor,
                                color_rgb255: Optional[torch.Tensor] = None, device: str = 'cuda') -> Meshes:
    """Construct a cuboid Mesh based on the OBB center, half axes (extents) and rotation (rotation)."""
    device = center.device
    # local vertices (order matches get_obb_corners_3d)
    local = torch.tensor([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ], dtype=torch.float32, device=device)
    local = local * extents.view(1, 3)
    verts_world = (rotation @ local.T).T + center.view(1, 3)

    # triangle faces (12 triangles)
    faces = torch.tensor([
        # bottom (-z)
        [0, 1, 2], [0, 2, 3],
        # top (+z)
        [4, 5, 6], [4, 6, 7],
        # sides
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=torch.int64, device=device)

    if color_rgb255 is None:
        color_rgb255 = torch.tensor([60, 160, 220], dtype=torch.uint8, device=device)
    colors = (color_rgb255.float() / 255.0).expand_as(verts_world)
    textures = TexturesVertex(verts_features=colors.unsqueeze(0))

    return Meshes(verts=[verts_world], faces=[faces], textures=textures)


def combine_meshes_for_scene(mesh_list: List[Meshes]) -> Optional[Meshes]:
    """Combine multiple independent Meshes into a single scene Meshes (single batch). Returns None if empty."""
    if len(mesh_list) == 0:
        return None
    device = mesh_list[0].verts_list()[0].device
    verts_all = []
    faces_all = []
    colors_all = []
    v_ofs = 0
    for m in mesh_list:
        v = m.verts_list()[0]
        f = m.faces_list()[0]
        verts_all.append(v)
        faces_all.append(f + v_ofs)
        v_ofs += v.shape[0]
        # get vertex colors from TexturesVertex
        if isinstance(m.textures, TexturesVertex):
            col = m.textures.verts_features_list()[0]
        else:
            col = torch.ones_like(v) * 0.7
        colors_all.append(col)
    verts_cat = torch.cat(verts_all, dim=0)
    faces_cat = torch.cat(faces_all, dim=0)
    colors_cat = torch.cat(colors_all, dim=0)
    textures = TexturesVertex(verts_features=colors_cat.unsqueeze(0))
    return Meshes(verts=[verts_cat], faces=[faces_cat], textures=textures).to(device)


def render_meshes_pytorch3d(
    meshes: Optional[Meshes],
    K: torch.Tensor,
    T_world_camera: torch.Tensor,
    image_size: Tuple[int, int],
    background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rendering the mesh using PyTorch3D. Return RGB(uint8), depth(float32), mask(bool)."""
    H, W = image_size
    device = K.device
    if meshes is None or meshes.num_verts_per_mesh().sum().item() == 0:
        return (
            torch.full((H, W, 3), int(background_color[0] * 255), dtype=torch.uint8, device=device),
            torch.zeros((H, W), dtype=torch.float32, device=device),
            torch.zeros((H, W), dtype=torch.bool, device=device),
        )

    cameras = _build_cameras_from_extrinsics(K, T_world_camera, image_size)

    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
    )
    lights = PointLights(location=[[2.0, 2.0, 2.0]], device=device)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = HardPhongShader(device=device, cameras=cameras, lights=lights)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    rendered = renderer(meshes)[0]  # (H, W, 4)
    rgb = (torch.clamp(rendered[..., :3], 0, 1) * 255).to(torch.uint8)

    fragments = rasterizer(meshes)
    pix_to_face = fragments.pix_to_face[0, ..., 0]  # (H,W)
    mask = pix_to_face != -1
    depth = fragments.zbuf[0, ..., 0]
    depth[~mask] = 0.0
    
    # Sets the background area to the specified background color
    bg_color_uint8 = torch.tensor(
        [int(c * 255) for c in background_color],
        dtype=torch.uint8, device=device
    )
    rgb[~mask] = bg_color_uint8
    
    return rgb, depth, mask


def render_meshes_pytorch3d_batch(
    meshes_list: List[Optional[Meshes]],
    Ks: torch.Tensor,
    Ts: torch.Tensor,
    image_size: Tuple[int, int],
    background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    use_fp16: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch rendering of meshes (optimized version + FP16 support).
    
    Args:
        meshes_list: List of B meshes (each can be None)
        Ks: (B, 3, 3) internal parameter matrix
        Ts: (B, 4, 4) external parameter matrix
        image_size: (H, W)
        background_color: background color
        use_fp16: whether to use FP16 acceleration
    
    Returns:
        rgb: (B, H, W, 3) RGB image (uint8)
        depth: (B, H, W) depth map (float32)
        masks: (B, H, W) mask (bool)
    """
    H, W = image_size
    device = Ks.device
    B = len(meshes_list)
    
    # Initialize output
    rgb_batch = torch.full((B, H, W, 3), int(background_color[0] * 255), dtype=torch.uint8, device=device)
    depth_batch = torch.zeros((B, H, W), dtype=torch.float32, device=device)
    mask_batch = torch.zeros((B, H, W), dtype=torch.bool, device=device)
    
    # Find non-empty mesh
    valid_indices = [i for i, m in enumerate(meshes_list) if m is not None and m.num_verts_per_mesh().sum().item() > 0]
    
    if len(valid_indices) == 0:
        return rgb_batch, depth_batch, mask_batch
    
    # Merge valid meshes
    valid_meshes = [meshes_list[i] for i in valid_indices]
    valid_Ks = Ks[valid_indices]
    valid_Ts = Ts[valid_indices]
    
    # FP16 Optimization: Convert input data types
    compute_dtype = torch.float16 if use_fp16 else torch.float32
    
    # Build batch cameras
    cameras = _build_cameras_from_extrinsics_batch(
        valid_Ks.to(compute_dtype), valid_Ts.to(compute_dtype), image_size
    )
    
    # Merge all meshes into a batch
    merged_meshes = join_meshes_as_batch(valid_meshes)
    
    # Optimize rasterization settings
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,  # Automatically select the optimal bin size
        max_faces_per_bin=None  # automatic selection
    )
    
    # Simplify lighting setup
    lights = PointLights(location=[[0.0, 0.0, 0.0]], device=device)
    
    # Create renderer
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = HardPhongShader(device=device, cameras=cameras, lights=lights)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    # Batch rendering - using torch.no_grad()
    with torch.no_grad():
        # FP16 automatic mixed precision
        if use_fp16 and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                rendered = renderer(merged_meshes)  # (B_valid, H, W, 4)
        else:
            rendered = renderer(merged_meshes)  # (B_valid, H, W, 4)
        
        # Transfer back to FP32 for post-processing
        rgb_valid = (torch.clamp(rendered[..., :3].float(), 0, 1) * 255).to(torch.uint8)

        fragments = rasterizer(merged_meshes)
        pix_to_face = fragments.pix_to_face[..., 0]  # (B_valid, H, W)
        mask_valid = pix_to_face != -1
        depth_valid = fragments.zbuf[..., 0].float().clone()  # (B_valid, H, W) - Make sure it is FP32
        depth_valid[~mask_valid] = 0.0
    
    # Set background color
    bg_color_uint8 = torch.tensor(
        [int(c * 255) for c in background_color],
        dtype=torch.uint8, device=device
    )
    rgb_valid[~mask_valid] = bg_color_uint8
    
    # Fill to output batch
    for i, idx in enumerate(valid_indices):
        rgb_batch[idx] = rgb_valid[i]
        depth_batch[idx] = depth_valid[i]
        mask_batch[idx] = mask_valid[i]
    
    return rgb_batch, depth_batch, mask_batch


def composite_by_depth(
    bg_rgb: torch.Tensor,
    bg_depth: torch.Tensor,
    fg_rgb: torch.Tensor,
    fg_depth: torch.Tensor,
    fg_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Depth-based foreground and background synthesis. Returns the synthesized RGB (uint8) and depth."""
    assert bg_rgb.shape[:2] == bg_depth.shape and fg_rgb.shape[:2] == fg_depth.shape
    H, W, _ = bg_rgb.shape
    device = bg_rgb.device
    take_fg = fg_mask & ((bg_depth <= 0) | (fg_depth > 0) & (fg_depth < bg_depth - 1e-6))
    out_rgb = bg_rgb.clone()
    out_rgb[take_fg] = fg_rgb[take_fg]
    out_depth = bg_depth.clone()
    out_depth[take_fg] = fg_depth[take_fg]
    return out_rgb, out_depth


def composite_by_depth_batch(
    bg_rgb: torch.Tensor,
    bg_depth: torch.Tensor,
    fg_rgb: torch.Tensor,
    fg_depth: torch.Tensor,
    fg_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch version of deep compositing.
    
    Args:
        bg_rgb: (B, H, W, 3)
        bg_depth: (B, H, W)
        fg_rgb: (B, H, W, 3)
        fg_depth: (B, H, W)
        fg_mask: (B, H, W)
    
    Returns:
        out_rgb: (B, H, W, 3)
        out_depth: (B, H, W)
    """
    take_fg = fg_mask & ((bg_depth <= 0) | ((fg_depth > 0) & (fg_depth < bg_depth - 1e-6)))
    out_rgb = bg_rgb.clone()
    out_rgb[take_fg] = fg_rgb[take_fg]
    out_depth = bg_depth.clone()
    out_depth[take_fg] = fg_depth[take_fg]
    return out_rgb, out_depth


def composite_rendered_sequences(
    bg_rgb_frames: List[torch.Tensor],
    bg_depth_frames: List[torch.Tensor],
    bg_masks: List[torch.Tensor],
    fg_rgb_frames: List[torch.Tensor],
    fg_depth_frames: List[torch.Tensor],
    fg_masks: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Synthesize already rendered background and foreground frame sequences to avoid re-rendering
    
    Returns:
        rgb_frames, depth_frames, bg_masks, fg_masks
    """
    assert len(bg_rgb_frames) == len(fg_rgb_frames), "background and foreground frame counts must match"
    
    rgb_frames = []
    depth_frames = []
    
    for bg_rgb, bg_depth, bg_mask, fg_rgb, fg_depth, fg_mask in zip(
        bg_rgb_frames, bg_depth_frames, bg_masks, fg_rgb_frames, fg_depth_frames, fg_masks
    ):
        rgb_out, depth_out = composite_by_depth(bg_rgb, bg_depth, fg_rgb, fg_depth, fg_mask)
        rgb_frames.append(rgb_out)
        depth_frames.append(depth_out)
    
    return rgb_frames, depth_frames, bg_masks, fg_masks


def composite_rendered_sequences_soft(
    bg_rgb_frames: List[torch.Tensor],
    bg_depth_frames: List[torch.Tensor],
    bg_masks: List[torch.Tensor],
    fg_rgb_frames: List[torch.Tensor],
    fg_depth_frames: List[torch.Tensor],
    fg_masks: List[torch.Tensor],
    line_thickness: int = 3,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Synthesize a sequence of rendered background and foreground frames without using Gaussian blur, maintaining clear boundaries.

    Only used for foregrounds such as points/trajectories that require clear edges.
    """
    assert len(bg_rgb_frames) == len(fg_rgb_frames), "background and foreground frame counts must match"

    rgb_frames = []
    depth_frames = []

    for bg_rgb, bg_depth, bg_mask, fg_rgb, fg_depth, fg_mask in zip(
        bg_rgb_frames, bg_depth_frames, bg_masks, fg_rgb_frames, fg_depth_frames, fg_masks
    ):
        # Determine whether the foreground is covered by depth testing
        take_fg = fg_mask & ((bg_depth <= 0) | ((fg_depth > 0) & (fg_depth < bg_depth - 1e-6)))

        # Do not use Gaussian blur, use hard boundary mask directly
        alpha_effective = take_fg.float()

        # RGB direct replacement (hard border)
        out_rgb = bg_rgb.clone()
        out_rgb[take_fg] = fg_rgb[take_fg]

        # Depth still uses hard selection (foreground is closer)
        out_depth = bg_depth.clone()
        out_depth[take_fg] = fg_depth[take_fg]

        rgb_frames.append(out_rgb)
        depth_frames.append(out_depth)

    return rgb_frames, depth_frames, bg_masks, fg_masks


def render_frame_sequence(
    background_points: torch.Tensor,
    background_colors: torch.Tensor,
    foreground_points_per_frame: List[torch.Tensor],
    foreground_colors_per_frame: List[torch.Tensor],
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_size: Tuple[int, int],
    mode: str = 'full',
    point_size: float = 0.01,
    device: str = 'cuda'
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    render frame sequence
    
    Args:
        mode: 'full' (background + foreground), 'foreground' (foreground only), 'background' (background only)
    
    Returns:
        rgb_frames: List of (H, W, 3) RGB frames
        depth_frames: List of (H, W) depth frames
    """
    num_frames = len(foreground_points_per_frame)
    rgb_frames = []
    depth_frames = []
    
    # Make sure all input tensors are on the correct device
    background_points = background_points.to(device)
    background_colors = background_colors.to(device)
    
    for t in tqdm(range(num_frames), desc=f"Rendering {mode}"):
        if mode == 'full':
            # background + foreground
            fg_pts = foreground_points_per_frame[t].to(device)
            fg_colors = foreground_colors_per_frame[t].to(device)
            points = torch.cat([background_points, fg_pts], dim=0)
            colors = torch.cat([background_colors, fg_colors], dim=0)
        elif mode == 'foreground':
            # Foreground only
            points = foreground_points_per_frame[t].to(device)
            colors = foreground_colors_per_frame[t].to(device)
        elif mode == 'background':
            # background only
            points = background_points
            colors = background_colors
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        rgb, depth, _ = render_point_cloud_pytorch3d(
            points, colors,
            intrinsics[t], extrinsics[t],
            image_size, point_size
        )
        
        rgb_frames.append(rgb)
        depth_frames.append(depth)
    
    return rgb_frames, depth_frames


def render_frame_sequence_mesh_composited(
    background_points: torch.Tensor,
    background_colors: torch.Tensor,
    foreground_meshes_per_frame: List[Optional[Meshes]],
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_size: Tuple[int, int],
    mode: str = 'full',  # 'full' | 'foreground' | 'background'
    point_size: float = 0.01,
    device: str = 'cuda',
    batch_size: int = 1,  # Added batch_size parameter
    use_fp16: bool = False,  # Added FP16 parameters
    pin_memory: bool = False  # Added pin_memory parameter
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Rendering sequence: The background is rendered with a point cloud, the foreground is rendered with a mesh, and then composited based on depth.
    
    Args:
        batch_size: Batch processing size, default is 1 (frame-by-frame processing). Setting a larger value speeds up rendering.
        use_fp16: Use FP16 mixed precision acceleration
        pin_memory: Use pinned memory to accelerate GPU transfers
    """
    num_frames = len(foreground_meshes_per_frame)
    rgb_frames: List[torch.Tensor] = []
    depth_frames: List[torch.Tensor] = []
    background_masks: List[torch.Tensor] = []
    foreground_masks: List[torch.Tensor] = []

    if background_points is not None:
        # Pin memory optimization
        if pin_memory and not background_points.is_pinned():
            background_points = background_points.pin_memory()
            background_colors = background_colors.pin_memory()
        background_points = background_points #.to(device)
        background_colors = background_colors #.to(device)

    # If batch_size is 1 or equal to num_frames, use the original logic to avoid additional overhead
    if batch_size == 1:
        for t in tqdm(range(num_frames), desc=f"Rendering(mesh) {mode}"):
            K = intrinsics[t]
            T = extrinsics[t]

            rgb_bg = None
            depth_bg = None
            mask_bg = None
            if mode in ('full', 'background') and background_points is not None:
                rgb_bg, depth_bg, mask_bg = render_point_cloud_pytorch3d(
                    background_points, background_colors, K, T, image_size, point_size
                )
            
            if rgb_bg is None:
                # dummy background (black)
                H, W = image_size
                rgb_bg = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
                depth_bg = torch.zeros((H, W), dtype=torch.float32, device=device)
                mask_bg = torch.zeros((H, W), dtype=torch.bool, device=device)

            rgb_fg = None
            depth_fg = None
            mask_fg = None
            if mode in ('full', 'foreground'):
                rgb_fg, depth_fg, mask_fg = render_meshes_pytorch3d(
                    foreground_meshes_per_frame[t], K, T, image_size
                )
            else:
                H, W = image_size
                rgb_fg = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
                depth_fg = torch.zeros((H, W), dtype=torch.float32, device=device)
                mask_fg = torch.zeros((H, W), dtype=torch.bool, device=device)

            if mode == 'foreground':
                rgb_out = rgb_fg
                depth_out = depth_fg
            elif mode == 'background':
                rgb_out = rgb_bg
                depth_out = depth_bg
            else:
                rgb_out, depth_out = composite_by_depth(rgb_bg, depth_bg, rgb_fg, depth_fg, mask_fg)

            rgb_frames.append(rgb_out)
            depth_frames.append(depth_out)
            background_masks.append(mask_bg)
            foreground_masks.append(mask_fg)
    else:
        # Batch mode - optimized version
        num_batches = (num_frames + batch_size - 1) // batch_size
        
        # Preallocate output list to reduce append operations
        rgb_frames = [None] * num_frames
        depth_frames = [None] * num_frames
        background_masks = [None] * num_frames
        foreground_masks = [None] * num_frames
        
        for batch_idx in tqdm(range(num_batches), desc=f"Rendering(mesh-batch) {mode}"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_frames)
            current_batch_size = end_idx - start_idx
            
            # Prepare the internal and external parameters of the batch
            Ks_batch = intrinsics[start_idx:end_idx]  # (B, 3, 3)
            Ts_batch = extrinsics[start_idx:end_idx]  # (B, 4, 4)
            
            # render background
            rgb_bg_batch = None
            depth_bg_batch = None
            mask_bg_batch = None
            if mode in ('full', 'background') and background_points is not None:
                rgb_bg_batch, depth_bg_batch, mask_bg_batch = render_point_cloud_pytorch3d_batch(
                    background_points, background_colors, Ks_batch, Ts_batch, image_size, point_size, use_fp16=use_fp16
                )
            
            if rgb_bg_batch is None:
                H, W = image_size
                rgb_bg_batch = torch.zeros((current_batch_size, H, W, 3), dtype=torch.uint8, device=device)
                depth_bg_batch = torch.zeros((current_batch_size, H, W), dtype=torch.float32, device=device)
                mask_bg_batch = torch.zeros((current_batch_size, H, W), dtype=torch.bool, device=device)
            
            # Render foreground
            rgb_fg_batch = None
            depth_fg_batch = None
            mask_fg_batch = None
            if mode in ('full', 'foreground'):
                meshes_batch = foreground_meshes_per_frame[start_idx:end_idx]
                rgb_fg_batch, depth_fg_batch, mask_fg_batch = render_meshes_pytorch3d_batch(
                    meshes_batch, Ks_batch, Ts_batch, image_size, use_fp16=use_fp16
                )
            else:
                H, W = image_size
                rgb_fg_batch = torch.zeros((current_batch_size, H, W, 3), dtype=torch.uint8, device=device)
                depth_fg_batch = torch.zeros((current_batch_size, H, W), dtype=torch.float32, device=device)
                mask_fg_batch = torch.zeros((current_batch_size, H, W), dtype=torch.bool, device=device)
            
            # synthesis
            if mode == 'foreground':
                rgb_out_batch = rgb_fg_batch
                depth_out_batch = depth_fg_batch
            elif mode == 'background':
                rgb_out_batch = rgb_bg_batch
                depth_out_batch = depth_bg_batch
            else:
                rgb_out_batch, depth_out_batch = composite_by_depth_batch(
                    rgb_bg_batch, depth_bg_batch, rgb_fg_batch, depth_fg_batch, mask_fg_batch
                )
            
            # Directly assign values ​​to pre-allocated locations to avoid append
            for i in range(current_batch_size):
                idx = start_idx + i
                rgb_frames[idx] = rgb_out_batch[i]
                depth_frames[idx] = depth_out_batch[i]
                background_masks[idx] = mask_bg_batch[i]
                foreground_masks[idx] = mask_fg_batch[i]
            
            # Clean up intermediate variables to free up GPU memory
            del rgb_bg_batch, depth_bg_batch, mask_bg_batch
            del rgb_fg_batch, depth_fg_batch, mask_fg_batch
            del rgb_out_batch, depth_out_batch
            if device == 'cuda':
                torch.cuda.empty_cache()

    return rgb_frames, depth_frames, background_masks, foreground_masks


# ============================================================================
# Step 12: Auxiliary video generation
# ============================================================================

def generate_background_mask_video(
    masks_per_frame: Dict[int, Dict[int, np.ndarray]],
    image_size: Tuple[int, int],
    device: str = 'cuda'
) -> List[torch.Tensor]:
    """
    Generate background mask video: background is white (255), foreground is black (0)
    
    Returns:
        mask_frames: List of (H, W) grayscale frames
    """
    H, W = image_size
    num_frames = max(masks_per_frame.keys()) + 1
    
    mask_frames = []
    for t in range(num_frames):
        # Initialized to background (white)
        bg_mask = torch.ones((H, W), dtype=torch.uint8, device=device) * 255
        
        # Set foreground area to black
        if t in masks_per_frame:
            for obj_id, mask in masks_per_frame[t].items():
                mask_tensor = mask
                if mask_tensor.ndim == 3:
                    mask_tensor = mask_tensor[0]
                if mask_tensor.shape != (H, W):
                    mask_np = cv2.resize(
                        mask_tensor.cpu().numpy().astype(np.uint8), (W, H),
                        interpolation=cv2.INTER_NEAREST
                    )
                    mask_tensor = torch.from_numpy(mask_np.astype(bool)).to(device)
                bg_mask[mask_tensor] = 0
        
        mask_frames.append(bg_mask)
    
    return mask_frames


def generate_object_id_video(
    masks_per_frame: Dict[int, Dict[int, np.ndarray]],
    obj_id_to_color_idx: Dict[int, int],
    image_size: Tuple[int, int],
    device: str = 'cuda'
) -> List[torch.Tensor]:
    """
    Generate object ID video: each object is identified with a different color and the background is black
    
    Returns:
        id_frames: List of (H, W, 3) RGB frames
    """
    H, W = image_size
    num_frames = max(masks_per_frame.keys()) + 1
    
    id_frames = []
    for t in range(num_frames):
        # Initialize to black background
        id_map = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
        
        # Color each object
        if t in masks_per_frame:
            for obj_id, mask in masks_per_frame[t].items():
                mask_tensor = mask
                if mask_tensor.ndim == 3:
                    mask_tensor = mask_tensor[0]
                if mask_tensor.shape != (H, W):
                    mask_np = cv2.resize(
                        mask_tensor.cpu().numpy().astype(np.uint8), (W, H),
                        interpolation=cv2.INTER_NEAREST
                    )
                    mask_tensor = torch.from_numpy(mask_np.astype(bool)).to(device)
                
                # Use a unified color assignment function (consistent with other videos)
                color_255 = get_object_color(obj_id, obj_id_to_color_idx, device)
                
                # Coloring
                id_map[mask_tensor] = color_255
        
        id_frames.append(id_map)
    
    return id_frames


def generate_rendered_object_id_video(
    rendered_masks: List[torch.Tensor],
    obj_id_to_color_idx: Dict[int, int],
    background_mask: Optional[torch.Tensor] = None,
    device: str = 'cuda'
) -> List[torch.Tensor]:
    """
    Generate rendered object ID video: generate ID video based on rendered mask
    
    Args:
        rendered_masks: List of (H, W) rendered object masks per frame
        obj_id_to_color_idx: mapping of object ID to color index
        background_mask: (H, W) background mask, True represents the background area
        
    Returns:
        id_frames: List of (H, W, 3) RGB frames
    """
    cmap = matplotlib.colormaps['tab20']
    id_frames = []
    
    for t, mask in enumerate(rendered_masks):
        H, W = mask.shape
        # Initialized to background color (ID=0, black)
        id_map = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
        
        # The foreground area is colored according to the object ID
        if torch.any(mask):
            # Since the rendered mesh is the result of merging, we need to assign a uniform color to the entire foreground area.
            # A mixed color is used here to represent a combination of multiple objects
            if len(obj_id_to_color_idx) > 0:
                # If there are multiple objects, use the color of the first object or mixed colors
                if len(obj_id_to_color_idx) == 1:
                    # single object
                    obj_id = list(obj_id_to_color_idx.keys())[0]
                    color_idx = obj_id_to_color_idx[obj_id]
                    color_rgb = cmap(color_idx % 20)[:3]
                    color_255 = torch.tensor(
                        [int(c * 255) for c in color_rgb],
                        dtype=torch.uint8, device=device
                    )
                    id_map[mask] = color_255
                else:
                    # Multiple objects, using mixed colors or zone shading
                    # To simplify the process here, use a mixed ID for all foreground areas
                    mixed_color_idx = sum(obj_id_to_color_idx.values()) % 20
                    color_rgb = cmap(mixed_color_idx)[:3]
                    color_255 = torch.tensor(
                        [int(c * 255) for c in color_rgb],
                        dtype=torch.uint8, device=device
                    )
                    id_map[mask] = color_255
        
        id_frames.append(id_map)
    
    return id_frames


def generate_mesh_based_object_id_video(
    individual_meshes_per_frame: List[List[Meshes]],  # Change to independent mesh list for each frame
    obj_id_to_color_idx: Dict[int, int],
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_size: Tuple[int, int],
    device: str = 'cuda'
) -> List[torch.Tensor]:
    """
    Generate accurate object ID video based on independent mesh rendering
    
    Args:
        individual_meshes_per_frame: List[List[Meshes]] Individual mesh list per frame
        obj_id_to_color_idx: mapping of object ID to color index
        intrinsics: (T, 3, 3) camera intrinsics
        extrinsics: (T, 4, 4) camera extrinsics
        image_size: (H, W) image size
        
    Returns:
        id_frames: List of (H, W, 3) RGB frames
    """
    id_frames = []
    H, W = image_size
    
    # Get a list of object IDs (in order)
    obj_ids = list(obj_id_to_color_idx.keys())
    
    for t, individual_meshes in enumerate(individual_meshes_per_frame):
        # Initialized to background color (ID=0, black)
        id_map = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
        
        # Assign a different color to each individual mesh
        for mesh_idx, mesh in enumerate(individual_meshes):
            if mesh is not None and mesh.num_verts_per_mesh().sum().item() > 0:
                # Render a single mesh to get the mask
                _, _, mask = render_meshes_pytorch3d(
                    mesh, intrinsics[t], extrinsics[t], image_size
                )
                
                # Color the current mesh area
                if torch.any(mask):
                    # Determine object ID
                    if mesh_idx < len(obj_ids):
                        obj_id = obj_ids[mesh_idx]
                    else:
                        # If the number of mesh exceeds the number of obj_id, use circular index
                        obj_id = obj_ids[mesh_idx % len(obj_ids)]
                    
                    # Use a unified color assignment function (consistent with other videos)
                    color_255 = get_object_color(obj_id, obj_id_to_color_idx, device)
                    
                    # Use the current mesh mask to color, overwriting the previous color.
                    id_map[mask] = color_255
        
        id_frames.append(id_map)
    
    return id_frames


def generate_rendered_mask_video(
    background_masks: List[torch.Tensor],
    foreground_masks: List[torch.Tensor],
    device: str = 'cuda'
) -> List[torch.Tensor]:
    """
    Generate a rendered mask video: the background mask is inverted and the foreground mask is merged
    
    Args:
        background_masks: List of (H, W) background rendering mask, True means there are background pixels
        foreground_masks: List of (H, W) foreground rendering masks, True means there are foreground pixels
        
    Returns:
        mask_frames: List of (H, W) combined masks, True means there is content (background or foreground)
    """
    mask_frames = []
    
    for bg_mask, fg_mask in zip(background_masks, foreground_masks):
        # The background mask is inverted: the original True (with background) becomes False, and the original False (empty) place becomes True.
        bg_inverted = ~bg_mask
        
        # Merge background negation mask and foreground mask
        # Final mask: Hole part of the background (True) OR foreground part (True)
        combined_mask = bg_inverted | fg_mask
        
        # Convert to uint8 format for video saving
        mask_uint8 = (combined_mask * 255).to(torch.uint8)
        
        mask_frames.append(mask_uint8)
    
    return mask_frames


def generate_rendered_mask_video_with_depth(
    background_depth_frames: List[torch.Tensor],
    foreground_depth_frames: List[torch.Tensor],
    background_masks: List[torch.Tensor],
    foreground_masks: List[torch.Tensor],
    device: str = 'cuda'
) -> List[torch.Tensor]:
    """
    Generate rendering mask video considering depth occlusion
    
    Args:
        background_depth_frames: List of (H, W) background depth frames
        foreground_depth_frames: List of (H, W) foreground depth frames
        background_masks: List of (H, W) background rendering mask, True means there are background pixels
        foreground_masks: List of (H, W) foreground rendering masks, True means there are foreground pixels
        
    Returns:
        mask_frames: List of (H, W, 3) RGB mask frames, taking into account depth occlusion
    """
    mask_frames = []
    
    for bg_depth, fg_depth, bg_mask, fg_mask in zip(
        background_depth_frames, foreground_depth_frames, background_masks, foreground_masks
    ):
        # Use the same logic as composite_by_depth to determine whether the foreground is visible
        # Conditions for the foreground to be visible: there is a foreground mask and (the background has no depth or the foreground depth is closer)
        take_fg = fg_mask & ((bg_depth <= 0) | ((fg_depth > 0) & (fg_depth < bg_depth - 1e-6)))
        
        bg_mask = ~bg_mask
        out_mask = bg_mask.clone()

        out_mask[take_fg] = fg_mask[take_fg]
        
        # Convert to RGB format for video saving (H, W, 3)
        mask_rgb = torch.stack([out_mask, out_mask, out_mask], dim=-1)
        mask_uint8 = (mask_rgb * 255).to(torch.uint8)
        
        mask_frames.append(mask_uint8)
    
    return mask_frames


def generate_gaussian_projection_video(
    gaussian_params_per_frame: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    obj_id_to_color_idx: Dict[int, int],
    intrinsics: List[np.ndarray],
    extrinsics: List[np.ndarray],
    image_size: Tuple[int, int],
    threshold: float = 0.1,
    background: str = 'white',
    device: str = 'cuda'
) -> List[torch.Tensor]:
    """
    Generate 3D Gaussian projection to 2D visualization video
    
    Args:
        gaussian_params_per_frame: Gaussian parameters of each frame, the format is {obj_id: (mean, cov)}
        obj_id_to_color_idx: mapping of object ID to color index
        intrinsics: camera intrinsic parameter list
        extrinsics: camera external parameter list
        image_size: (width, height)
        threshold: density threshold
        background: background color
        device: 'cuda' or 'cpu'
    
    Returns:
        projection_frames: List of (H, W, 3) RGB projection images
    """
    width, height = image_size
    projection_frames = []
    
    for frame_idx, gaussian_params in enumerate(gaussian_params_per_frame):
        if frame_idx >= len(intrinsics) or frame_idx >= len(extrinsics):
            break
            
        # Handle intrinsics - possibly numpy or tensor
        if isinstance(intrinsics[frame_idx], np.ndarray):
            K = torch.from_numpy(intrinsics[frame_idx]).float().to(device)
        else:
            K = intrinsics[frame_idx].float().to(device)
        
        # Handle extrinsics - possibly numpy or tensor
        if isinstance(extrinsics[frame_idx], np.ndarray):
            extrinsic = torch.from_numpy(extrinsics[frame_idx]).float().to(device)
        else:
            extrinsic = extrinsics[frame_idx].float().to(device)
        
        # Extract R and t from extrinsic
        R = extrinsic[:3, :3]  # world to camera rotation
        t = extrinsic[:3, 3:4]  # world to camera translation
        
        # Collect all density maps and IDs of the current frame
        density_maps = []
        obj_ids = []
        
        for obj_id, (mean, cov) in gaussian_params.items():
            # Make sure mean and cov are on GPU
            if not isinstance(mean, torch.Tensor):
                mean = torch.from_numpy(mean).float().to(device)
            else:
                mean = mean.to(device)
            
            if not isinstance(cov, torch.Tensor):
                cov = torch.from_numpy(cov).float().to(device)
            else:
                cov = cov.to(device)
            
            # Project 3D Gaussian to 2D
            density, z_depth = project_gaussian_to_2d_gpu(
                mean, cov, K, R, t, image_size, device
            )
            
            # Keep only the Gaussian in front of the camera
            if z_depth > 0:
                density_maps.append(density)
                obj_ids.append(obj_id)
        
        # Color and composite
        colored_frame = colorize_density_maps_alpha(
            density_maps,
            obj_ids,
            obj_id_to_color_idx,
            height,
            width,
            threshold=threshold,
            background=background,
            device=device
        )
        
        projection_frames.append(colored_frame)
    
    return projection_frames


def generate_gaussian_projection_with_alpha(
    gaussian_params_per_frame: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    obj_id_to_color_idx: Dict[int, int],
    intrinsics: List[np.ndarray],
    extrinsics: List[np.ndarray],
    image_size: Tuple[int, int],
    threshold: float = 0.05,
    device: str = 'cuda'
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Generate 3D Gaussian projection to 2D RGB and Alpha channels, accounting for depth occlusion
    Use soft alpha blending to achieve smooth edges
    
    Args:
        gaussian_params_per_frame: Gaussian parameters of each frame, the format is {obj_id: (mean, cov)}
        obj_id_to_color_idx: mapping of object ID to color index
        intrinsics: camera intrinsic parameter list
        extrinsics: camera external parameter list
        image_size: (width, height)
        threshold: density threshold (used to generate mask)
        device: 'cuda' or 'cpu'
    
    Returns:
        rgb_frames: List of (H, W, 3) RGB images (using soft alpha blending)
        alpha_frames: List of (H, W) alpha channel (based on gaussian density)
    """
    width, height = image_size
    rgb_frames = []
    alpha_frames = []
    
    for frame_idx, gaussian_params in enumerate(gaussian_params_per_frame):
        if frame_idx >= len(intrinsics) or frame_idx >= len(extrinsics):
            break
            
        # Dealing with intrinsics
        if isinstance(intrinsics[frame_idx], np.ndarray):
            K = torch.from_numpy(intrinsics[frame_idx]).float().to(device)
        else:
            K = intrinsics[frame_idx].float().to(device)
        
        # Handling extrinsics
        if isinstance(extrinsics[frame_idx], np.ndarray):
            extrinsic = torch.from_numpy(extrinsics[frame_idx]).float().to(device)
        else:
            extrinsic = extrinsics[frame_idx].float().to(device)
        
        # Extract R and t from extrinsic
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3:4]
        
        # Initialize RGB and Alpha channels
        rgb_frame = torch.zeros((height, width, 3), dtype=torch.float32, device=device)
        alpha_frame = torch.zeros((height, width), dtype=torch.float32, device=device)
        
        # Collect density map, depth and color of all objects
        density_data = []
        
        for obj_id, (mean, cov) in gaussian_params.items():
            # Make sure mean and cov are on GPU
            if not isinstance(mean, torch.Tensor):
                mean = torch.from_numpy(mean).float().to(device)
            else:
                mean = mean.to(device)
            
            if not isinstance(cov, torch.Tensor):
                cov = torch.from_numpy(cov).float().to(device)
            else:
                cov = cov.to(device)
            
            # Project 3D Gaussian to 2D
            density, z_depth = project_gaussian_to_2d_gpu(
                mean, cov, K, R, t, image_size, device
            )
            
            # Keep only the Gaussian in front of the camera
            if z_depth > 0:
                # Use a unified color assignment function (consistent with trajectories)
                color_rgb = get_object_color(obj_id, obj_id_to_color_idx, device, return_float=True)
                
                # Normalized density
                pmax = density.max()
                if pmax > 0:
                    density_norm = density / (pmax + 1e-8)
                else:
                    density_norm = density
                
                density_data.append((density_norm, color_rgb, z_depth, obj_id))
        
        # Sort by depth (from far to near), render distant objects first, then render nearby objects
        density_data.sort(key=lambda x: x[2], reverse=True)
        
        # Use soft alpha blending (refer to the implementation of colorize_density_maps_alpha)
        for density_norm, color_rgb, z_depth, obj_id in density_data:
            # Soft alpha calculation: use smooth alpha instead of hard threshold
            # alpha = (density_norm - threshold) / (1.0 - threshold + 1e-8)
            # alpha = alpha.clamp(0.0, 1.0)
            
            # A softer approach: directly use the normalized density as alpha, but remove the parts below the threshold
            alpha_new = torch.where(
                density_norm > threshold,
                (density_norm - threshold) / (1.0 - threshold + 1e-8),
                torch.zeros_like(density_norm)
            ).clamp(0.0, 1.0)
            
            # RGB blending: using alpha compositing
            # C_out = C_fg * alpha_fg + C_bg * (1 - alpha_fg)
            alpha_3d = alpha_new.unsqueeze(-1)  # (H, W, 1)
            rgb_frame = color_rgb.view(1, 1, 3) * alpha_3d + rgb_frame * (1 - alpha_3d)
            
            # Alpha blending: using over operation
            # alpha_out = alpha_fg + alpha_bg * (1 - alpha_fg)
            alpha_frame = alpha_new + alpha_frame * (1 - alpha_new)
        
        # Limit alpha to the range [0, 1]
        alpha_frame = alpha_frame.clamp(0, 1)
        
        # Convert RGB to uint8
        rgb_frame_uint8 = (rgb_frame.clamp(0, 1) * 255).to(torch.uint8)
        
        rgb_frames.append(rgb_frame_uint8)
        alpha_frames.append(alpha_frame)
    
    return rgb_frames, alpha_frames


def composite_gaussian_with_background_alpha(
    gaussian_rgb_frames: List[torch.Tensor],
    gaussian_alpha_frames: List[torch.Tensor],
    background_frames: List[torch.Tensor]
) -> List[torch.Tensor]:
    """
    Merge gaussian projection with background using alpha channel
    
    Args:
        gaussian_rgb_frames: List of (H, W, 3) uint8, Gaussian projected RGB
        gaussian_alpha_frames: List of (H, W) float32, Gaussian alpha channel
        background_frames: List of (H, W, 3) uint8, background RGB frames
    
    Returns:
        merged_frames: List of (H, W, 3) uint8, merged RGB frames
    """
    assert len(gaussian_rgb_frames) == len(gaussian_alpha_frames) == len(background_frames), \
        "all input lists must have the same length"
    
    merged_frames = []
    
    for gaussian_rgb, alpha, bg_rgb in zip(gaussian_rgb_frames, gaussian_alpha_frames, background_frames):
        # Make sure devices are consistent
        device = gaussian_rgb.device
        bg_rgb = bg_rgb.to(device)
        
        # Convert to float [0, 1]
        gaussian_rgb_f = gaussian_rgb.float() / 255.0  # (H, W, 3)
        bg_rgb_f = bg_rgb.float() / 255.0  # (H, W, 3)
        
        # Alpha blending: C_out = C_fg * alpha + C_bg * (1 - alpha)
        alpha_3d = alpha.unsqueeze(-1)  # (H, W, 1)
        merged_f = gaussian_rgb_f * alpha_3d + bg_rgb_f * (1 - alpha_3d)
        
        # Convert back to uint8
        merged_uint8 = (merged_f.clamp(0, 1) * 255).to(torch.uint8)
        merged_frames.append(merged_uint8)
    
    return merged_frames


# ============================================================================
# Video saving tool
# ============================================================================

def save_video_from_frames(
    frames: List[torch.Tensor],
    output_path: Path,
    fps: int = 10
):
    """save video"""
    if len(frames) == 0:
        logger.warning(f"No frames to save for {output_path}")
        return
    
    # Make sure all frames have the same dimensions
    first_frame = frames[0]
    if first_frame.ndim == 2:
        # Grayscale -> RGB
        frames = [f.unsqueeze(-1).repeat(1, 1, 3) for f in frames]
    
    # Stack frames
    frames_tensor = torch.stack(frames)
    
    # save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(
        str(output_path),
        frames_tensor.cpu(),
        fps=fps,
        video_codec='h264',
        options={"crf": "18"}
    )
    # logger.info(f"✓ Save video: {output_path}")


def save_segmentation_masks_video(
    video_segments: Dict[int, Dict[int, torch.Tensor]],
    output_path: Path,
    num_frames: int,
    image_size: Tuple[int, int],
    fps: int = 10,
    device: str = 'cuda'
):
    """
    Save segmentation mask as mp4 video
    
    Args:
        video_segments: {frame_idx: {obj_id: mask (H,W)}}
        output_path: output video path
        num_frames: total number of frames
        image_size: image size (H, W)
        fps: frame rate
        device: device
    """
    if not video_segments or len(video_segments) == 0:
        logger.warning(f"No segmentation masks to save for {output_path}")
        return
    
    H, W = image_size
    mask_frames = []
    
    # Count all object IDs
    all_obj_ids = set()
    for frame_masks in video_segments.values():
        all_obj_ids.update(frame_masks.keys())
    
    # Use white for all objects
    white_color = torch.tensor([255, 255, 255], dtype=torch.uint8, device=device)
    
    # Generate a mask image (white) for each frame
    for t in range(num_frames):
        frame_mask = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
        
        if t in video_segments:
            masks = video_segments[t]
            for obj_id, mask in masks.items():
                # Process the shape of the mask: if it is 3-dimensional, remove the first dimension
                if mask.ndim == 3:
                    mask = mask.squeeze(0)  # [1, H, W] -> [H, W]
                
                # Convert mask to boolean type and apply white color
                mask_bool = mask.bool()
                frame_mask[mask_bool] = white_color
        
        mask_frames.append(frame_mask)
    
    # save video
    save_video_from_frames(mask_frames, output_path, fps)
    logger.info(f"✓ Segmentation mask video saved: {output_path} ({len(all_obj_ids)} objects total)")


def colorize_depth_for_video(
    depth_frames: List[torch.Tensor],
    cmap: str = 'Spectral'
) -> List[torch.Tensor]:
    """
    Convert depth map to color visualization
    
    Returns:
        colored_frames: List of (H, W, 3) RGB frames
    """
    colored_frames = []
    
    # Compute global depth range
    valid_depth_tensors = [d[d > 0].flatten() for d in depth_frames if torch.any(d > 0)]
    if valid_depth_tensors:
        all_depths = torch.cat(valid_depth_tensors)
        
        # If the data is too large, sample
        if len(all_depths) > 1000000:  # 1M points
            indices = torch.randperm(len(all_depths))[:1000000]
            sampled_depths = all_depths[indices]
        else:
            sampled_depths = all_depths
        
        try:
            min_depth = torch.quantile(sampled_depths, 0.001)
            max_depth = torch.quantile(sampled_depths, 0.99)
        except RuntimeError:
            # If quantile still fails, use min/max as fallback
            min_depth = torch.min(sampled_depths)
            max_depth = torch.max(sampled_depths)
    else:
        min_depth, max_depth = 0.0, 1.0
    
    colormap = matplotlib.colormaps[cmap]
    
    for depth in depth_frames:
        # Convert to parallax
        disp = torch.where(depth > 0, 1.0 / depth, torch.tensor(float('nan'), device=depth.device))
        
        # normalization
        min_disp = 1.0 / max_depth if max_depth > 0 else 0
        max_disp = 1.0 / min_depth if min_depth > 0 else 1
        disp_norm = (disp - min_disp) / (max_disp - min_disp + 1e-8)
        disp_norm = torch.clamp(disp_norm, 0, 1)
        
        # Coloring
        disp_np = disp_norm.cpu().numpy()
        colored_np = np.nan_to_num(colormap(1.0 - disp_np)[..., :3], 0)
        colored_np = (colored_np * 255).astype(np.uint8)
        colored = torch.from_numpy(colored_np).to(depth.device)
        
        colored_frames.append(colored)
    
    return colored_frames


def grayscale_depth_for_video(
    depth_frames: List[torch.Tensor],
    global_min_depth: Optional[float] = None,
    global_max_depth: Optional[float] = None
) -> List[torch.Tensor]:
    """
    Convert depth map to black and white visualization, the closer it is, the whiter it is
    
    Args:
        depth_frames: depth frame list
        global_min_depth: Global minimum depth value (optional), if provided this value is used instead of calculated from data
        global_max_depth: global maximum depth value (optional), if provided this value is used rather than calculated from the data
    
    Returns:
        grayscale_frames: List of (H, W, 3) RGB frames (grayscale)
    """
    grayscale_frames = []
    
    # If no global depth range is provided, computes
    if global_min_depth is None or global_max_depth is None:
        valid_depth_tensors = [d[d > 0].flatten() for d in depth_frames if torch.any(d > 0)]
        if valid_depth_tensors:
            all_depths = torch.cat(valid_depth_tensors)
            
            # If the data is too large, sample
            if len(all_depths) > 1000000:  # 1M points
                indices = torch.randperm(len(all_depths))[:1000000]
                sampled_depths = all_depths[indices]
            else:
                sampled_depths = all_depths
            
            try:
                min_depth = torch.quantile(sampled_depths, 0.001)
                max_depth = torch.quantile(sampled_depths, 0.99)
            except RuntimeError:
                # If quantile still fails, use min/max as fallback
                min_depth = torch.min(sampled_depths)
                max_depth = torch.max(sampled_depths)
        else:
            min_depth, max_depth = 0.0, 1.0
    else:
        min_depth = global_min_depth
        max_depth = global_max_depth
    
    for depth in depth_frames:
        # Convert depth to disparity (larger values ​​the closer you get)
        disp = torch.where(depth > 0, 1.0 / depth, torch.tensor(0.0, device=depth.device))
        
        # Normalizes the disparity to the range [0, 1], the closer the value is (larger the disparity), the closer the value is to 1
        if max_depth > 0 and min_depth > 0:
            min_disp = 1.0 / max_depth
            max_disp = 1.0 / min_depth
            disp_norm = (disp - min_disp) / (max_disp - min_disp + 1e-8)
        else:
            disp_norm = disp
        
        disp_norm = torch.clamp(disp_norm, 0, 1)
        
        # The closer it is, the whiter it is: directly use the normalized disparity value as the grayscale value
        # disp_norm = 1 means closest (white), disp_norm = 0 means farthest or background (black)
        gray_value = (disp_norm * 255).to(torch.uint8)
        
        # Convert to 3-channel RGB format (keep grayscale)
        gray_rgb = gray_value.unsqueeze(-1).repeat(1, 1, 3)
        
        grayscale_frames.append(gray_rgb)
    
    return grayscale_frames


def compute_global_depth_range(
    depth_frames_list: List[List[torch.Tensor]]
) -> Tuple[float, float]:
    """
    Compute global depth range from multiple list of depth frames
    
    Args:
        depth_frames_list: A list of multiple depth frame lists (for example: [bg_depths, fg_depths, combined_depths])
    
    Returns:
        (global_min_depth, global_max_depth): global minimum and maximum depth values
    """
    all_valid_depths = []
    
    for depth_frames in depth_frames_list:
        valid_depth_tensors = [d[d > 0].flatten() for d in depth_frames if torch.any(d > 0)]
        if valid_depth_tensors:
            all_valid_depths.extend(valid_depth_tensors)
    
    if not all_valid_depths:
        return 0.0, 1.0
    
    # Combine all valid depths
    combined_depths = torch.cat(all_valid_depths)
    
    # If the data is too large, sample
    if len(combined_depths) > 1000000:  # 1M points
        indices = torch.randperm(len(combined_depths))[:1000000]
        sampled_depths = combined_depths[indices]
    else:
        sampled_depths = combined_depths
    
    try:
        min_depth = torch.quantile(sampled_depths, 0.001).item()
        max_depth = torch.quantile(sampled_depths, 0.99).item()
    except RuntimeError:
        # If quantile fails, use min/max as fallback
        min_depth = torch.min(sampled_depths).item()
        max_depth = torch.max(sampled_depths).item()
    
    return min_depth, max_depth


def save_parameters_to_json(
    gaussian_params_per_frame: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    center_points_per_frame: Dict[int, Dict[int, torch.Tensor]],
    obj_id_to_color_idx: Dict[int, int],
    obb_scale_factor: float,
    output_path: Path,
    intrinsics: torch.Tensor = None,
    extrinsics: torch.Tensor = None,
    device: str = 'cuda'
):
    """
    Save 3D Gaussian, 3D bounding box and trajectory parameters to JSON file
    
    Args:
        gaussian_params_per_frame: Gaussian parameters per frame [{obj_id: (mean, cov)}]
        center_points_per_frame: center point of each frame {frame_idx: {obj_id: center_point}}
        obj_id_to_color_idx: mapping of object ID to color index
        obb_scale_factor: OBB scaling factor
        output_path: output JSON file path
        intrinsics: camera intrinsic parameter matrix (T, 3, 3)
        extrinsics: camera extrinsic matrix (T, 4, 4)
        device: device
    """
    # logger.info("=" * 80)
    # logger.info("Save 3D Gaussian, Bounding Box and trajectory parameters to JSON")
    # logger.info("=" * 80)
    
    # Build output data structure
    output_data = {
        "metadata": {
            "num_frames": len(gaussian_params_per_frame),
            "num_objects": len(obj_id_to_color_idx),
            "obb_scale_factor": obb_scale_factor,
            "object_ids": list(obj_id_to_color_idx.keys())
        },
        "frames": []
    }
    
    # Iterate through each frame
    for frame_idx, gaussian_params in enumerate(gaussian_params_per_frame):
        frame_data = {
            "frame_index": frame_idx,
            "objects": []
        }
        
        # Traverse each object in the current frame
        for obj_id, (mean, cov) in gaussian_params.items():
            # Convert Tensor to numpy and then to list
            mean_np = mean.cpu().numpy().tolist()
            cov_np = cov.cpu().numpy().tolist()
            
            # Calculate OBB parameters
            center, extents, rotation = compute_obb_from_gaussian(
                mean, cov, scale_factor=obb_scale_factor
            )
            center_np = center.cpu().numpy().tolist()
            extents_np = extents.cpu().numpy().tolist()
            rotation_np = rotation.cpu().numpy().tolist()
            
            # Get the trajectory center point (same as Gaussian mean, but also saved for completeness)
            trajectory_center = None
            trajectory_2d_with_depth = None  # [x, y, depth]
            
            if frame_idx in center_points_per_frame and obj_id in center_points_per_frame[frame_idx]:
                trajectory_center = center_points_per_frame[frame_idx][obj_id].cpu().numpy().tolist()
                
                # Calculate 2D projected coordinates and depth (corresponding to points in the visualization video)
                if intrinsics is not None and extrinsics is not None:
                    center_3d = center_points_per_frame[frame_idx][obj_id]
                    intrinsic = intrinsics[frame_idx]
                    extrinsic = extrinsics[frame_idx]
                    point_2d, depth = project_3d_point_to_2d(center_3d, intrinsic, extrinsic)
                    trajectory_2d_with_depth = [
                        float(point_2d[0].item()),  # x coordinate
                        float(point_2d[1].item()),  # y coordinate
                        float(depth)                # depth depth
                    ]
            
            obj_data = {
                "object_id": int(obj_id),
                "color_index": int(obj_id_to_color_idx[obj_id]),
                "gaussian_3d": {
                    "mean": mean_np,  # [x, y, z]
                    "covariance": cov_np  # [[c11, c12, c13], [c21, c22, c23], [c31, c32, c33]]
                },
                "bounding_box_3d": {
                    "center": center_np,  # [x, y, z]
                    "extents": extents_np,  # [half_length_x, half_length_y, half_length_z]
                    "rotation": rotation_np  # 3x3 rotation matrix
                },
                "trajectory": {
                    "center": trajectory_center,  # [x, y, z] 3D world coordinates
                    "center_2d_with_depth": trajectory_2d_with_depth  # [x, y, depth] 2D projection coordinates + depth (corresponding to visualization point)
                }
            }
            
            frame_data["objects"].append(obj_data)
        
        output_data["frames"].append(frame_data)
    
    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # logger.info(f"✓ Parameters saved to: {output_path}")
    # logger.info(f" - Total number of frames: {len(gaussian_params_per_frame)}")
    # logger.info(f" - Number of objects: {len(obj_id_to_color_idx)}")
    # logger.info("=" * 80)


# ============================================================================
# Main processing flow
# ============================================================================

def process_single_video(
    video_path: str,
    npz_path: str,
    output_dir: Path,
    args,
    grounding_model,
    video_predictor,
    image_predictor,
    device: str
):
    """
    Complete process for processing a single video
    """
    # logger.info("=" * 80)
    logger.info(f"Processing video: {video_path}")
    # logger.info(f"NPZ file: {npz_path}")
    # logger.info("=" * 80)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if it has been processed
    check_file = output_dir / "background_gaussian_projection.mp4"
    if check_file.exists():
        logger.info(f"✓ Video already processed, skipping: {output_dir}")
        # return str(output_dir)
    
    # ========== Step 1: Grounded SAM2 Segmentation ==========
    video_segments = run_grounded_sam2_segmentation(
        video_path, output_dir,
        grounding_model, video_predictor, image_predictor,
        device, args.max_frames
    )
    
    # Check if object is detected
    has_objects = video_segments and len(video_segments) > 0
    
    if not has_objects:
        logger.warning(f"⚠ No objects detected in video {Path(video_path).name}, rendering background only")
        # Save failure record
        no_detection_log = output_dir / "no_detection.txt"
        with open(no_detection_log, 'w') as f:
            f.write(f"No objects detected in video: {video_path}\n")
    
    # ========== Step 2: Load NPZ data ==========
    # logger.info("=" * 80)
    # logger.info("Step 2: Load depth data and camera parameters")
    # logger.info("=" * 80)
    
    data = np.load(npz_path)
    depths = torch.from_numpy(data['depths'][:args.max_frames].astype(np.float32)).to(device)
    intrinsics = torch.from_numpy(data['intrinsics'][:args.max_frames].astype(np.float32)).to(device)
    extrinsics = torch.from_numpy(data['extrinsics'][:args.max_frames].astype(np.float32)).to(device)
    
    H, W = depths.shape[1:]
    T = depths.shape[0]
    logger.info(f"✓ Depth data: {depths.shape}, image size: {W}x{H}, frames: {T}")
    
    # ========== Save segmentation mask video ==========
    if has_objects:
        save_segmentation_masks_video(
            video_segments=video_segments,
            output_path=output_dir / "segmentation_masks.mp4",
            num_frames=T,
            image_size=(H, W),
            fps=args.fps,
            device=device
        )
    
    # Read the first frame RGB
    video_tensor, _, _ = read_video(video_path) # start_pts=0, end_pts=1, pts_unit='sec'
    first_frame_rgb_tensor = video_tensor[0].to(device)
    
    # ========== Step 3: Process each frame - generate point cloud, fit Gaussian, calculate OBB ==========
    # logger.info("=" * 80)
    # logger.info("Step 3: Frame-by-frame processing - point cloud generation, Gaussian fitting, OBB calculation")
    # logger.info("=" * 80)
    
    # data container
    background_points_3d = None
    background_colors = None
    
    if has_objects:
        # Only initialize the foreground related data structure when there is an object
        reference_covariances = {}
        obj_id_to_color_idx = {}
        next_color_idx = 0
        
        # Ellipsoid data (step 4)
        ellipsoid_points_per_frame = []  # legacy points (kept for compatibility)
        ellipsoid_colors_per_frame = []
        ellipsoid_meshes_per_frame: List[Optional[Meshes]] = []  # Merged mesh (for rendering)
        individual_ellipsoid_meshes_per_frame: List[List[Meshes]] = []  # Independent mesh (for ID video)

        # Trajectory data (step 6)
        center_points_per_frame: Dict[int, Dict[int, torch.Tensor]] = {}  # {frame_idx: {obj_id: center_point}}
        
        # 3D Gaussian parameters (for projection visualization)
        gaussian_params_per_frame: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = []  # {obj_id: (mean, cov)}
    
    for t in tqdm(range(T), desc="Processing frames"):
        depth = depths[t]
        intrinsic = intrinsics[t]
        extrinsic = extrinsics[t]
        
        # Generate point cloud
        pc = get_point_cloud_from_depth_cuda(depth, intrinsic, extrinsic)

        # First frame: Extract background point cloud (required whether there are objects or not)
        if t == 0:
            masks = video_segments.get(t, {}) if has_objects else {}
            background_points_3d, background_colors, bg_mask = \
                get_background_pointcloud_from_masks(masks, pc, first_frame_rgb_tensor, H, W, device)
            # logger.info(f"✓ Background point cloud: {len(background_points_3d)} points")
        
        # Only process the foreground when there are objects
        if not has_objects:
            continue
        
        # Extract object point cloud (apply mask erosion to remove boundary noise)
        masks = video_segments.get(t, {})
        object_pcs = extract_object_point_clouds_gpu(
            pc, masks, depth, device,
            erode_kernel_size=args.mask_erode_kernel,
            erode_iterations=args.mask_erode_iterations
        )

        # Fitting 3D Gaussian
        frame_ellipsoid_points = []
        frame_ellipsoid_colors = []
        frame_ellipsoid_meshes: List[Meshes] = []
        frame_gaussian_params: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}  # Gaussian parameters of the current frame

        for obj_id, obj_pc in object_pcs.items():
            # Assign color index to new object
            if obj_id not in obj_id_to_color_idx:
                obj_id_to_color_idx[obj_id] = next_color_idx
                next_color_idx += 1
            
            # Fit Gaussian
            if t == 0:
                mean, cov = fit_3d_gaussian_gpu(obj_pc, device=device)
                if mean is not None:
                    reference_covariances[obj_id] = cov
            else:
                mean, cov = fit_3d_gaussian_gpu(
                    obj_pc,
                    use_consistent_shape=True,
                    reference_covariance=reference_covariances.get(obj_id),
                    device=device
                )
            
            if mean is None:
                continue
            
            # Save Gaussian parameters for projection visualization
            # IMPORTANT: Apply the same scale_factor as the ellipsoid mesh for consistency
            # The covariance matrix needs to be enlarged by scale_factor^2 times because cov = V * diag(eigenvalues) * V^T
            # When the semi-axis is enlarged by scale_factor times, eigenvalues ​​needs to be enlarged by scale_factor^2 times
            # scaled_cov = cov * (args.ellipsoid_scale_factor ** 2)
            scaled_cov = cov
            frame_gaussian_params[obj_id] = (mean.clone(), scaled_cov.clone())
            
            # Save center points for trajectory generation
            if t not in center_points_per_frame:
                center_points_per_frame[t] = {}
            center_points_per_frame[t][obj_id] = mean.clone()
            
            # Get object color
            obj_color = get_object_color(obj_id, obj_id_to_color_idx, device)
            
            # Step 4: Generate ellipsoid (point cloud + mesh)
            ellipsoid_pts = sample_ellipsoid_points_gpu(
                mean, cov, args.ellipsoid_samples, device
            )

            ellipsoid_clrs = obj_color.unsqueeze(0).repeat(len(ellipsoid_pts), 1)
            frame_ellipsoid_points.append(ellipsoid_pts)
            frame_ellipsoid_colors.append(ellipsoid_clrs)
            # mesh
            ellipsoid_mesh = make_ellipsoid_mesh(
                mean, cov,
                scale_factor=args.ellipsoid_scale_factor,
                subdivisions=int(args.ellipsoid_subdiv),
                color_rgb255=obj_color,
                device=device,
            )
            frame_ellipsoid_meshes.append(ellipsoid_mesh)
        
        # Save the Gaussian parameters of the current frame
        gaussian_params_per_frame.append(frame_gaussian_params)
        
        # Merge all objects in the current frame
        if len(frame_ellipsoid_points) > 0:
            ellipsoid_points_per_frame.append(torch.cat(frame_ellipsoid_points, dim=0))
            ellipsoid_colors_per_frame.append(torch.cat(frame_ellipsoid_colors, dim=0))
            ellipsoid_meshes_per_frame.append(combine_meshes_for_scene(frame_ellipsoid_meshes))
            individual_ellipsoid_meshes_per_frame.append(frame_ellipsoid_meshes)  # Save independent mesh
        else:
            ellipsoid_points_per_frame.append(torch.empty((0, 3), device=device))
            ellipsoid_colors_per_frame.append(torch.empty((0, 3), dtype=torch.uint8, device=device))
            ellipsoid_meshes_per_frame.append(None)
            individual_ellipsoid_meshes_per_frame.append([])  # empty list
    
    # logger.info(f"✓ Processing completed {T} frames")
    
    if has_objects:
        # ========== Save parameters to JSON file ==========
        save_parameters_to_json(
            gaussian_params_per_frame=gaussian_params_per_frame,
            center_points_per_frame=center_points_per_frame,
            obj_id_to_color_idx=obj_id_to_color_idx,
            obb_scale_factor=args.obb_scale_factor,
            output_path=output_dir / "parameters.json",
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            device=device
        )
    else:
        # When there are no objects, create an empty mesh list
        ellipsoid_meshes_per_frame = [None] * T
    
    # ========== Step 9: Ellipsoid scene rendering (background only) ==========
    # logger.info("=" * 80)
    # logger.info("Step 9: Ellipsoid scene rendering - background only")
    # logger.info("=" * 80)
    start_t = time()
    ellipsoid_bg_rgb_frames, ellipsoid_bg_depth_frames, bg_masks_bg, fg_masks_bg = render_frame_sequence_mesh_composited(
        background_points_3d, background_colors,
        ellipsoid_meshes_per_frame,
        intrinsics, extrinsics, (H, W),
        mode='background', point_size=args.point_size, device=device, 
        batch_size=args.render_batch_size, use_fp16=args.use_fp16, pin_memory=args.pin_memory
    )
    # print(f"Rendering Time: {time() - start_t:.2f} seconds")

    save_video_from_frames(ellipsoid_bg_rgb_frames, output_dir / "background_RGB.mp4", args.fps)
    
    # Generate a mask video of the background ellipsoid (background negation)
    ellipsoid_bg_mask_frames = [mask.to(torch.uint8) * 255 for mask in bg_masks_bg]
    save_video_from_frames(ellipsoid_bg_mask_frames, output_dir / "background_mask.mp4", args.fps)

    
    # ========== Step 8: Ellipsoid scene rendering (foreground only) ==========
    # logger.info("=" * 80)
    # logger.info("Step 8: Ellipsoid scene rendering - foreground only")
    # logger.info("=" * 80)
    # start_t = time()
    if has_objects:
        ellipsoid_fg_rgb_frames, ellipsoid_fg_depth_frames, bg_masks_fg, fg_masks_fg = render_frame_sequence_mesh_composited(
            background_points_3d, background_colors,
            ellipsoid_meshes_per_frame,
            intrinsics, extrinsics, (H, W),
            mode='foreground', point_size=args.point_size, device=device, 
            batch_size=args.render_batch_size, use_fp16=args.use_fp16, pin_memory=args.pin_memory
        )
    else:
        # Generates empty foreground frame (black screen)
        ellipsoid_fg_rgb_frames = [torch.zeros((H, W, 3), dtype=torch.uint8, device=device) for _ in range(T)]
        ellipsoid_fg_depth_frames = [torch.zeros((H, W), dtype=torch.float32, device=device) for _ in range(T)]
        bg_masks_fg = [torch.ones((H, W), dtype=torch.bool, device=device) for _ in range(T)]
        fg_masks_fg = [torch.zeros((H, W), dtype=torch.bool, device=device) for _ in range(T)]
    # print(f"Rendering Time ellipsoid: {time() - start_t:.2f} seconds")
    save_video_from_frames(ellipsoid_fg_rgb_frames, output_dir / "foreground_ellipsoid_rgb.mp4", args.fps)
    
    # Generate the ID and mask video of the foreground ellipsoid
    # ellipsoid_fg_id_frames = generate_mesh_based_object_id_video(
    #     individual_ellipsoid_meshes_per_frame, obj_id_to_color_idx,
    #     intrinsics, extrinsics, (H, W), device=device
    # )
    # save_video_from_frames(ellipsoid_fg_id_frames, output_dir / "foreground_ellipsoid_id.mp4", args.fps)
    ellipsoid_fg_mask_frames = [mask.to(torch.uint8) * 255 for mask in fg_masks_fg]
    save_video_from_frames(ellipsoid_fg_mask_frames, output_dir / "foreground_ellipsoid_mask.mp4", args.fps)
    
    # ========== Calculate global depth range ==========
    # Before all depth visualizations, the global depth range is calculated to ensure that all videos use the same normalized range.
    # logger.info("=" * 80)
    # logger.info("Calculate global depth range (used to uniformly normalize all depth videos)")
    # logger.info("=" * 80)
    
    # Synthetic background + foreground depth frames
    ellipsoid_combined_depth_frames = []
    for bg_depth, fg_depth, bg_mask, fg_mask in zip(
        ellipsoid_bg_depth_frames, ellipsoid_fg_depth_frames, bg_masks_bg, fg_masks_fg
    ):
        combined_depth = torch.where(fg_mask.bool(), fg_depth, bg_depth)
        ellipsoid_combined_depth_frames.append(combined_depth)
    
    # Compute global depth range from essential depth streams
    global_min_depth, global_max_depth = compute_global_depth_range([
        ellipsoid_bg_depth_frames,
        ellipsoid_fg_depth_frames,
        ellipsoid_combined_depth_frames,
    ])
    
    # logger.info(f"✓ Global depth range: min={global_min_depth:.4f}, max={global_max_depth:.4f}")
    
    # ========== Generate all depth visualization videos using global depth range ==========
    # logger.info("=" * 80)
    # logger.info("Generate all depth visualization videos (using a unified depth range)")
    # logger.info("=" * 80)
    
    # Ellipsoid background
    depth_grayscale = grayscale_depth_for_video(
        ellipsoid_bg_depth_frames, global_min_depth, global_max_depth
    )
    save_video_from_frames(depth_grayscale, output_dir / "background_depth.mp4", args.fps)
    
    # Ellipsoid foreground
    depth_grayscale = grayscale_depth_for_video(
        ellipsoid_fg_depth_frames, global_min_depth, global_max_depth
    )
    save_video_from_frames(depth_grayscale, output_dir / "3D_gaussian_depth.mp4", args.fps)
    
    # logger.info("✓ All foreground and background depth videos have been generated")
    
    # ========== Step 7: Ellipsoid scene rendering (background + foreground) ==========
    # logger.info("=" * 80)
    # logger.info("Step 7: Ellipsoid scene rendering - background + foreground (reuse the rendered results)")
    # logger.info("=" * 80)
    
    # Directly synthesize the rendered background and foreground to avoid repeated rendering
    start_t = time()
    rgb_frames, depth_frames, bg_masks, fg_masks = composite_rendered_sequences(
        ellipsoid_bg_rgb_frames, ellipsoid_bg_depth_frames, bg_masks_bg,
        ellipsoid_fg_rgb_frames, ellipsoid_fg_depth_frames, fg_masks_fg
    )
    # print(f"Compositing Time: {time() - start_t:.2f} seconds")

    save_video_from_frames(rgb_frames, output_dir / "background_foreground_ellipsoid_rgb.mp4", args.fps)
    depth_grayscale = grayscale_depth_for_video(depth_frames, global_min_depth, global_max_depth)
    save_video_from_frames(depth_grayscale, output_dir / "background_foreground_ellipsoid_depth_bw.mp4", args.fps)
    
    # Generate the ID and mask video of the ellipsoid
    # ellipsoid_id_frames = generate_mesh_based_object_id_video(
    #     individual_ellipsoid_meshes_per_frame, obj_id_to_color_idx, 
    #     intrinsics, extrinsics, (H, W), device=device
    # )
    # save_video_from_frames(ellipsoid_id_frames, output_dir / "background_foreground_ellipsoid_id.mp4", args.fps)
    ellipsoid_mask_frames = generate_rendered_mask_video_with_depth(
        ellipsoid_bg_depth_frames, ellipsoid_fg_depth_frames, bg_masks, fg_masks, device=device
    )
    save_video_from_frames(ellipsoid_mask_frames, output_dir / "merged_mask.mp4", args.fps)

    # ========== Step 16: 3D Gaussian projection visualization video ==========
    # logger.info("=" * 80)
    # logger.info("Step 16: Generate 3D Gaussian projection visualization video (with alpha channel)")
    # logger.info("=" * 80)
    
    if has_objects:
        start_t = time()
        # Generate gaussian projection with alpha channel
        gaussian_rgb_frames, gaussian_alpha_frames = generate_gaussian_projection_with_alpha(
            gaussian_params_per_frame,
            obj_id_to_color_idx,
            intrinsics,
            extrinsics,
            (W, H),  # image_size is (width, height)
            threshold=args.gaussian_mask_threshold,  # Use threshold to generate mask
            device=device
        )
        
        # Save pure gaussian projection video (black background version, for visualization)
        # Simulate black background by applying alpha to RGB
        gaussian_projection_frames = []
        for rgb, alpha in zip(gaussian_rgb_frames, gaussian_alpha_frames):
            # Apply alpha channel to RGB (background is black)
            alpha_3d = alpha.unsqueeze(-1)  # (H, W, 1)
            rgb_f = rgb.float() / 255.0
            result = (rgb_f * alpha_3d * 255).to(torch.uint8)
            gaussian_projection_frames.append(result)
        
        save_video_from_frames(
            gaussian_projection_frames, 
            output_dir / "3D_gaussian_RGB.mp4", 
            args.fps
        )
        # logger.info(f"✓ 3D Gaussian projection video has been generated (time: {time() - start_t:.2f} seconds)")
    else:
        # Generating empty Gaussian projection video (black screen)
        gaussian_rgb_frames = [torch.zeros((H, W, 3), dtype=torch.uint8, device=device) for _ in range(T)]
        gaussian_alpha_frames = [torch.zeros((H, W), dtype=torch.float32, device=device) for _ in range(T)]
        gaussian_projection_frames = [torch.zeros((H, W, 3), dtype=torch.uint8, device=device) for _ in range(T)]
        save_video_from_frames(
            gaussian_projection_frames, 
            output_dir / "3D_gaussian_RGB.mp4", 
            args.fps
        )
    
    # ========== Step 17: Merge gaussian_projection with background RGB video using alpha channel ==========
    # logger.info("=" * 80)
    # logger.info("Step 17: Combine gaussian_projection and ellipsoid background rendering using alpha channel")
    # logger.info("=" * 80)
    
    start_t = time()
    # Use alpha channel for smooth blending
    gaussian_with_bg_frames = composite_gaussian_with_background_alpha(
        gaussian_rgb_frames,
        gaussian_alpha_frames,
        ellipsoid_bg_rgb_frames
    )
    # print(f"Compositing Time Gaussian: {time() - start_t:.2f} seconds")
    # save_video_from_frames(
    #     gaussian_with_bg_frames,
    #     output_dir / "background+gaussian_projection.mp4",
    #     args.fps
    # )

    # Generate fg_masks from gaussian density (density > 0.01)
    gaussian_fg_masks = [alpha > 0.001 for alpha in gaussian_alpha_frames]

    # Consider depth for compositing (optional)
    gaussian_with_bg_frames, _, _, _ = composite_rendered_sequences(
        ellipsoid_bg_rgb_frames, ellipsoid_bg_depth_frames, bg_masks_bg,
        gaussian_with_bg_frames, ellipsoid_fg_depth_frames, gaussian_fg_masks
    )

    save_video_from_frames(
        gaussian_with_bg_frames,
        output_dir / "background_gaussian_projection.mp4",
        args.fps
    )
    # logger.info(f"✓ Gaussian projection + background video has been generated (time: {time() - start_t:.2f} seconds)")
    
    # ========== Clean up temporary files ==========
    temp_frame_dir = output_dir / "custom_video_frames_temp"
    save_tracking_results_dir = output_dir / "annotated_frames"
    if temp_frame_dir.exists():
        shutil.rmtree(temp_frame_dir)
    if save_tracking_results_dir.exists():
        shutil.rmtree(save_tracking_results_dir)
    
    # logger.info("=" * 80)
    logger.info(f"✓ Processing complete! Output directory: {output_dir}")
    # logger.info("=" * 80)
    
    return str(output_dir)


# ============================================================================
# Main function
# ============================================================================

def main():
    args = parse_args()
    
    # If ellipsoid_scale_factor is not set, calculated from probability
    if args.ellipsoid_scale_factor is None:
        args.ellipsoid_scale_factor = compute_ellipsoid_scale_factor_from_probability(
            args.ellipsoid_probability, df=3
        )
        logger.info(
            f"Computed ellipsoid_scale_factor from probability {args.ellipsoid_probability} = "
            f"{args.ellipsoid_scale_factor:.6f}"
        )
    
    # Validation parameters
    if args.max_frames < 2:
        raise ValueError(f"--max_frames must be >= 2 (got {args.max_frames}). "
                        "At least 2 frames are required to generate trajectory segments.")
    
    # Load CSV
    df = pd.read_csv(args.csv_path)
    logger.info(f"Loaded {len(df)} samples from {args.csv_path}")
    
    # Check required columns
    if 'clipPath' not in df.columns:
        raise ValueError("CSV file must contain 'clipPath' column")
    if 'crowdDensity' not in df.columns:
        raise ValueError("CSV file must contain 'crowdDensity' column")
    
    # Split data
    if args.num_parts > 1:
        part_size = (len(df) + args.num_parts - 1) // args.num_parts
        start = args.part_idx * part_size
        end = min((args.part_idx + 1) * part_size, len(df))
        df = df.iloc[start:end].copy()
        logger.info(f"Processing shard {args.part_idx}/{args.num_parts}: {len(df)} samples")
    
    if args.max_samples:
        df = df.head(args.max_samples).copy()
    
    # Load model (one-time load)
    logger.info("Loading Grounded SAM2 models...")
    grounding_model, video_predictor, image_predictor, device = load_models_and_init(
        sam2_checkpoint="checkpoints/sam2.1_hiera_large.pt",
        sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        grounding_dino_config="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint="gdino_checkpoints/groundingdino_swint_ogc.pth",
        device=args.device
    )
    logger.info("✓ Model loading complete")
    
    # Process each video
    success_count = 0
    fail_count = 0
    missing_count = 0
    
    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Processing videos"
    ):
        # Read data from CSV rows
        crowdDensity = str(row['crowdDensity'])
        video_file = str(row['clipPath'])
        npz_file = video_file.replace('.mp4', '_depth_pose.npz')
        
        # Build full path
        # sekai
        # video_id = "-FIJHCY8j-0"
        # video_file = "-FIJHCY8j-0_0018150_0019950_0000614_0000695.mp4"
        # npz_file = "-FIJHCY8j-0_0018150_0019950_0000614_0000695_depth_pose.npz"
        video_id = '_'.join(video_file.split('_')[:-4]) # Extract video ID (eg: -NNoJ26XRSo)
        full_video_path = os.path.join(args.video_root, f"vstreams_{crowdDensity}_clip", video_id, video_file)
        full_npz_path = os.path.join(args.video_root, f"vstreams_{crowdDensity}_clip", video_id, npz_file)

        # Check file exists
        if not Path(full_npz_path).is_file():
            logger.warning(f"Missing NPZ: {full_npz_path}")
            missing_count += 1
            continue
        
        if not Path(full_video_path).is_file():
            logger.warning(f"Missing video: {full_video_path}")
            missing_count += 1
            continue
        
        # Set output directory
        output_dir = Path(args.output_root) / video_id / Path(video_file).stem
        # output_dir = Path(args.output_root) / Path(video_file).stem
        
        try:
            process_single_video(
                full_video_path, full_npz_path, output_dir,
                args, grounding_model, video_predictor, image_predictor, device
            )
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to process {video_file}: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
    
    # Print statistics
    logger.info("=" * 80)
    logger.info("Processing complete!")
    logger.info(f"Success: {success_count}, Failed: {fail_count}, Missing: {missing_count}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

import torch
import argparse
from pi3.utils.basic import load_images_as_tensor
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
import os
import json
import struct
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
import numpy as np

def estimate_camera_intrinsics_from_pointcloud(
    points: np.ndarray,
    images: np.ndarray,
    camera_poses: np.ndarray | None = None,
    poses_are_cam_to_world: bool = True,
    conf: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    pixel_center: float = 0.5,
    max_points: int = 2_000_000,
    huber_delta_px: float = 3.0,
    irls_iters: int = 5,
    z_min: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    """
    Estimate a single pinhole intrinsics matrix K from a set of frames that all
    come from the same physical camera, using dense per-pixel 3D points.

    The fitting uses the projection model (no skew, no distortion):
        u = fx * Xc / Zc + cx
        v = fy * Yc / Zc + cy
    and solves for fx, cx and fy, cy with robust, confidence-weighted least squares.

    Parameters
    ----------
    points : np.ndarray
        Array of 3D points per pixel. Shape: (N, H, W, 3).
        - If `camera_poses is None`, these are assumed to already be in each
          frame's *camera coordinates* (Xc, Yc, Zc).
        - If `camera_poses` is provided, these are assumed to be in *world coordinates*,
          and they will be transformed into camera coordinates for each frame.
    images : np.ndarray
        Corresponding images. Only used for H, W and pixel indexing.
        Shape: (N, H, W, 3) or (N, H, W, C). Values are not used.
    camera_poses : np.ndarray | None
        If given, per-frame 4x4 or 3x4 SE(3) matrices describing the camera pose.
        Shape: (N, 4, 4) or (N, 3, 4).
        - If `poses_are_cam_to_world=True` (default), each pose maps camera -> world (c2w).
          We'll invert per frame to get world -> camera.
        - If False, poses are world -> camera already (w2c) and are used directly.
    poses_are_cam_to_world : bool
        See `camera_poses`. Ignored if `camera_poses is None`.
    conf : np.ndarray | None
        Optional confidence map. Shape: (N, H, W) or (N, H, W, 1).
        Used as weights (clipped to [1e-6, 1]).
    valid_mask : np.ndarray | None
        Optional boolean mask of valid pixels. Shape: (N, H, W).
        Combined with Zc > z_min and finite checks.
    pixel_center : float
        Add this offset to pixel indices to model pixel centers (0.5 is common).
    max_points : int
        Downsample the set of correspondences to at most this many to keep memory in check.
    huber_delta_px : float
        Huber delta for robust IRLS in *pixel* units.
    irls_iters : int
        Number of IRLS iterations.
    z_min : float
        Minimum positive depth (in camera coordinates) to consider a point valid.

    Returns
    -------
    K : np.ndarray
        3x3 intrinsics matrix:
            [[fx, 0,  cx],
             [0,  fy, cy],
             [0,  0,  1 ]]
    stats : dict
        Diagnostics:
            {
              'num_used': int,
              'rmse_u_px': float,
              'rmse_v_px': float,
              'fx': float, 'fy': float, 'cx': float, 'cy': float,
              'H': int, 'W': int, 'N': int
            }

    Notes
    -----
    - This ignores lens distortion and skew. If your data has distortion, consider
      pre-undistorting or extend the solver to estimate k1/k2 with non-linear least squares.
    - If your `points` are already in camera coordinates per frame, pass `camera_poses=None`.
    - For your Pi3 predictions dict, a typical call is:

        K, stats = estimate_camera_intrinsics_from_pointcloud(
            points=predictions['points'],                # (N,H,W,3)
            images=predictions['images'],                # (N,H,W,3) just for H,W
            camera_poses=predictions.get('camera_poses'),# (N,4,4) if points are world-space
            poses_are_cam_to_world=True,                 # Pi3 usually returns c2w
            conf=predictions.get('conf'),                # (N,H,W)
        )
    """
    # ---- helper: robust weighted least squares (Huber IRLS) ----
    def _irls(A: np.ndarray, y: np.ndarray, w0: np.ndarray) -> tuple[np.ndarray, float]:
        # A: (M,2), y: (M,), w0: (M,)
        w = w0.copy()
        theta = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
        for _ in range(max(0, irls_iters - 1)):
            r = y - A @ theta
            # Huber weights in pixels
            abs_r = np.abs(r)
            w = np.where(abs_r <= huber_delta_px, 1.0, huber_delta_px / (abs_r + 1e-12))
            # Re-apply base confidence as minimum
            w = np.maximum(w, w0)
            theta = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
        # final RMSE
        r = y - A @ theta
        rmse = np.sqrt(np.mean((r)**2))
        return theta, float(rmse)

    # ---- shapes and basic checks ----
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("`points` must have shape (N, H, W, 3).")
    if images.ndim < 4:
        raise ValueError("`images` must have shape (N, H, W, C).")

    N, H, W, _ = points.shape
    # Build pixel coordinate grid (u: x, v: y)
    u_grid, v_grid = np.meshgrid(
        np.arange(W, dtype=np.float64) + pixel_center,
        np.arange(H, dtype=np.float64) + pixel_center,
        indexing="xy"
    )  # each (H, W)

    # ---- get points in camera coordinates for each frame ----
    # If camera_poses is provided, assume `points` are world-space and transform to camera-space.
    if camera_poses is not None:
        if camera_poses.shape[-2:] == (3, 4):
            # upgrade to 4x4
            last_row = np.array([0, 0, 0, 1], dtype=camera_poses.dtype)
            last_row = np.broadcast_to(last_row, (*camera_poses.shape[:-2], 1, 4))
            poses_4x4 = np.concatenate([camera_poses, last_row], axis=-2)
        elif camera_poses.shape[-2:] == (4, 4):
            poses_4x4 = camera_poses
        else:
            raise ValueError("`camera_poses` must have shape (N, 4, 4) or (N, 3, 4).")

        if poses_are_cam_to_world:
            # invert to get w2c
            # vectorized inverse of rigid transform
            R = poses_4x4[:, :3, :3]
            t = poses_4x4[:, :3, 3:4]
            Rt = np.transpose(R, (0, 2, 1))
            t_inv = -Rt @ t
            w2c = np.concatenate([np.concatenate([Rt, t_inv], axis=-1),
                                  np.tile(np.array([0, 0, 0, 1])[None, None, :], (N, 1, 1))], axis=-2)
        else:
            w2c = poses_4x4  # already world->camera

        # Apply transform: Xc = R^T (Xw - t)   (which is what w2c does)
        pts_cam = np.empty_like(points, dtype=np.float64)
        # reshape for batched matmul
        Pw = np.concatenate([points, np.ones((*points.shape[:-1], 1), dtype=np.float64)], axis=-1)  # (N,H,W,4)
        Pw = Pw.reshape(N, -1, 4).transpose(0, 2, 1)  # (N,4,M)
        Xc = (w2c @ Pw)  # (N,4,M)
        Xc = Xc[:, :3, :].transpose(0, 2, 1).reshape(N, H, W, 3)  # (N,H,W,3)
    else:
        # Assume already camera-space
        Xc = points.astype(np.float64, copy=False)

    # ---- validity mask ----
    Z = Xc[..., 2]
    base_valid = np.isfinite(Xc).all(axis=-1) & np.isfinite(u_grid) & np.isfinite(v_grid)
    base_valid &= (Z > z_min)

    if valid_mask is not None:
        if valid_mask.shape != (N, H, W):
            raise ValueError("`valid_mask` must match (N,H,W).")
        base_valid &= valid_mask

    # Confidence weights
    if conf is not None:
        if conf.ndim == 4 and conf.shape[-1] == 1:
            conf = conf[..., 0]
        if conf.shape != (N, H, W):
            raise ValueError("`conf` must have shape (N,H,W) or (N,H,W,1).")
        w0 = np.clip(conf.astype(np.float64), 1e-6, 1.0)
    else:
        w0 = np.ones((N, H, W), dtype=np.float64)

    valid = base_valid & (w0 > 0)
    num_valid = int(valid.sum())
    if num_valid < 10:
        raise ValueError("Not enough valid correspondences to estimate intrinsics.")

    # ---- build linear systems ----
    # Flatten per-frame and stack across frames
    X = Xc[..., 0][valid]
    Y = Xc[..., 1][valid]
    Z = Xc[..., 2][valid]
    u = np.broadcast_to(u_grid, (N, H, W))[valid]
    v = np.broadcast_to(v_grid, (N, H, W))[valid]
    w = w0[valid]

    # Optional downsampling to keep things light
    if num_valid > max_points:
        idx = np.random.default_rng(0).choice(num_valid, size=max_points, replace=False)
        X, Y, Z, u, v, w = X[idx], Y[idx], Z[idx], u[idx], v[idx], w[idx]
        num_valid = max_points

    xnz = X / Z
    ynz = Y / Z

    # Two independent 2-parameter fits:
    #   u = [xnz, 1] @ [fx, cx]
    #   v = [ynz, 1] @ [fy, cy]
    Au = np.stack([xnz, np.ones_like(xnz)], axis=1)  # (M,2)
    Av = np.stack([ynz, np.ones_like(ynz)], axis=1)  # (M,2)

    # Base weights incorporate confidence; Huber IRLS provides robustness
    theta_u, rmse_u = _irls(Au, u, w0=w)
    theta_v, rmse_v = _irls(Av, v, w0=w)

    fx, cx = float(theta_u[0]), float(theta_u[1])
    fy, cy = float(theta_v[0]), float(theta_v[1])

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    stats = {
        "num_used": num_valid,
        "rmse_u_px": rmse_u,
        "rmse_v_px": rmse_v,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "H": H, "W": W, "N": N,
    }
    return K, stats


def save_nerfstudio_format(predictions, intrinsics, path):
    """
    Serialize a sequence of frames (poses + intrinsics) to a NeRF/Colmap-style JSON.

    This writes one shared pinhole intrinsics for all frames of a single camera and
    a per-frame SE(3) transform. It is compatible with common loaders that expect
    fields like `fl_x`, `fl_y`, `cx`, `cy`, `w`, `h`, and `camera_model="OPENCV"`.

    Parameters
    ----------
    predictions : dict
        Dictionary produced by your pipeline. Expected keys:
          - 'points': np.ndarray of shape (N, H, W, 3). Used to infer N, H, W if images absent.
          - 'camera_poses': np.ndarray of shape (N, 4, 4) or (N, 3, 4). Camera-to-world is typical.
          - 'images' (optional): np.ndarray of shape (N, H, W, C). If present, used for H and W.
        Any other keys in `predictions` are passed through unchanged (function returns the same dict).

    intrinsics : one of
        - np.ndarray shape (3, 3): camera matrix K.
        - tuple/list: (K, stats) as returned by `estimate_camera_intrinsics_from_pointcloud`,
          where K is (3,3) and stats may include 'H' and 'W' used for automatic scaling.
        - dict with keys 'fx', 'fy', 'cx', 'cy' (and optional 'H', 'W' for source resolution).
          Optional distortion keys 'k1', 'k2', 'p1', 'p2', 'k3' are supported; if omitted they default to 0.

        If `intrinsics` contains a source resolution (stats['H']/['W'] or keys 'H'/'W'),
        the focal lengths and principal point will be scaled to the current (H, W) of the frames.

    path : str
        Output file path (e.g., "transforms.json").

    Notes
    -----
    - The JSON contains a list of `frames`, each with:
        {
          "file_path": "camera/XYZ.jpg",
          "transform_matrix": 4x4 (list of lists),
          "w": W, "h": H,
          "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
          "camera_model": "OPENCV",
          "k1": 0, "k2": 0, "p1": 0, "p2": 0, "k3": 0   # included if available/assumed
        }
      File paths are templated as "camera/{i:03}.jpg". Adjust later if your filenames differ.
    """
    # ---- resolve N, H, W from predictions ----
    if 'images' in predictions and predictions['images'] is not None:
        N, H, W = predictions['images'].shape[:3]
    else:
        N, H, W, _ = predictions['points'].shape

    # ---- parse intrinsics into fx, fy, cx, cy and (optional) distortion ----
    def _parse_intrinsics(_intr):
        k1 = k2 = p1 = p2 = k3 = 0.0
        srcH = srcW = None

        if isinstance(_intr, (list, tuple)) and len(_intr) >= 1 and (
            isinstance(_intr[0], np.ndarray) and _intr[0].shape == (3, 3)
        ):
            K = _intr[0]
            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
            # source resolution from stats if present
            if len(_intr) > 1 and isinstance(_intr[1], dict):
                srcH = _intr[1].get('H')
                srcW = _intr[1].get('W')
        elif isinstance(_intr, np.ndarray) and _intr.shape == (3, 3):
            fx, fy, cx, cy = float(_intr[0, 0]), float(_intr[1, 1]), float(_intr[0, 2]), float(_intr[1, 2])
        elif isinstance(_intr, dict):
            if 'K' in _intr and isinstance(_intr['K'], np.ndarray) and _intr['K'].shape == (3, 3):
                K = _intr['K']
                fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
            else:
                fx = float(_intr['fx']); fy = float(_intr['fy']); cx = float(_intr['cx']); cy = float(_intr['cy'])
            # optional distortion
            k1 = float(_intr.get('k1', 0.0)); k2 = float(_intr.get('k2', 0.0))
            p1 = float(_intr.get('p1', 0.0)); p2 = float(_intr.get('p2', 0.0))
            k3 = float(_intr.get('k3', 0.0))
            # optional source resolution
            srcH = _intr.get('H'); srcW = _intr.get('W')
            if 'stats' in _intr and isinstance(_intr['stats'], dict):
                srcH = _intr['stats'].get('H', srcH)
                srcW = _intr['stats'].get('W', srcW)
        else:
            raise TypeError("Unsupported `intrinsics` type. Provide (K, stats), K, or a dict with fx/fy/cx/cy.")

        # scale intrinsics if they were estimated at a different resolution
        if srcH is not None and srcW is not None and (srcH != H or srcW != W):
            sx = W / float(srcW)
            sy = H / float(srcH)
            fx *= sx; cx *= sx
            fy *= sy; cy *= sy

        return {
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2, 'k3': k3
        }

    intr = _parse_intrinsics(intrinsics)

    # ---- normalize/upgrade poses to 4x4 lists ----
    poses = predictions.get('camera_poses')
    if poses is None:
        # fall back to identity transforms if not provided
        poses = np.tile(np.eye(4, dtype=float), (N, 1, 1))
    poses = np.asarray(poses)

    if poses.ndim != 3 or poses.shape[0] != N or poses.shape[1] not in (3, 4) or poses.shape[2] != 4:
        raise ValueError("`predictions['camera_poses']` must have shape (N, 4, 4) or (N, 3, 4).")

    if poses.shape[1] == 3:  # (N,3,4) -> (N,4,4)
        bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)[None, None, :]
        bottom = np.repeat(bottom, repeats=N, axis=0)
        poses_4x4 = np.concatenate([poses, bottom], axis=1)
    else:
        poses_4x4 = poses.astype(float, copy=False)

    # ---- build JSON ----
    output = {"frames": []}
    for i in range(N):
        frame = {
            "file_path": f"camera/{i:03}.jpg",
            "transform_matrix": (poses_4x4[i] @ np.diag([1.0, -1.0, -1.0, 1.0])).tolist(), # convert OpenCV to OpenGL
            "w": int(W),
            "h": int(H),
            "fl_x": float(intr['fx']),
            "fl_y": float(intr['fy']),
            "cx": float(intr['cx']),
            "cy": float(intr['cy']),
            "camera_model": "PINHOLE",
        }
        # include distortion terms if non-zero (or include all with zeros for compatibility)
        frame.update({
            "k1": float(intr.get('k1', 0.0)),
            "k2": float(intr.get('k2', 0.0)),
            "p1": float(intr.get('p1', 0.0)),
            "p2": float(intr.get('p2', 0.0)),
            "k3": float(intr.get('k3', 0.0)),
        })
        output["frames"].append(frame)

    # ---- write ----
    with open(path, 'w') as f:
        json.dump(output, f, indent=4)

    return


def save_images(
    predictions: dict,
    out_dir: str,
    poses_are_cam_to_world: bool = True,
    image_quality: int = 95,
) -> dict:
    """
    Save per-frame RGB images

  
    Images are written as JPEG to `<out_dir>/camera/{i:03}.jpg`.
   
    Parameters
    ----------
    predictions : dict
        Expected keys:
          - 'images': (N, H, W, C) array in [0,1] or [0,255] (dtype float/uint8).
          - 'points' : (N, H, W, 3) array. Camera-space or world-space (see below).
          - 'local_points' (optional): (N, H, W, 3) camera-space points.
          - 'camera_poses' (optional): (N, 4, 4) or (N, 3, 4) extrinsics.
            If `poses_are_cam_to_world=True`, poses are c2w and will be inverted.
            If False, they are world->camera and used directly.
    out_dir : str
        Root folder where `camera/` is created.
    poses_are_cam_to_world : bool
        See `camera_poses` above.
    image_quality : int
        JPEG quality for image saving (1..100).

    Returns
    -------
    dict
        The unmodified `predictions` (returned for convenience).
    
    """
    os.makedirs(out_dir, exist_ok=True)
    cam_dir = os.path.join(out_dir, "camera")
    os.makedirs(cam_dir, exist_ok=True)

    images = predictions["images"]
    N = images.shape[0]

    # ---- save per-frame ----
    for i in range(N):
        # Save image
        img = images[i]
        if img.shape[-1] > 3:
            img = img[..., :3]  # drop extra channels if any
        if img.dtype != np.uint8:
            # normalize into [0,255]
            finite = np.isfinite(img)
            if img.max() <= 1.01 and img.min() >= -0.01:
                img = np.clip(img, 0.0, 1.0) * 255.0
            else:
                img = np.clip(img, 0.0, 255.0)
            img = np.where(finite, img, 0)
            img = np.round(img).astype(np.uint8)
        Image.fromarray(img).save(
            os.path.join(cam_dir, f"{i:03}.jpg"),
            quality=image_quality,
            subsampling=0
        )
    
    return 


def save_local_points_pcd(
    predictions: dict,
    out_dir: str,
    per_frame: bool = True,
    poses_are_cam_to_world: bool = True,
    stride: int = 1,
    conf_thresh: float = 0.0,
    z_min: float = 1e-6,
    binary: bool = True,
) -> dict:
    """
    Save colored point clouds from `predictions` in PCD format.

    This function exports **camera-space** point clouds using `local_points`
    (preferred), or derives them by transforming `points` with `camera_poses`.
    Colors are sampled from `images` and stored in the PCD `rgb` field
    (packed float, PCL-compatible). You can save one PCD per frame or a single
    merged PCD.

    Parameters
    ----------
    predictions : dict
        Expected keys:
          - 'images' : (N,H,W,3) uint8 or float array in [0,1] or [0,255].
          - 'local_points' (preferred): (N,H,W,3) camera-space 3D.
          - 'points' : (N,H,W,3) world- or camera-space 3D (fallback).
          - 'camera_poses' (optional): (N,4,4) or (N,3,4) extrinsics.
                If `poses_are_cam_to_world=True`, poses are c2w and will be inverted
                when deriving camera-space points from world points.
          - 'conf' (optional): (N,H,W) per-pixel confidence in [0,1] used to filter.
    out_dir : str
        Directory to write PCD files into. Subfolder `pcd/` will be created.
    per_frame : bool
        If True, writes one PCD per frame: `pcd/{i:03}.pcd`.
        If False, writes a single merged file: `pcd/all_local.pcd`.
        The merged file is still in **camera coordinates per frame** (i.e., not a
        global/world-aligned cloud). If you need a world-space merged cloud, transform
        `local_points` with c2w first and then call this function with `per_frame=False`.
    poses_are_cam_to_world : bool
        How to interpret `camera_poses` if we need to derive local points from world points.
    stride : int
        Spatial subsampling step (e.g., 2 keeps every other pixel) to reduce file size.
    conf_thresh : float
        Drop points with confidence < conf_thresh if `conf` is provided.
    z_min : float
        Minimum positive depth (Z in camera coordinates) to keep a point.
    binary : bool
        If True, write `DATA binary` PCD (smaller, faster). If False, write ASCII.

    Notes
    -----
    - PCD layout: FIELDS x y z rgb  (x,y,z float32; rgb packed float32).
    - Invalid / non-finite / behind-camera points are omitted.
    - Colors are RGB from `images`; if `images` are float, they are scaled to [0..255].
    """
    os.makedirs(out_dir, exist_ok=True)
    pcd_dir = os.path.join(out_dir, "pcd")
    os.makedirs(pcd_dir, exist_ok=True)

    images = np.asarray(predictions["images"])
    N, H, W = images.shape[:3]

    # Resolve camera-space points (local)
    if "local_points" in predictions and predictions["local_points"] is not None:
        local = np.asarray(predictions["local_points"], dtype=np.float32)
        assert local.shape[:3] == (N, H, W), "`local_points` shape must match images."
    else:
        pts = np.asarray(predictions["points"], dtype=np.float64)
        assert pts.shape[:3] == (N, H, W), "`points` shape must match images."
        poses = predictions.get("camera_poses", None)
        if poses is None:
            # Assume `points` already in camera coords
            local = pts.astype(np.float32, copy=False)
        else:
            poses = np.asarray(poses)
            if poses.shape[-2:] == (3, 4):
                bottom = np.array([0, 0, 0, 1.0], dtype=poses.dtype)[None, None, :]
                bottom = np.repeat(bottom, repeats=N, axis=0)
                poses = np.concatenate([poses, bottom], axis=1)
            if poses.shape[-2:] != (4, 4):
                raise ValueError("camera_poses must be (N,4,4) or (N,3,4).")

            # world->camera
            if poses_are_cam_to_world:
                R = poses[:, :3, :3]  # c2w
                t = poses[:, :3, 3]
                Rt = np.transpose(R, (0, 2, 1))
                t_inv = -(Rt @ t[..., None])[..., 0]
                R_w2c, t_w2c = Rt, t_inv
            else:
                R_w2c = poses[:, :3, :3]
                t_w2c = poses[:, :3, 3]

            local = np.empty_like(pts, dtype=np.float32)
            for i in range(N):
                Xw = pts[i].reshape(-1, 3)
                Xc = (Xw @ R_w2c[i].T) + t_w2c[i][None, :]
                local[i] = Xc.reshape(H, W, 3)

    # Prepare masks
    valid_z = local[..., 2] > z_min
    finite = np.isfinite(local).all(axis=-1)
    valid = valid_z & finite
    if "conf" in predictions and predictions["conf"] is not None:
        conf = np.asarray(predictions["conf"])
        if conf.ndim == 4 and conf.shape[-1] == 1:
            conf = conf[..., 0]
        assert conf.shape[:3] == (N, H, W)
        valid &= (conf >= conf_thresh)

    # Normalize colors to uint8 RGB
    imgs_u8 = images
    if imgs_u8.dtype != np.uint8:
        # Try to auto-scale
        if imgs_u8.max() <= 1.01 and imgs_u8.min() >= -0.01:
            imgs_u8 = np.clip(imgs_u8, 0, 1) * 255.0
        imgs_u8 = np.clip(imgs_u8, 0, 255).round().astype(np.uint8)

    # Optional stride
    if stride < 1:
        stride = 1

    def _pack_rgb_float(r, g, b) -> float:
        """Pack 3 uint8 into a single float32 'rgb' as PCL expects."""
        u = (int(r) << 16) | (int(g) << 8) | int(b)
        return struct.unpack('<f', struct.pack('<I', u))[0]

    def _write_pcd_xyzrgb(path, xyz: np.ndarray, rgbf32: np.ndarray, binary_flag: bool):
        """Write a PCD file with fields x y z rgb."""
        n_pts = xyz.shape[0]
        header = (
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z rgb\n"
            "SIZE 4 4 4 4\n"
            "TYPE F F F F\n"
            "COUNT 1 1 1 1\n"
            f"WIDTH {n_pts}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {n_pts}\n"
            f"DATA {'binary' if binary_flag else 'ascii'}\n"
        )
        with open(path, 'wb' if binary_flag else 'w') as f:
            if binary_flag:
                f.write(header.encode('ascii'))
                # interleave x y z rgb (all float32)
                interleaved = np.empty((n_pts, 4), dtype=np.float32)
                interleaved[:, :3] = xyz.astype(np.float32, copy=False)
                interleaved[:, 3] = rgbf32.astype(np.float32, copy=False)
                f.write(interleaved.tobytes(order='C'))
            else:
                f.write(header)
                for p, c in zip(xyz, rgbf32):
                    f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c:.8f}\n")

    if per_frame:
        for i in range(N):
            m = valid[i]
            if stride > 1:
                m_strided = np.zeros_like(m)
                m_strided[::stride, ::stride] = True
                m &= m_strided
            if not np.any(m):
                continue
            pts = local[i][m].reshape(-1, 3)
            cols = imgs_u8[i][m].reshape(-1, 3)  # RGB
            rgbf = np.array([_pack_rgb_float(r, g, b) for r, g, b in cols], dtype=np.float32)
            out_path = os.path.join(pcd_dir, f"{i:03}.pcd")
            _write_pcd_xyzrgb(out_path, pts, rgbf, binary_flag=binary)
    else:
        xyz_list = []
        rgb_list = []
        for i in range(N):
            m = valid[i]
            if stride > 1:
                m_strided = np.zeros_like(m)
                m_strided[::stride, ::stride] = True
                m &= m_strided
            if not np.any(m):
                continue
            pts = local[i][m].reshape(-1, 3)
            cols = imgs_u8[i][m].reshape(-1, 3)
            rgbf = np.array([_pack_rgb_float(r, g, b) for r, g, b in cols], dtype=np.float32)
            xyz_list.append(pts)
            rgb_list.append(rgbf)
        if xyz_list:
            xyz = np.concatenate(xyz_list, axis=0)
            rgbf = np.concatenate(rgb_list, axis=0)
        else:
            xyz = np.empty((0, 3), dtype=np.float32)
            rgbf = np.empty((0,), dtype=np.float32)
        out_path = os.path.join(pcd_dir, "all_local.pcd")
        _write_pcd_xyzrgb(out_path, xyz, rgbf, binary_flag=binary)

    return


if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the input image directory or a video file.")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save the output .ply file.")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
    parser.add_argument("--save_points", action='store_true',
                         help="Save colorized local_points in pcd format")
                        
    args = parser.parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1
    print(f'Sampling interval: {args.interval}')

    # 1. Prepare model
    print(f"Loading model...")
    device = torch.device(args.device)
    if args.ckpt is not None:
        model = Pi3().to(device).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        
        model.load_state_dict(weight)
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()

    # 2. Prepare input data
    imgs = load_images_as_tensor(args.data_path, interval=args.interval).to(device) # (N, 3, H, W)
    # 3. create output directory
    os.makedirs(args.save_path, exist_ok=True)

    # 4. Infer
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            predictions = model(imgs[None]) # Add batch dimensionx


    predictions['images'] = imgs[None].permute(0, 1, 3, 4, 2)
    predictions['conf'] = torch.sigmoid(predictions['conf'])
    edge = depth_edge(predictions['local_points'][..., 2], rtol=0.03)
    predictions['conf'][edge] = 0.0
    # del predictions['local_points']
    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    print(predictions['camera_poses'].shape)
    print(predictions['points'].shape)
    print(predictions['conf'].shape)


    # 5. estimate intrinsic parameters
    _, H, W, _ = predictions['images'].shape

    # Flatten 3D points and image coordinates
    points_3d = predictions['points'][0].reshape(-1, 3)
    pixels_2d = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1).reshape(-1, 2)


    # Estimate intrinsics
    K, stats = estimate_camera_intrinsics_from_pointcloud(
        points=predictions['points'],                 # (N,H,W,3)
        images=predictions['images'],                 # (N,H,W,3)
        camera_poses=predictions.get('camera_poses'), # (N,4,4) if points are world-space
        poses_are_cam_to_world=True,                  # Pi3 typically returns c2w poses
        conf=predictions.get('conf'),                 # (N,H,W)
    )
    print("K =\n", K)
    print("diagnostics:", stats)

    # save predicitons as transfo
    # rms.json (nerfstudio format)
    save_nerfstudio_format(predictions, K, os.path.join(args.save_path, "transforms.json"))

    
    save_images(predictions, 
                out_dir=args.save_path, 
                poses_are_cam_to_world=True)

    if args.save_points:
        save_local_points_pcd(
            predictions,
            out_dir=args.save_path,
            per_frame=True,
            poses_are_cam_to_world=True,
            stride=2,
            conf_thresh=3.0,
            z_min=1e-6,
            binary=True,
        )

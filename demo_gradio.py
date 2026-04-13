import gc
import glob
import os
import shutil
import time
from datetime import datetime

import cv2
import gradio as gr
import matplotlib
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor

# import spaces         # only for web demo
from pi3.utils.geometry import depth_edge

"""
Gradio utils
"""

def predictions_to_glb(
    predictions,
    conf_thres=50.0,
    filter_by_frames="all",
    show_cam=True,
) -> trimesh.Scene:
    """
    Converts VGGT predictions to a 3D scene represented as a GLB file.

    Args:
        predictions (dict): Dictionary containing model predictions with keys:
            - world_points: 3D point coordinates (S, H, W, 3)
            - world_points_conf: Confidence scores (S, H, W)
            - images: Input images (S, H, W, 3)
            - extrinsic: Camera extrinsic matrices (S, 3, 4)
        conf_thres (float): Percentage of low-confidence points to filter out (default: 50.0)
        filter_by_frames (str): Frame filter specification (default: "all")
        show_cam (bool): Include camera visualization (default: True)

    Returns:
        trimesh.Scene: Processed 3D scene containing point cloud and cameras

    Raises:
        ValueError: If input predictions structure is invalid
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    if conf_thres is None:
        conf_thres = 10

    print("Building GLB scene")
    selected_frame_idx = None
    if filter_by_frames != "all" and filter_by_frames != "All":
        try:
            # Extract the index part before the colon
            selected_frame_idx = int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            pass

    pred_world_points = predictions["points"]
    pred_world_points_conf = predictions.get("conf", np.ones_like(pred_world_points[..., 0]))

    # Get images from predictions
    images = predictions["images"]
    # Use extrinsic matrices instead of pred_extrinsic_list
    camera_poses = predictions["camera_poses"]

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        camera_poses = camera_poses[selected_frame_idx][None]

    vertices_3d = pred_world_points.reshape(-1, 3)
    # Handle different image formats - check if images need transposing
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    conf = pred_world_points_conf.reshape(-1)
    # Convert percentage threshold to actual confidence value
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        # conf_threshold = np.percentile(conf, conf_thres)
        conf_threshold = conf_thres / 100

    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        # Calculate the 5th and 95th percentiles along each axis
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)

        # Calculate the diagonal length of the percentile bounding box
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

    # Initialize a 3D scene
    scene_3d = trimesh.Scene()

    # Add point cloud data to the scene
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)

    scene_3d.add_geometry(point_cloud_data)

    # Prepare 4x4 matrices for camera extrinsics
    num_cameras = len(camera_poses)

    if show_cam:
        # Add camera models to the scene
        for i in range(num_cameras):
            camera_to_world = camera_poses[i]
            rgba_color = colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])

            # integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale)
            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, 1.)          # fixed camera size

    # Rotate scene for better visualize
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 100, degrees=True).as_matrix()            # plane rotate
    align_rotation[:3, :3] = align_rotation[:3, :3] @ Rotation.from_euler("x", 155, degrees=True).as_matrix()           # roll
    scene_3d.apply_transform(align_rotation)

    print("GLB Scene built")
    return scene_3d

def integrate_camera_into_scene(scene: trimesh.Scene, transform: np.ndarray, face_colors: tuple, scene_scale: float):
    """
    Integrates a fake camera mesh into the 3D scene.

    Args:
        scene (trimesh.Scene): The 3D scene to add the camera model.
        transform (np.ndarray): Transformation matrix for camera positioning.
        face_colors (tuple): Color of the camera face.
        scene_scale (float): Scale of the scene.
    """

    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    # Create cone shape for camera
    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    opengl_transform = get_opengl_conversion_matrix()
    # Combine transformations
    complete_transform = transform @ opengl_transform @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    # Generate mesh for the camera
    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    vertices_combined = np.concatenate(
        [
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            transform_points(slight_rotation, camera_cone_shape.vertices),
        ]
    )
    vertices_transformed = transform_points(complete_transform, vertices_combined)

    mesh_faces = compute_camera_faces(camera_cone_shape)

    # Add the camera mesh to the scene
    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def get_opengl_conversion_matrix() -> np.ndarray:
    """
    Constructs and returns the OpenGL conversion matrix.

    Returns:
        numpy.ndarray: A 4x4 OpenGL conversion matrix.
    """
    # Create an identity matrix
    matrix = np.identity(4)

    # Flip the y and z axes
    matrix[1, 1] = -1
    matrix[2, 2] = -1

    return matrix


def transform_points(transformation: np.ndarray, points: np.ndarray, dim: int = None) -> np.ndarray:
    """
    Applies a 4x4 transformation to a set of points.

    Args:
        transformation (np.ndarray): Transformation matrix.
        points (np.ndarray): Points to be transformed.
        dim (int, optional): Dimension for reshaping the result.

    Returns:
        np.ndarray: Transformed points.
    """
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]

    # Apply transformation
    transformation = transformation.swapaxes(-1, -2)  # Transpose the transformation matrix
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]

    # Reshape the result
    result = points[..., :dim].reshape(*initial_shape, dim)
    return result


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    """
    Computes the faces for the camera mesh.

    Args:
        cone_shape (trimesh.Trimesh): The shape of the camera cone.

    Returns:
        np.ndarray: Array of faces for the camera mesh.
    """
    # Create pseudo cameras
    faces_list = []
    num_vertices_cone = len(cone_shape.vertices)

    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices_cone
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone

        faces_list.extend(
            [
                (v1, v2, v2_offset),
                (v1, v1_offset, v3),
                (v3_offset, v2, v3),
                (v1, v2, v2_offset_2),
                (v1, v1_offset_2, v3),
                (v3_offset_2, v2, v3),
            ]
        )

    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)


# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
# @spaces.GPU(duration=120)
def run_model(target_dir, model) -> dict:
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    # interval = 10 if target_dir.endswith('.mp4') else 1
    interval = 1
    imgs = load_images_as_tensor(os.path.join(target_dir, "images"), interval=interval).to(device) # (N, 3, H, W)

    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            predictions = model(imgs[None]) # Add batch dimension
    predictions['images'] = imgs[None].permute(0, 1, 3, 4, 2)
    predictions['conf'] = torch.sigmoid(predictions['conf'])
    edge = depth_edge(predictions['local_points'][..., 2], rtol=0.03)
    predictions['conf'][edge] = 0.0
    del predictions['local_points']

    # # transform to first camera coordinate
    # predictions['points'] = torch.einsum('bij, bnhwj -> bnhwi', se3_inverse(predictions['camera_poses'][:, 0]), homogenize_points(predictions['points']))[..., :3]
    # predictions['camera_poses'] = torch.einsum('bij, bnjk -> bnik', se3_inverse(predictions['camera_poses'][:, 0]), predictions['camera_poses'])

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension

    # Clean up
    torch.cuda.empty_cache()
    return predictions


# -------------------------------------------------------------------------
# 2) Handle uploaded video/images --> produce target_dir + images
# -------------------------------------------------------------------------
def handle_uploads(input_video, input_images, interval=-1):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images or extracted frames from video into it. Return (target_dir, image_paths).
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(target_dir_images, exist_ok=True)

    image_paths = []

    # --- Handle images ---
    if input_images is not None:
        if interval is not None and interval > 0:
            input_images = input_images[::interval]

        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)
        
    # --- Handle video ---
    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        if interval is not None and interval > 0:
            frame_interval = interval
        else:
            frame_interval = int(fps * 1)  # 1 frame/sec

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


# -------------------------------------------------------------------------
# 3) Update gallery on upload
# -------------------------------------------------------------------------
def update_gallery_on_upload(input_video, input_images, interval=-1):
    """
    Whenever user uploads or changes files, immediately handle them
    and show in the gallery. Return (target_dir, image_paths).
    If nothing is uploaded, returns "None" and empty list.
    """
    if not input_video and not input_images:
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images, interval=interval)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."


# -------------------------------------------------------------------------
# 4) Reconstruction: uses the target_dir plus any viz parameters
# -------------------------------------------------------------------------
# commit below for local demo
# @spaces.GPU(duration=120)
def gradio_demo(
    target_dir,
    conf_thres=3.0,
    frame_filter="All",
    show_cam=True,
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None, None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running run_model...")
    with torch.no_grad():
        predictions = run_model(target_dir, model)

    # Save predictions
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    # Handle None frame_filter
    if frame_filter is None:
        frame_filter = "All"

    # Build a GLB file name
    glbfile = build_glb_path(target_dir, conf_thres, frame_filter, show_cam)

    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        show_cam=show_cam,
    )
    glbscene.export(file_obj=glbfile)

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."

    return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True)


# -------------------------------------------------------------------------
# 5) Helper functions for UI resets + re-visualization
# -------------------------------------------------------------------------
def clear_fields():
    """
    Clears the 3D viewer, the stored target_dir, and empties the gallery.
    """
    return None


def update_log():
    """
    Display a quick log message while waiting.
    """
    return "Loading and Reconstructing..."


def build_glb_path(target_dir, conf_thres, frame_filter, show_cam):
    """
    Build a stable GLB output path for current visualization settings.
    """
    return os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_cam{show_cam}.glb",
    )


def update_visualization(
    target_dir, conf_thres, frame_filter, show_cam, is_example
):
    """
    Reload saved predictions from npz, create (or reuse) the GLB for new parameters,
    and return it for the 3D viewer. If is_example == "True", skip.
    """

    # If it's an example click, skip as requested
    no_reconstruction_msg = "No reconstruction available. Please click the Reconstruct button first."

    if is_example == "True":
        return None, no_reconstruction_msg

    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, no_reconstruction_msg

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first."

    key_list = [
        "images",
        "points",
        "conf",
        "camera_poses",
    ]

    loaded = np.load(predictions_path)
    predictions = {key: np.array(loaded[key]) for key in key_list}

    glbfile = build_glb_path(target_dir, conf_thres, frame_filter, show_cam)

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            show_cam=show_cam,
        )
        glbscene.export(file_obj=glbfile)

    return glbfile, "Updating Visualization"


# -------------------------------------------------------------------------
# Example images
# -------------------------------------------------------------------------

house = "examples/gradio_examples/house.mp4"
man_walking_long = "examples/gradio_examples/man_walking_long.mp4"
parkour = "examples/gradio_examples/parkour.mp4"
valley = "examples/gradio_examples/valley.mp4"
cartoon_horse = "examples/cartoon_horse.mp4"
parkour_long = "examples/parkour_long.mp4"
skating = "examples/skating.mp4"
skiing = "examples/skiing.mp4"

# -------------------------------------------------------------------------
# 6) Build Gradio UI
# -------------------------------------------------------------------------

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Initializing and loading Pi3 model...")

    model = Pi3.from_pretrained("yyfz233/Pi3")
    # model = Pi3()
    # model.load_state_dict(torcdtype = torch.bfloat16h.load('ckpts/pi3.pt', weights_only=False, map_location=device))

    model.eval()
    model = model.to(device)

    theme = gr.themes.Ocean()
    theme.set(
        checkbox_label_background_fill_selected="*button_primary_background_fill",
        checkbox_label_text_color_selected="*button_primary_text_color",
    )

    with gr.Blocks(
        theme=theme,
        css="""
        /* --- Google 字体导入 (科技感字体) --- */
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;500;700&display=swap');

        /* --- 动画关键帧 --- */
        /* 背景动态星云效果 */
        @keyframes gradient-animation {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* 标题和状态文字的霓虹灯光效 */
        @keyframes text-glow {
            0%, 100% {
                text-shadow: 0 0 10px #0ea5e9, 0 0 20px #0ea5e9, 0 0 30px #4f46e5, 0 0 40px #4f46e5;
            }
            50% {
                text-shadow: 0 0 5px #0ea5e9, 0 0 10px #0ea5e9, 0 0 15px #4f46e5, 0 0 20px #4f46e5;
            }
        }

        /* 卡片边框呼吸光晕 */
        @keyframes border-glow {
            0% { border-color: rgba(79, 70, 229, 0.5); box-shadow: 0 0 15px rgba(79, 70, 229, 0.3); }
            50% { border-color: rgba(14, 165, 233, 0.8); box-shadow: 0 0 25px rgba(14, 165, 233, 0.5); }
            100% { border-color: rgba(79, 70, 229, 0.5); box-shadow: 0 0 15px rgba(79, 70, 229, 0.3); }
        }

        /* --- 全局样式：宇宙黑暗主题 --- */
        .gradio-container {
            font-family: 'Rajdhani', sans-serif;
            background: linear-gradient(-45deg, #020617, #111827, #082f49, #4f46e5);
            background-size: 400% 400%;
            animation: gradient-animation 20s ease infinite;
            color: #9ca3af;
        }

        /* --- 全局文字颜色修复 (解决Light Mode问题) --- */
        
        /* 1. 修复全局、标签和输入框内的文字颜色 */
        .gradio-container, .gr-label label, .gr-input, input, textarea, .gr-check-radio label {
            color: #d1d5db !important; /* 设置一个柔和的浅灰色 */
        }

        /* 2. 修复 Examples 表头 (这是您问题的核心) */
        thead th {
            color: white !important;
            background-color: #1f2937 !important; /* 同时给表头一个背景色，视觉效果更好 */
        }

        /* 3. 修复 Examples 表格内容文字 */
        tbody td {
            color: #d1d5db !important;
        }
        
        /* --- 状态信息 & 输出标题样式 (custom-log) ✨ --- */
        .custom-log * {
            font-family: 'Orbitron', sans-serif;
            font-size: 24px !important;
            font-weight: 700 !important;
            text-align: center !important;
            color: transparent !important;
            background-image: linear-gradient(120deg, #93c5fd, #6ee7b7, #fde047);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            background-clip: text;
            animation: gradient-animation 8s ease-in-out infinite, text-glow 3s ease-in-out infinite;
            padding: 10px 0;
        }
        
        /* --- UI 卡片/分组样式 (玻璃拟态) 💎 --- */
        .gr-block.gr-group {
            background-color: rgba(17, 24, 39, 0.6);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(55, 65, 81, 0.5);
            border-radius: 16px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: all 0.3s ease;
            /* 应用边框呼吸光晕动画 */
            animation: border-glow 5s infinite alternate;
        }
        .gr-block.gr-group:hover {
            box-shadow: 0 0 25px rgba(14, 165, 233, 0.4);
            border-color: rgba(14, 165, 233, 0.6);
        }
        
        /* --- 酷炫按钮样式 🚀 --- */
        .gr-button {
            background: linear-gradient(to right, #4f46e5, #7c3aed, #0ea5e9) !important;
            background-size: 200% auto !important;
            color: white !important;
            font-weight: bold !important;
            border: none !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 15px 0 rgba(79, 70, 229, 0.5) !important;
            transition: all 0.4s ease-in-out !important;
            font-family: 'Orbitron', sans-serif !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .gr-button:hover {
            background-position: right center !important;
            box-shadow: 0 4px 20px 0 rgba(14, 165, 233, 0.6) !important;
            transform: translateY(-3px) scale(1.02);
        }
        .gr-button.primary {
            /* 主按钮增加呼吸光晕动画 */
            animation: border-glow 3s infinite alternate;
        }
        """,
    ) as demo:
        # Instead of gr.State, we use a hidden Textbox:
        is_example = gr.Textbox(label="is_example", visible=False, value="None")
        num_images = gr.Textbox(label="num_images", visible=False, value="None")
        target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

        gr.HTML(
        """
        <style>
                /* --- 介绍文字区专属样式 --- */
                .intro-content { font-size: 17px !important; line-height: 1.7; color: #C0C0C0 !important; }
                /* 额外为 p 标签添加规则，确保覆盖 */
                .intro-content p { color: #C0C0C0 !important; }
                
                .intro-content h1 {
                    font-family: 'Orbitron', sans-serif; font-size: 2.8em !important; font-weight: 900;
                    text-align: center; color: #C0C0C0 !important; animation: text-glow 4s ease-in-out infinite; margin-bottom: 0px;
                }
                .intro-content .pi-symbol {
                    display: inline-block; color: transparent;
                    background-image: linear-gradient(120deg, #38bdf8, #818cf8, #c084fc);
                    -webkit-background-clip: text; background-clip: text;
                    text-shadow: 0 0 15px rgba(129, 140, 248, 0.5);
                }
                .intro-content .subtitle { text-align: center; font-size: 1.1em; margin-bottom: 2rem; }
                .intro-content a.themed-link {
                    color: #C0C0C0 !important; text-decoration: none; font-weight: 700; transition: all 0.3s ease;
                }
                .intro-content a.themed-link:hover { color: #EAEAEA !important; text-shadow: 0 0 8px rgba(234, 234, 234, 0.7); }
                .intro-content h3 {
                    font-family: 'Orbitron', sans-serif; color: #C0C0C0 !important; text-transform: uppercase;
                    letter-spacing: 2px; border-bottom: 1px solid #374151; padding-bottom: 8px; margin-top: 25px;
                }
                .intro-content ol { list-style: none; padding-left: 0; counter-reset: step-counter; }
                .intro-content ol li {
                    counter-increment: step-counter; margin-bottom: 15px; padding-left: 45px; position: relative;
                    color: #C0C0C0 !important; /* 确保列表项文字也是银白色 */
                }
                /* 自定义酷炫列表数字 */
                .intro-content ol li::before {
                    content: counter(step-counter); position: absolute; left: 0; top: 0;
                    width: 30px; height: 30px; background: linear-gradient(135deg, #1e3a8a, #4f46e5);
                    border-radius: 50%; color: white; font-weight: 700; font-family: 'Orbitron', sans-serif;
                    display: flex; align-items: center; justify-content: center;
                    box-shadow: 0 0 10px rgba(79, 70, 229, 0.5);
                }
                .intro-content strong { color: #C0C0C0 !important; font-weight: 700; }
                .intro-content .performance-note {
                    background-color: rgba(14, 165, 233, 0.1); border-left: 4px solid #0ea5e9;
                    padding: 15px; border-radius: 8px; margin-top: 20px;
                }
                /* 确保提示框内的文字也生效 */
                .intro-content .performance-note p { color: #C0C0C0 !important; }

        </style>
                
        <div class="intro-content">
            <h1>🌌 <span class="pi-symbol">&pi;³</span>: Scalable Permutation-Equivariant Visual Geometry Learning</h1>
            <p class="subtitle">
                <a class="themed-link" href="">🐙 GitHub Repository</a> |
                <a class="themed-link" href="#">🚀 Project Page</a>
            </p>
            
            <p>Transform your videos or image collections into detailed 3D models. The <strong class="pi-symbol">&pi;³</strong> model processes your visual data to generate a rich 3D point cloud and calculate the corresponding camera perspectives.</p>
            
            <h3>How to Use:</h3>
            <ol>
                <li><strong>Provide Your Media:</strong> Upload a video or image set. You can specify a sampling interval below. By default, videos are sampled at 1 frame per second, and for image sets, every image is used (interval of 1). Your inputs will be displayed in the "Preview" gallery.</li>
                <li><strong>Generate the 3D Model:</strong> Press the "Reconstruct" button to initiate the process.</li>
                <li><strong>Explore and Refine Your Model:</strong> The generated 3D model will appear in the viewer on the right. Interact with it by rotating, panning, and zooming. You can also download the model as a GLB file. For further refinement, use the options below the viewer to adjust point confidence, filter by frame, or toggle camera visibility.</li>
            </ol>
            
            <div class="performance-note">
                <p><strong>A Quick Note on Performance:</strong> The core processing by <strong class="pi-symbol">&pi;³</strong> is incredibly fast, typically finishing in under a second. However, rendering the final 3D point cloud can take longer, depending on the complexity of the scene and the capabilities of the rendering engine.</p>
            </div>
        </div>
        """
    )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 1. Upload Media")
                    input_video = gr.Video(label="Upload Video", interactive=True)
                    input_images = gr.File(file_count="multiple", label="Or Upload Images", interactive=True)
                    interval = gr.Number(None, label='Frame/Image Interval', info="Sampling interval. Video default: 1 FPS. Image default: 1 (all images).")
                
                image_gallery = gr.Gallery(
                    label="Image Preview",
                    columns=4,
                    height="300px",
                    show_download_button=True,
                    object_fit="contain",
                    preview=True,
                )

            with gr.Column(scale=2):
                gr.Markdown("### 2. View Reconstruction")
                log_output = gr.Markdown("Please upload media and click Reconstruct.", elem_classes=["custom-log"])
                reconstruction_output = gr.Model3D(height=480, zoom_speed=0.5, pan_speed=0.5, label="3D Output")
                
                with gr.Row():
                    submit_btn = gr.Button("Reconstruct", scale=3, variant="primary")
                    clear_btn = gr.ClearButton(
                        scale=1
                    )
                
                with gr.Group():
                    gr.Markdown("### 3. Adjust Visualization")
                    with gr.Row():
                        conf_thres = gr.Slider(minimum=0, maximum=100, value=20, step=0.1, label="Confidence Threshold (%)")
                        show_cam = gr.Checkbox(label="Show Cameras", value=True)
                    frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")

        # Set clear button targets
        clear_btn.add([input_video, input_images, reconstruction_output, log_output, target_dir_output, image_gallery, interval])

        # ---------------------- Examples section ----------------------
        examples = [
            [skating, None, 10, 20, True],
            [parkour_long, None, 20, 10, True],
            [cartoon_horse, None, 10, 20, True],
            [skiing, None, 30, 70, True],
            [man_walking_long, None, 1, 50, True],
            [house, None, 1, 20, True],
            [parkour, None, 1, 20, True],
            [valley, None, 1, 20, True],
        ]

        def example_pipeline(
            input_video,
            input_images,
            interval,
            conf_thres,
            show_cam,
        ):
            """
            1) Copy example images to new target_dir
            2) Reconstruct
            3) Return model3D + logs + new_dir + updated dropdown + gallery
            We do NOT return is_example. It's just an input.
            """
            target_dir, image_paths = handle_uploads(input_video, input_images, interval)
            # Always use "All" for frame_filter in examples
            frame_filter = "All"
            glbfile, log_msg, dropdown = gradio_demo(
                target_dir, conf_thres, frame_filter, show_cam
            )
            return glbfile, log_msg, target_dir, dropdown, image_paths

        gr.Markdown("Click any row to load an example.", elem_classes=["example-log"])

        gr.Examples(
            examples=examples,
            inputs=[
                input_video,
                input_images,
                interval,
                conf_thres,
                show_cam,
            ],
            outputs=[reconstruction_output, log_output, target_dir_output, frame_filter, image_gallery],
            fn=example_pipeline,
            cache_examples=False,
            examples_per_page=50,
            run_on_click=False,
        )

        # -------------------------------------------------------------------------
        # "Reconstruct" button logic:
        #  - Clear fields
        #  - Update log
        #  - gradio_demo(...) with the existing target_dir
        #  - Then set is_example = "False"
        # -------------------------------------------------------------------------
        submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
            fn=update_log, inputs=[], outputs=[log_output]
        ).then(
            fn=gradio_demo,
            inputs=[
                target_dir_output,
                conf_thres,
                frame_filter,
                show_cam,
            ],
            outputs=[reconstruction_output, log_output, frame_filter],
        ).then(
            fn=lambda: "False", inputs=[], outputs=[is_example]  # set is_example to "False"
        )

        # -------------------------------------------------------------------------
        # Real-time Visualization Updates
        # -------------------------------------------------------------------------
        conf_thres.change(
            update_visualization,
            [
                target_dir_output,
                conf_thres,
                frame_filter,
                show_cam,
                is_example,
            ],
            [reconstruction_output, log_output],
        )
        frame_filter.change(
            update_visualization,
            [
                target_dir_output,
                conf_thres,
                frame_filter,
                show_cam,
                is_example,
            ],
            [reconstruction_output, log_output],
        )
    
        show_cam.change(
            update_visualization,
            [
                target_dir_output,
                conf_thres,
                frame_filter,
                show_cam,
                is_example,
            ],
            [reconstruction_output, log_output],
        )

        # -------------------------------------------------------------------------
        # Auto-update gallery whenever user uploads or changes their files
        # -------------------------------------------------------------------------
        input_video.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images, interval],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
        )
        input_images.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images, interval],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
        )

    demo.queue(max_size=20).launch(show_error=True, share=True)

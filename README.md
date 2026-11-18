<h1 align="center">🌌 <em>&pi;³</em>: Permutation-Equivariant Visual Geometry Learning</h1>

<div align="center">
    <p>
        <a href="https://github.com/yyfz">Yifan Wang</a><sup>1*</sup>&nbsp;&nbsp;
        <a href="https://zhoutimemachine.github.io">Jianjun Zhou</a><sup>123*</sup>&nbsp;&nbsp;
        <a href="https://www.haoyizhu.site">Haoyi Zhu</a><sup>1</sup>&nbsp;&nbsp;
        <a href="https://github.com/AmberHeart">Wenzheng Chang</a><sup>1</sup>&nbsp;&nbsp;
        <a href="https://github.com/yangzhou24">Yang Zhou</a><sup>1</sup>
        <br>
        <a href="https://github.com/LiZizun">Zizun Li</a><sup>1</sup>&nbsp;&nbsp;
        <a href="https://github.com/SOTAMak1r">Junyi Chen</a><sup>1</sup>&nbsp;&nbsp;
        <a href="https://oceanpang.github.io">Jiangmiao Pang</a><sup>1</sup>&nbsp;&nbsp;
        <a href="https://cshen.github.io">Chunhua Shen</a><sup>2</sup>&nbsp;&nbsp;
        <a href="https://tonghe90.github.io">Tong He</a><sup>13†</sup>
    </p>
    <p>
        <sup>1</sup>Shanghai AI Lab &nbsp;&nbsp;&nbsp;
        <sup>2</sup>ZJU &nbsp;&nbsp;&nbsp;
        <sup>3</sup>SII
    </p>
    <p>
        <sup>*</sup> Equal Contribution &nbsp;&nbsp;&nbsp;
        <sup>†</sup> Corresponding Author
    </p>
</div>

<p align="center">
    <a href="https://arxiv.org/abs/2507.13347" target="_blank">
    <img src="https://img.shields.io/badge/Paper-00AEEF?style=plastic&logo=arxiv&logoColor=white" alt="Paper">
    </a>
    <a href="https://yyfz.github.io/pi3/" target="_blank">
    <img src="https://img.shields.io/badge/Project Page-F78100?style=plastic&logo=google-chrome&logoColor=white" alt="Project Page">
    </a>
    <a href="https://huggingface.co/spaces/yyfz233/Pi3" target="_blank">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue" alt="Hugging Face Demo">
    </a>
</p>

<div align="center">
    <a href="[PROJECT_PAGE_LINK_HERE]">
        <img src="assets/main.png" width="90%">
    </a>
    <p>
        <i>&pi;³ reconstructs visual geometry without a fixed reference view, achieving robust, state-of-the-art performance.</i>
    </p>
</div>

> [!TIP]                                                                                                                                                                                                           
   🔥🔥🔥 **Check out our new [Pi4](https://github.com/yyfz/Pi4): better geometry, motion estimation, and stronger generalization.** 

## 📣 Updates
* **[December 28, 2025]** 🚀 **Pi3X Released!** We have upgraded the model to **Pi3X**. This improved version eliminates grid artifacts (smoother point clouds), offers more precise confidence scoring, supports **conditional injection** (camera pose, intrinsics, depth), and enables **approximate metric scale** reconstruction.
* **[September 3, 2025]** ⭐️ Training code is updated! See [`training`](https://github.com/yyfz/Pi3/tree/training) branch for details.
* **[July 29, 2025]** 📈 Evaluation code is released! See [`evaluation`](https://github.com/yyfz/Pi3/tree/evaluation) branch for details.
* **[July 16, 2025]** 🚀 Hugging Face Demo and inference code are released!


## ✨ Overview
We introduce $\pi^3$, a novel feed-forward neural network that revolutionizes visual geometry reconstruction by **eliminating the need for a fixed reference view**. Traditional methods, which rely on a designated reference frame, are often prone to instability and failure if the reference is suboptimal.

In contrast, $\pi^3$ employs a fully **permutation-equivariant** architecture. This allows it to directly predict affine-invariant camera poses and scale-invariant local point maps from an unordered set of images, breaking free from the constraints of a reference frame. This design makes our model inherently **robust to input orderi

A key emergent property of our simple, bias-free design is the learning of a dense and structured latent representation of the camera pose manifold. Without complex priors or training schemes, $\pi^3$ achieves **state-of-the-art performance** 🏆 on a wide range of tasks, including camera pose estimation, monocular/video depth estimation, and dense point map estimation.

### Introducing Pi3X (Engineering Update)
Building upon the original framework, we present **Pi3X**, an enhanced version focused on flexibility and reconstruction quality:
* **Smoother Reconstruction:** We replaced the original output head with a **Convolutional Head**, significantly reducing grid-like artifacts and producing much smoother point clouds.
* **Flexible Conditioning:** Pi3X supports the optional injection of **camera poses, intrinsics, and depth**. This allows for more controlled reconstruction when partial priors are available.
* **Reliable Confidence:** We improved how confidence is learned. Instead of approximating a binary mask, the model now predicts continuous quality levels, making the confidence scores significantly more reliable for filtering noise.
* **Metric Scale:** The model now supports **metric scale reconstruction** (approximate), moving beyond purely scale-invariant predictions.

Overall, Pi3X offers slightly better reconstruction quality than the original $\pi^3$ while supporting a wider range of modal inputs.

## 🚀 Quick Start

### 1. Clone & Install Dependencies
First, clone the repository and install the required packages.
```bash
git clone https://github.com/yyfz/Pi3.git
cd Pi3
pip install -r requirements.txt
```

### 2\. Run Inference from Command Line

Try our example inference script. You can run it on a directory of images or a video file.

If the automatic download from Hugging Face is slow, you can download the model checkpoint manually from [Pi3](https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors) or [Pi3X](https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors) and specify its local path using the `--ckpt` argument.

```bash
# Run with the default example video
# python example.py    # Inference with Pi3 (Original)
python example_mm.py   # [New] Inference with Pi3X (Recommended)

# Run on your own data (image folder or .mp4 file)
# python example.py --data_path <path/to/data>     # Pi3
python example_mm.py --data_path <path/to/data>    # Pi3X
```

### Advanced: Multimodal Conditioning (Pi3X Only)
To utilize additional input modalities (e.g., camera poses, intrinsics, or depth), please refer to example_mm.py for specific data formatting details.

Below is an example comparing reconstruction with and without condition injection. You can compare the resulting point clouds to observe the improvements brought by multimodal inputs.
``` bash
# 1. Inference WITH conditioning (poses, intrinsics, etc.)
python example_mm.py --data_path examples/room/rgb --conditions_path examples/room/condition.npz --save_path examples/room_with_conditions.ply

# 2. Inference WITHOUT conditioning (image only)
python example_mm.py --data_path examples/room/rgb --save_path examples/room_no_conditions.ply
```

**Optional Arguments:**

  * `--data_path`: Path to the input image directory or a video file. (Default: `examples/skating.mp4`)
  * `--save_path`: Path to save the output `.ply` point cloud. (Default: `examples/result.ply`)
  * `--interval`: Frame sampling interval. (Default: `1` for images, `10` for video)
  * `--ckpt`: Path to a custom model checkpoint file.
  * `--device`: Device to run inference on. (Default: `cuda`)

### 3\. Run with Gradio Demo

You can also launch a local Gradio demo for an interactive experience.

```bash
# Install demo-specific requirements
pip install -r requirements_demo.txt

# Launch the demo
python demo_gradio.py
```


## 🛠️ Detailed Usage

### Model Input & Output

The model takes a tensor of images and outputs a dictionary containing the reconstructed geometry.

  * **Input**: A `torch.Tensor` of shape $B \times N \times 3 \times H \times W$ with pixel values in the range `[0, 1]`.
  * **Output**: A `dict` with the following keys:
      * `points`: Global point cloud unprojected by `local points` and `camerae_poses` (`torch.Tensor`, $B \times N \times H \times W \times 3$).
      * `local_points`: Per-view local point maps (`torch.Tensor`,  $B \times N \times H \times W \times 3$).
      * `conf`: Confidence scores for local points (Raw confidence logits. Apply `torch.sigmoid()` to obtain probabilities in `[0, 1]`, higher is better) (`torch.Tensor`,  $B \times N \times H \times W \times 1$).
      * `camera_poses`: Camera-to-world transformation matrices (`4x4` in OpenCV format) (`torch.Tensor`,  $B \times N \times 4 \times 4$).

### Example Code Snippet

Here is a minimal example of how to run the model on a batch of images.

```python
import torch
# from pi3.models.pi3 import Pi3            # old version
from pi3.models.pi3x import Pi3X            # new version (Recommended)
from pi3.utils.basic import load_images_as_tensor # Assuming you have a helper function

# --- Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
# or download checkpoints from `https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors`

# --- Load Data ---
# Load a sequence of N images into a tensor
# imgs shape: (N, 3, H, W).
# imgs value: [0, 1]
imgs = load_images_as_tensor('path/to/your/data', interval=10).to(device)

# --- Inference ---
print("Running model inference...")
# Use mixed precision for better performance on compatible GPUs
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=dtype):
        # Add a batch dimension -> (1, N, 3, H, W)
        results = model(imgs[None])

print("Reconstruction complete!")
# Access outputs: results['points'], results['camera_poses'] and results['local_points'].
```


## 🙏 Acknowledgements

Our work builds upon several fantastic open-source projects. We'd like to express our gratitude to the authors of:

  * [DUSt3R](https://github.com/naver/dust3r)
  * [CUT3R](https://github.com/CUT3R/CUT3R)
  * [VGGT](https://github.com/facebookresearch/vggt)


## 📜 Citation

If you find our work useful, please consider citing:

```bibtex
@article{wang2025pi,
  title={$$\backslash$pi\^{} 3$: Permutation-Equivariant Visual Geometry Learning},
  author={Wang, Yifan and Zhou, Jianjun and Zhu, Haoyi and Chang, Wenzheng and Zhou, Yang and Li, Zizun and Chen, Junyi and Pang, Jiangmiao and Shen, Chunhua and He, Tong},
  journal={arXiv preprint arXiv:2507.13347},
  year={2025}
}
```




## 📄 License
This project adopts a dual-licensing strategy:

| Component | License | Commercial Use |
| :--- | :--- | :--- |
| **Code** (Scripts, Tools, Logic) | [BSD 3-Clause](LICENSE) | **Permitted** |
| **Model Weights** (Pi3 and Pi3X Weights) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | **Strictly Non-Commercial** |

**Note on Model Weights:** Due to the nature of the training datasets, the model weights are restricted to non-commercial research and educational purposes only. Redistribution of the weights must maintain this restriction.

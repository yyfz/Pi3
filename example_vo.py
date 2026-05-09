import argparse
import os

import torch

from pi3.models.pi3x import Pi3X
from pi3.pipe.pi3x_vo import Pi3XVO
from pi3.utils.basic import load_multimodal_data, write_ply

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    
    parser.add_argument("--data_path", type=str, default='examples/skating.mp4',
                        help="Path to the input image directory or a video file.")
    
    parser.add_argument("--save_path", type=str, default='examples/result.ply',
                        help="Path to save the output .ply file.")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
                        
    args = parser.parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1
    print(f'Sampling interval: {args.interval}')

    # 1. Prepare model
    print("Loading model...")
    device = torch.device(args.device)
    if args.ckpt is not None:
        model = Pi3X().to(device).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        
        model.load_state_dict(weight, strict=False)
    else:
        model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
        # or download checkpoints from `https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors`

    pipe = Pi3XVO(model)

    # 2. Prepare input data
    # Load images (Required)
    imgs, _ = load_multimodal_data(args.data_path, conditions=None, interval=args.interval, device=device) 

    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        res = pipe(
            imgs=imgs, 
            dtype=dtype,
        )

    # 4. process mask
    masks = res['conf'][0] > 0.05

    # 5. Save points
    print(f"Saving point cloud to: {args.save_path}")
    if os.path.dirname(args.save_path):
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        
    write_ply(res['points'][0][masks].cpu(), imgs[0].permute(0, 2, 3, 1)[masks], args.save_path)
    print("Done.")

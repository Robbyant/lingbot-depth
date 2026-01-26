# LingBot-Depth: Masked Depth Modeling for Spatial Perception

[![arXiv](https://img.shields.io/badge/arXiv-2401.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2401.xxxxx)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**LingBot-Depth** transforms incomplete and noisy depth sensor data into high-quality, metric-accurate 3D measurements. By jointly aligning RGB appearance and depth geometry in a unified latent space, our model serves as a powerful spatial perception foundation for robot learning and 3D vision applications.

<p align="center">
  <img src="assets/teaser/teaser-crop.png" width="100%">
</p>

Our approach refines raw sensor depth into clean, complete measurements, enabling:
- **Depth Completion & Refinement**: Fills missing regions with metric accuracy and improved quality
- **Scene Reconstruction**: High-fidelity indoor mapping with a strong depth prior
- **4D Point Tracking**: Accurate dynamic tracking in metric space for robot learning
- **Dexterous Manipulation**: Robust grasping with precise geometric understanding

## Artifcats Release


### Model Zoo

We provide pretrained models for different scenarios:

| Model | Checkpoint | Description |
|-------|-----------|-------------|
| LingBot-Depth | [model.pt](https://example.com/model.pt) | General-purpose depth refinement |
| LingBot-Depth-DC | [model.pt](https://example.com/model-dc.pt) | Optimized for sparse depth completion |

### Data
The curated RGB-D dataset (3M RGB-depth pairs, including 2M real and 1M simulated samples) will be released upon completion of the necessary licensing and approval procedures. Our expected date is the mid of March.

## Installation

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0.0
- CUDA-capable GPU (recommended)

### From source

```bash
git clone <repository-url>
cd lingbot-depth
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

## Quick Start

**Inference:**

```python
import torch
import cv2
import numpy as np
from morgbd.model.v2 import MoRGBDModel

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MoRGBDModel.from_pretrained('ckpt/model.pt').to(device)

# Load and prepare inputs
image = cv2.imread('examples/0/rgb.png')
image = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)[None]

depth = cv2.imread('examples/0/raw_depth.png', cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
depth = torch.tensor(depth, dtype=torch.float32, device=device)[None]

intrinsics = np.loadtxt('examples/0/intrinsics.txt')
intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=device)[None]

# Run inference
output = model.infer(
    image,
    depth_in=depth,
    intrinsics=intrinsics)

depth_pred = output['depth']  # Refined depth map
points = output['points']      # 3D point cloud
```

**Run example:**

```bash
python example.py
```

This processes the example data from `examples/0/` and saves visualizations to `result/`.

## Method

We introduce a masked depth modeling approach that learns robust RGB-D representations through self-supervised learning. The model employs a Vision Transformer encoder with specialized depth-aware attention mechanisms to jointly process RGB and depth inputs.

<p align="center">
  <img src="assets/attention/fig-attention-vis.png" width="100%">
</p>

**Depth-aware attention visualization.** Visualizing attention from depth queries (Q1–Q3, marked with ⋆) to RGB tokens in two scenes: (a) aquarium and (b) indoor shelf. Each row shows masked input depth, attention weights on RGB, and refined output. Different queries attend to spatially corresponding regions, demonstrating cross-modal alignment.

**Key Innovations:**
- **Masked Depth Modeling**: Self-supervised pre-training via depth reconstruction
- **Cross-Modal Attention**: Joint RGB-Depth alignment in unified latent space
- **Metric-Scale Preservation**: Maintains real-world measurements for downstream tasks

## Training Data

Our model is trained on a large-scale diverse dataset combining real-world and simulated RGB-D captures:

<p align="center">
  <img src="assets/dataset/diversity_figure.png" width="100%">
</p>

**Training dataset.** 2.1M real-world and 1.0M simulated samples spanning diverse indoor environments (top). Representative RGB-D inputs with ground truth depth (bottom).

**Dataset Composition:**
- **Real Captures**: 2.1M samples from residential, office, and commercial environments
- **Simulated Data**: 1.0M photo-realistic renders with perfect ground truth
- **Modalities**: RGB images, raw depth, refined ground truth depth
- **Diversity**: Multiple sensor types, lighting conditions, and scene complexities

## Applications

### 4D Point Tracking

LingBot-Depth provides metric-accurate 3D geometry essential for tracking dynamic targets:

<p align="center">
  <img src="assets/downstream_tracking/fig-dynamic-tracking.png" width="100%">
</p>

**4D point tracking.** Robust tracking in gym environments with dynamic human motion. Top: query point selection. Middle: 3D tracking on deforming geometry. Bottom: refined depth maps. Demonstrated on scooter, rowing machine, gym equipment, and pull-up bar.

### Dexterous Manipulation

High-quality geometric understanding enables reliable robotic grasping across diverse objects and materials:

<p align="center">
  <img src="assets/downstream_grasp/fig-grasp-demo.png" width="100%">
</p>

**Dexterous grasping.** Robust manipulation enabled by refined depth. Top: point cloud reconstruction. Bottom: successful grasps on steel cup, glass cup, storage box, and toy car.

## Hardware Setup

We developed a scalable RGB-D capture system for large-scale data collection:

<p align="center">
  <img src="assets/device/device-full.jpg" width="60%">
</p>

**RGB-D capture system.** Multi-sensor setup with Intel RealSense, Orbbec Gemini, and Azure Kinect for scalable real-world data collection.

## Model Details

### Architecture

- **Encoder**: Vision Transformer (Large) with RGB-D fusion
- **Decoder**: Multi-scale feature pyramid with specialized heads
- **Heads**: Depth regression
- **Training**: Masked depth modeling with reconstruction objective

### Input Format

**RGB Image:**
- Shape: `[B, 3, H, W]` normalized to [0, 1]
- Format: PyTorch tensor, float32

**Depth Map:**
- Shape: `[B, H, W]`
- Unit: Meters (configurable via scale parameter)
- Invalid regions: 0 or NaN

**Camera Intrinsics:**
- Shape: `[B, 3, 3]`
- Normalized format: `fx'=fx/W, fy'=fy/H, cx'=cx/W, cy'=cy/H`
- Example:
  ```
  [[fx/W,   0,   cx/W],
   [  0,  fy/H,  cy/H],
   [  0,    0,    1  ]]
  ```

### Output Format

The model returns a dictionary:

```python
{
    'depth': torch.Tensor,   # Refined depth [B, H, W]
    'points': torch.Tensor,  # Point cloud [B, H, W, 3] in camera space
}
```

### Inference Parameters

```python
model.infer(
    image,                                   # RGB tensor [B, 3, H, W]
    depth_in=None,                           # Input depth [B, H, W]
    use_fp16=True,                           # Mixed precision inference
    intrinsics=None,                         # Camera intrinsics [B, 3, 3]
)
```

## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{lingbot-depth2026,
  title={Masked Depth Modeling for Spatial Perception},
  author={Tan, Bin and Sun, Changjiang and Qin, Xiage and Adai, Hanat and Fu, Zelin and Zhou, Tianxiang and Zhang, Han and Xu, Yinghao and Zhu, Xing and Shen, Yujun and Xue, Nan},
  journal={arXiv preprint arXiv:2401.xxxxx},
  year={2026}
}
```

Please also consider citing DINOv2, which serves as our backbone:

```bibtex
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={Transactions on Machine Learning Research},
  year={2024}
}
```

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon several excellent open-source projects:

- [DINOv2](https://github.com/facebookresearch/dinov2) - Self-supervised vision transformer backbone
- [Masked Autoencoders](https://github.com/facebookresearch/mae) - Self-supervised learning framework
- The broader open-source computer vision and robotics communities

## Contact

For questions, discussions, or collaborations:

- **Issues**: Open an [issue](https://github.com/example/RGBD-MAE/issues) on GitHub
- **Email**: Contact the corresponding authors

---

📄 **Paper:** [Masked Depth Modeling for Spatial Perception](https://example.com)
🌐 **Project Page:** [LingBot-Depth](https://example.com)
💻 **Code:** [GitHub Repository](https://github.com/example/RGBD-MAE)

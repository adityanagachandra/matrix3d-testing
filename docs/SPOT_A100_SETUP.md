# Reproducing the environment on a cloud spot instance (e.g. A100)

This fork pins **`transformers>=4.44,<5`** in `requirements.txt` so `diffusers==0.31.0` / HunyuanDiT can import `MT5Tokenizer` (removed from the public API in Transformers 5.x). Install **after** nerfstudio if pip upgrades you back to 5.x:

```bash
pip install "transformers>=4.44,<5"
```

## 1. Instance

- **GPU**: A100 (or any recent NVIDIA GPU). Driver must support your chosen PyTorch CUDA build.
- **OS**: Ubuntu 20.04/22.04 is a good match for the upstream README.

## 2. System dependencies

- NVIDIA driver + `nvcc` if you build extensions from source.
- For **`tiny-cuda-nn`**: the compiler’s CUDA major version should match PyTorch’s (e.g. PyTorch **cu118** → use CUDA **11.8** `nvcc`). If the machine only has CUDA 12.x system-wide, install a matching toolkit in conda and point **`CUDA_HOME`** at that env when installing `tiny-cuda-nn` (see main README troubleshooting).

```bash
conda install -y -n matrix3d -c nvidia/label/cuda-11.8.0 cuda-toolkit
# Then, when building tiny-cuda-nn:
export CUDA_HOME="$CONDA_PREFIX"
```

## 3. Conda env (same stack as upstream README)

```bash
conda create -y -n matrix3d python=3.10
conda activate matrix3d
pip install torch==2.4.0 torchvision==0.19.0 xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.7+pt2.4.0cu118
pip install -r requirements.txt
pip install timm==1.0.11
pip install "transformers>=4.44,<5"   # safety if nerfstudio pulled transformers 5.x
```

Optional: **`setuptools<71`** and **`pip install cmake ninja`** if `tiny-cuda-nn` fails on `pkg_resources` or build backends.

## 4. Checkpoints (not in git)

Create `checkpoints/` and add weights (see project README):

- [matrix3d_512.pt](https://ml-site.cdn-apple.com/models/matrix3d/matrix3d_512.pt)
- Optional for single-view: `isnet-general-use.pth` from [DIS / IS-Net](https://github.com/xuebinqin/DIS)

## 5. Spot instances

- Persist **`checkpoints/`** and **`outputs/`** / **`results/`** on a volume or sync out before termination.
- For long jobs, use `nohup`, `tmux`, or a job scheduler; spot preemption will kill the process unless you checkpoint and resume manually.

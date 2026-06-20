import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


# ========================== 1. 数据集（路径不变） ==========================

class COCORaw(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        self.imgs = [p for p in self.root.iterdir() if p.suffix.lower() in exts]
        self.imgs.sort()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# ========================== 2. 重叠窗口工具 ==========================

def window_partition_overlap(x, window_size, stride):
    B, C, H, W = x.shape
    ws = window_size
    x = F.unfold(x, kernel_size=ws, stride=stride)
    L = x.shape[-1]
    x = x.view(B, C, ws * ws, L)
    x = x.permute(0, 3, 2, 1).contiguous()
    return x, stride


def window_reverse_overlap(x, B, C, H, W, window_size, stride):
    ws = window_size
    x = x.permute(0, 3, 2, 1).contiguous()
    x = x.view(B, C * ws * ws, -1)
    out = F.fold(x, output_size=(H, W), kernel_size=ws, stride=stride)

    ones = torch.ones(1, 1, H, W, device=x.device, dtype=x.dtype)
    ones_unfold = F.unfold(ones, kernel_size=ws, stride=stride)
    norm = F.fold(ones_unfold, output_size=(H, W), kernel_size=ws, stride=stride)
    out = out / (norm + 1e-8)
    return out


# ========================== 3. 模型（原样保留） ==========================

class MultiScaleStem(nn.Module):
    def __init__(self, out_dim=18):
        super().__init__()
        assert out_dim == 18, "out_dim must be 18"
        self.scales = [1, 2, 4, 8, 16, 32]

    def forward(self, x):
        x = (x + 1.0) / 2.0
        x = x.clamp(0, 1)
        B, C, H, W = x.shape
        feats = []
        for s in self.scales:
            if s == 1:
                pooled = x
            else:
                pooled = F.max_pool2d(x, kernel_size=s, stride=s)
                pooled = F.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=False)
            feats.append(pooled)
        out = torch.cat(feats, dim=1)
        return out


class CNNTransformerBlock(nn.Module):
    def __init__(self, dim=18, window_size=4):
        super().__init__()
        self.dim = dim
        self.win = window_size
        self.shift = window_size // 2
        self.stride = window_size // 2

        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.q_conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.q_bn1 = nn.BatchNorm2d(dim)
        self.q_conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.q_bn2 = nn.BatchNorm2d(dim)

        self.out_norm = nn.BatchNorm2d(dim)

    def _window_attn(self, Q):
        B, C, H, W = Q.shape
        Q_w, stride = window_partition_overlap(Q, self.win, self.stride)
        attn = torch.matmul(Q_w, Q_w.transpose(-2, -1)) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, Q_w)
        out = window_reverse_overlap(out, B, C, H, W, self.win, stride)
        return out

    def forward(self, x):
        local = self.local(x)
        Q = self.q_bn2(self.q_conv2(F.relu(self.q_bn1(self.q_conv1(x)))))
        B1 = self._window_attn(Q)
        Q_shifted = torch.roll(Q, shifts=(self.shift, self.shift), dims=(2, 3))
        B2 = self._window_attn(Q_shifted)
        B2 = torch.roll(B2, shifts=(-self.shift, -self.shift), dims=(2, 3))
        attn = B1 + B2
        out = F.relu(self.out_norm(x + local + attn))
        return out


class TextureSegmentor(nn.Module):
    def __init__(self, dim=18, window_size=4):
        super().__init__()
        self.stem = MultiScaleStem(out_dim=dim)
        self.block1 = CNNTransformerBlock(dim=dim, window_size=window_size)
        self.block2 = CNNTransformerBlock(dim=dim, window_size=window_size)
        self.block3 = CNNTransformerBlock(dim=dim, window_size=window_size)

    def forward(self, x):
        z = self.stem(x)
        z = self.block1(z)
        z = self.block2(z)
        z = self.block3(z)
        return z


# ========================== 4. 可视化工具（原样保留） ==========================

@torch.no_grad()
def pca_project(z):
    C, H, W = z.shape
    x = z.reshape(C, -1).T
    x_mean = x.mean(dim=0, keepdim=True)
    x_centered = x - x_mean
    U, S, Vt = torch.linalg.svd(x_centered, full_matrices=False)
    proj = x_centered @ Vt[:3].T
    proj = proj - proj.min(dim=0, keepdim=True)[0]
    proj = proj / (proj.max(dim=0, keepdim=True)[0] + 1e-8)
    return proj.T.reshape(3, H, W)


@torch.no_grad()
def kmeans_cluster(z, K=8, max_iter=10, merge_thresh=0.8):
    C, H, W = z.shape
    x = z.reshape(C, -1).T
    N = x.shape[0]
    idx = torch.randperm(N, device=x.device)[:K]
    centers = x[idx].clone()

    for _ in range(max_iter):
        dists = torch.cdist(x, centers)
        labels = dists.argmin(dim=1)
        for k in range(K):
            mask = labels == k
            if mask.any():
                centers[k] = x[mask].mean(dim=0)

    labels = labels.reshape(H, W)

    valid_centers = []
    valid_ids = []
    for k in range(K):
        mask = labels == k
        if mask.any():
            mean_feat = z[:, mask].mean(dim=1)
            valid_centers.append(mean_feat)
            valid_ids.append(k)

    M = len(valid_ids)
    if M < 2:
        return labels

    valid_centers = torch.stack(valid_centers)
    valid_centers_norm = F.normalize(valid_centers, p=2, dim=1)
    sim_matrix = valid_centers_norm @ valid_centers_norm.T

    parent = list(range(M))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(M):
        for j in range(i + 1, M):
            if sim_matrix[i, j] > merge_thresh:
                union(i, j)

    root_to_new = {}
    new_label = 0
    old_to_new = {}
    for i in range(M):
        root = find(i)
        if root not in root_to_new:
            root_to_new[root] = new_label
            new_label += 1
        old_to_new[valid_ids[i]] = root_to_new[root]

    remapped = torch.zeros_like(labels)
    for old_id, new_id in old_to_new.items():
        remapped[labels == old_id] = new_id

    return remapped


def label_color(labels):
    n = labels.max().item() + 1
    rng = np.random.RandomState(42)
    colors = rng.randint(0, 255, (max(n, 1), 3), dtype=np.uint8)
    return colors[labels.cpu().numpy()]


# ========================== 5. 带中文标注的可视化 ==========================

def get_chinese_font(size=18):
    """Windows 常见中文字体路径"""
    candidates = [
        r"C:\Windows\Fonts\simhei.ttf",      # 黑体
        r"C:\Windows\Fonts\msyh.ttc",        # 微软雅黑
        r"C:\Windows\Fonts\simsun.ttc",      # 宋体
        r"C:\Windows\Fonts\simkai.ttf",      # 楷体
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_label(img_array, text, font):
    """在 numpy 数组左上角画中文，返回 numpy 数组"""
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    # 简单描边，保证任何背景都可见
    x, y = 5, 5
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        draw.text((x+dx, y+dy), text, font=font, fill=(0,0,0))
    draw.text((x, y), text, font=font, fill=(255, 0, 0))
    return np.array(img)


@torch.no_grad()
def save_vis_with_labels(img_tensor, z_tensor, path, font):
    orig = img_tensor.cpu().numpy().transpose(1, 2, 0)
    orig = np.clip((orig + 1.0) / 2.0, 0, 1)
    orig_uint8 = (orig * 255).astype(np.uint8)

    pca = pca_project(z_tensor).cpu().numpy().transpose(1, 2, 0)
    pca_uint8 = np.clip(pca * 255, 0, 255).astype(np.uint8)

    labels_z = kmeans_cluster(z_tensor, K=12, max_iter=10, merge_thresh=0.95)
    seg_z = label_color(labels_z)

    labels_rgb = kmeans_cluster(img_tensor, K=12, max_iter=10, merge_thresh=1.0)
    seg_rgb = label_color(labels_rgb)

    # 加中文标注
    orig_uint8 = draw_label(orig_uint8, "原始图像", font)
    pca_uint8  = draw_label(pca_uint8,  "PCA特征投影", font)
    seg_z      = draw_label(seg_z,      "纹理聚类分割", font)
    seg_rgb    = draw_label(seg_rgb,    "RGB色彩聚类", font)

    H, W = orig_uint8.shape[:2]
    canvas = Image.new("RGB", (W * 4, H))
    canvas.paste(Image.fromarray(orig_uint8), (0,     0))
    canvas.paste(Image.fromarray(pca_uint8),  (W,     0))
    canvas.paste(Image.fromarray(seg_z),      (W * 2, 0))
    canvas.paste(Image.fromarray(seg_rgb),    (W * 3, 0))
    canvas.save(path)


# ========================== 6. 主逻辑：生成 100 张 ==========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 路径保持原样
    coco_root = r"data\coco\train2017"

    transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = COCORaw(coco_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=1,          # 一次一张，方便处理
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    win_size = 4
    model = TextureSegmentor(dim=18, window_size=win_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}")

    # 加载训练好的权重
    ckpt_latest = Path("checkpoints") / "encoder.pth"
    if ckpt_latest.exists():
        ckp = torch.load(ckpt_latest, map_location=device)
        model.load_state_dict(ckp["model_state_dict"])
        print(f"Loaded checkpoint from {ckpt_latest}")
    else:
        print(f"Warning: checkpoint {ckpt_latest} not found, using random weights.")

    model.eval()

    # 输出目录
    out_dir = Path("textsp")
    out_dir.mkdir(exist_ok=True)

    # 加载字体
    font = get_chinese_font(size=18)

    # 生成 100 张
    count = 0
    pbar = tqdm(total=100, desc="Generating")
    for imgs in loader:
        if count >= 100:
            break

        imgs = imgs.to(device)
        with torch.no_grad():
            z = model(imgs)

        # 随机数命名：8 位随机整数
        rand_name = f"{random.randint(10000000, 99999999)}.png"
        save_path = out_dir / rand_name

        save_vis_with_labels(imgs[0], z[0], str(save_path), font)

        count += 1
        pbar.update(1)

    pbar.close()
    print(f"Done. {count} images saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
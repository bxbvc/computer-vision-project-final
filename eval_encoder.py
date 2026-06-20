import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages  # ← 新增：PDF后端

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


# ========================== 中文字体加载 ==========================
font_paths = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]
loaded = False
for fp in font_paths:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        prop = font_manager.FontProperties(fname=fp)
        plt.rcParams['font.family'] = prop.get_name()
        loaded = True
        break
if not loaded:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ========================== 模型定义 ==========================

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


# ========================== 数据集 ==========================

class COCOTestDataset(Dataset):
    def __init__(self, img_root, coco, transform=None, seed=42):
        self.img_root = Path(img_root)
        self.coco = coco
        self.img_ids = coco.getImgIds()
        self.transform = transform
        self.seed = seed

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_root / img_info['file_name']
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        return img, anns, img_info


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    anns = [b[1] for b in batch]
    infos = [b[2] for b in batch]
    return imgs, anns, infos


# ========================== K-means ==========================

@torch.no_grad()
def kmeans_cluster(features, num_clusters=8, max_iter=10):
    C, H, W = features.shape
    x = features.reshape(C, -1).T
    N = x.shape[0]
    indices = torch.randperm(N, device=x.device)[:num_clusters]
    centers = x[indices].clone()

    for _ in range(max_iter):
        dists = torch.cdist(x, centers)
        labels = dists.argmin(dim=1)
        for k in range(num_clusters):
            mask = labels == k
            if mask.any():
                centers[k] = x[mask].mean(dim=0)

    return labels.reshape(H, W)


# ========================== IoU 与 连通区域 ==========================

def compute_iou(mask_a, mask_b):
    intersection = (mask_a & mask_b).sum().float()
    union = (mask_a | mask_b).sum().float()
    if union == 0:
        return 0.0
    return (intersection / union).item()


def count_connected_components(mask_np):
    """纯numpy 4-连通区域计数"""
    if mask_np.sum() == 0:
        return 0
    h, w = mask_np.shape
    visited = np.zeros((h, w), dtype=bool)
    count = 0
    for i in range(h):
        for j in range(w):
            if mask_np[i, j] and not visited[i, j]:
                count += 1
                stack = [(i, j)]
                visited[i, j] = True
                while stack:
                    ci, cj = stack.pop()
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < h and 0 <= nj < w and mask_np[ni, nj] and not visited[ni, nj]:
                            visited[ni, nj] = True
                            stack.append((ni, nj))
    return count


def eval_mask_clustering(image, latent, annotations, coco, img_info, num_clusters=8):
    device = image.device
    H, W = 192, 192
    orig_h, orig_w = img_info['height'], img_info['width']
    scale_x = W / orig_w
    scale_y = H / orig_h

    labels_z = kmeans_cluster(latent, num_clusters=num_clusters, max_iter=10)
    rgb = (image + 1.0) / 2.0
    labels_rgb = kmeans_cluster(rgb, num_clusters=num_clusters, max_iter=10)

    iou_z_list = []
    iou_rgb_list = []
    cc_z_list = []      # 连通区域数
    cc_rgb_list = []

    for ann in annotations:
        mask_raw = coco.annToMask(ann)
        mask_tensor = torch.from_numpy(mask_raw).float().to(device)
        mask = F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest')[0, 0]
        mask_bool = mask > 0.5

        if mask_bool.sum() < 10:
            continue

        # bbox 映射到 192x192
        x, y, bw, bh = ann['bbox']
        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + bw) * scale_x)
        y2 = int((y + bh) * scale_y)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        # Z空间
        labels_in_mask_z = labels_z[mask_bool]
        if labels_in_mask_z.numel() == 0:
            continue
        unique_z, counts_z = torch.unique(labels_in_mask_z, return_counts=True)
        dominant_z = unique_z[counts_z.argmax()]
        cluster_mask_z = (labels_z == dominant_z)
        iou_z = compute_iou(cluster_mask_z, mask_bool)
        iou_z_list.append(iou_z)

        # Z主导簇在bbox内的连通区域数
        bbox_region_z = cluster_mask_z[y1:y2, x1:x2].cpu().numpy()
        cc_z_list.append(count_connected_components(bbox_region_z))

        # RGB空间
        labels_in_mask_rgb = labels_rgb[mask_bool]
        if labels_in_mask_rgb.numel() == 0:
            continue
        unique_rgb, counts_rgb = torch.unique(labels_in_mask_rgb, return_counts=True)
        dominant_rgb = unique_rgb[counts_rgb.argmax()]
        cluster_mask_rgb = (labels_rgb == dominant_rgb)
        iou_rgb = compute_iou(cluster_mask_rgb, mask_bool)
        iou_rgb_list.append(iou_rgb)

        # RGB主导簇在bbox内的连通区域数
        bbox_region_rgb = cluster_mask_rgb[y1:y2, x1:x2].cpu().numpy()
        cc_rgb_list.append(count_connected_components(bbox_region_rgb))

    return iou_z_list, iou_rgb_list, cc_z_list, cc_rgb_list


# ========================== 主流程 ==========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ann_file = r"D:\WorkSpace\class-pjs\computer-vision\data\coco\annotations\instances_train2017.json"
    img_root = r"data\coco\train2017"
    ckpt_path = r"checkpoints\encoder.pth"

    save_dir = Path("eval_encoder")
    save_dir.mkdir(exist_ok=True)

    from pycocotools.coco import COCO
    coco = COCO(ann_file)

    transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = COCOTestDataset(img_root, coco, transform=transform, seed=42)

    # 随机选100个
    rng = np.random.RandomState(42)
    dataset.img_ids = rng.choice(dataset.img_ids, size=100, replace=False).tolist()
    print(f"测试图片数量: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = TextureSegmentor(dim=18, window_size=4).to(device)
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"已加载模型: {ckpt_path}")
    else:
        print("警告: 未找到模型，使用随机权重")
    model.eval()

    all_iou_z = []
    all_iou_rgb = []
    all_cc_z = []
    all_cc_rgb = []

    pbar = tqdm(loader, desc="测试中")
    for imgs, anns_list, info_list in pbar:
        img = imgs[0].to(device)
        anns = anns_list[0]
        info = info_list[0]

        with torch.no_grad():
            z = model(img.unsqueeze(0))[0]

        iou_z, iou_rgb, cc_z, cc_rgb = eval_mask_clustering(img, z, anns, coco, info, num_clusters=8)

        all_iou_z.extend(iou_z)
        all_iou_rgb.extend(iou_rgb)
        all_cc_z.extend(cc_z)
        all_cc_rgb.extend(cc_rgb)

        if len(all_iou_z) > 0:
            pbar.set_postfix({
                "Z_IoU": f"{np.mean(all_iou_z):.3f}",
                "RGB_IoU": f"{np.mean(all_iou_rgb):.3f}",
                "Z_CC": f"{np.mean(all_cc_z):.2f}",
                "RGB_CC": f"{np.mean(all_cc_rgb):.2f}",
            })

    print("\n========== 结果 ==========")
    print(f"Z空间聚类 平均IoU:   {np.mean(all_iou_z):.4f} ± {np.std(all_iou_z):.4f}")
    print(f"RGB空间聚类 平均IoU: {np.mean(all_iou_rgb):.4f} ± {np.std(all_iou_rgb):.4f}")
    print(f"Z - RGB IoU提升:     {np.mean(all_iou_z) - np.mean(all_iou_rgb):.4f}")
    print()
    print(f"Z空间 平均连通区域数:   {np.mean(all_cc_z):.4f} ± {np.std(all_cc_z):.4f}")
    print(f"RGB空间 平均连通区域数: {np.mean(all_cc_rgb):.4f} ± {np.std(all_cc_rgb):.4f}")
    print(f"Z - RGB 连通区域减少:   {np.mean(all_cc_rgb) - np.mean(all_cc_z):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 图1: IoU分布
    axes[0].hist(all_iou_rgb, bins=30, alpha=0.6, label="RGB聚类", color='orange', edgecolor='black')
    axes[0].hist(all_iou_z, bins=30, alpha=0.6, label="Z空间聚类", color='steelblue', edgecolor='black')
    axes[0].axvline(np.mean(all_iou_rgb), color='orange', linestyle='--', linewidth=2, label=f"RGB均值: {np.mean(all_iou_rgb):.3f}")
    axes[0].axvline(np.mean(all_iou_z), color='steelblue', linestyle='--', linewidth=2, label=f"Z均值: {np.mean(all_iou_z):.3f}")
    axes[0].set_xlabel("IoU (标注Mask vs 最大聚类簇)")
    axes[0].set_ylabel("频数")
    axes[0].set_title("聚类IoU分布对比")
    axes[0].legend()

    # 图2: 逐Mask IoU散点
    min_len = min(len(all_iou_z), len(all_iou_rgb))
    x_vals = np.arange(min_len)
    axes[1].scatter(x_vals, all_iou_rgb[:min_len], s=10, alpha=0.5, label="RGB聚类", color='orange')
    axes[1].scatter(x_vals, all_iou_z[:min_len], s=10, alpha=0.5, label="Z空间聚类", color='steelblue')
    axes[1].axhline(np.mean(all_iou_rgb), color='orange', linestyle='--', alpha=0.7)
    axes[1].axhline(np.mean(all_iou_z), color='steelblue', linestyle='--', alpha=0.7)
    axes[1].set_xlabel("Mask序号")
    axes[1].set_ylabel("IoU")
    axes[1].set_title("逐Mask IoU对比")
    axes[1].legend()

    # 图3: 连通区域数分布
    max_cc = max(max(all_cc_rgb) if all_cc_rgb else 1, max(all_cc_z) if all_cc_z else 1)
    bins = np.arange(1, max_cc + 2) - 0.5
    axes[2].hist(all_cc_rgb, bins=bins, alpha=0.6, label="RGB聚类", color='orange', edgecolor='black')
    axes[2].hist(all_cc_z, bins=bins, alpha=0.6, label="Z空间聚类", color='steelblue', edgecolor='black')
    axes[2].axvline(np.mean(all_cc_rgb), color='orange', linestyle='--', linewidth=2, label=f"RGB均值: {np.mean(all_cc_rgb):.2f}")
    axes[2].axvline(np.mean(all_cc_z), color='steelblue', linestyle='--', linewidth=2, label=f"Z均值: {np.mean(all_cc_z):.2f}")
    axes[2].set_xlabel("BBox内主导簇连通区域数")
    axes[2].set_ylabel("频数")
    axes[2].set_title("连通区域碎片化对比")
    axes[2].legend()
    axes[2].set_xticks(np.arange(1, max_cc + 1))

    plt.tight_layout()

    # ========================== 修改：输出PDF ==========================
    save_path = save_dir / "results.pdf"
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        # 可选：设置PDF元数据（学术出版常用）
        d = pdf.infodict()
        d['Title'] = 'Texture Segmentation Evaluation'
        d['Author'] = 'Anonymous'
        d['Subject'] = 'Z-space vs RGB Clustering on COCO'
    plt.close()
    # ========================== 修改结束 ==========================

    print(f"PDF已保存: {save_path}")
    print("完成。")


if __name__ == "__main__":
    main()
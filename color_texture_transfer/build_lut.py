from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch


def splat_3d(x, bins):
    """x: (1,3,H,W) in [0,1], return (bins,bins,bins) 累加器，全程在 x.device"""
    x = x.clamp(0, 1)
    _, _, H, W = x.shape
    x = x.permute(0, 2, 3, 1).reshape(1, H * W, 3)
    pos = x * (bins - 1)
    pos = pos.clamp(0, bins - 1 - 1e-4)
    base = pos.floor().long()
    frac = pos - base.float()

    acc = torch.zeros(bins, bins, bins, device=x.device, dtype=torch.float32)
    for dz in [0, 1]:
        wz = frac[..., 2] if dz else (1 - frac[..., 2])
        for dy in [0, 1]:
            wy = frac[..., 1] if dy else (1 - frac[..., 1])
            for dx in [0, 1]:
                wx = frac[..., 0] if dx else (1 - frac[..., 0])
                w = wx * wy * wz
                idx = base + torch.tensor([dx, dy, dz], device=x.device).long().view(1, 1, 3)
                idx = idx.clamp(0, bins - 1)
                flat = idx[..., 0] * bins * bins + idx[..., 1] * bins + idx[..., 2]
                acc.view(-1).scatter_add_(0, flat[0], w[0])
    return acc


def suppress_gray(prob, bins=33, thresh=2, factor=0.1):
    """
    抑制对角线附近的灰色概率，让彩色峰值更突出。
    prob: numpy array (bins,bins,bins)
    """
    r = np.arange(bins).reshape(bins, 1, 1)
    g = np.arange(bins).reshape(1, bins, 1)
    b = np.arange(bins).reshape(1, 1, bins)
    gray_mask = (np.abs(r - g) <= thresh) & (np.abs(g - b) <= thresh) & (np.abs(r - b) <= thresh)
    prob = prob.copy()
    prob[gray_mask] *= factor
    total = prob.sum()
    if total > 0:
        prob /= total
    return prob


def build_stats(wikiart_root, out_dir, bins=33):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wikiart_root = Path(wikiart_root)

    # 收集所有图片，按类别聚合，不区分 train/test
    class_images = {}
    for split in ["train", "test"]:
        src = wikiart_root / split
        if not src.exists():
            continue
        for style_dir in src.iterdir():
            if not style_dir.is_dir():
                continue
            name = style_dir.name
            if name not in class_images:
                class_images[name] = []
            for p in style_dir.iterdir():
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                    class_images[name].append(p)

    for name, imgs in tqdm(sorted(class_images.items()), desc="Classes"):
        if not imgs:
            continue

        acc = torch.zeros(bins, bins, bins, device=device, dtype=torch.float32)
        for t, p in enumerate(tqdm(imgs, desc=name, leave=False), start=1):
            img = Image.open(p).convert("RGB")
            img = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            img = img.to(device, non_blocking=True)

            h = splat_3d(img, bins)
            acc = (acc * (t - 1) + h) / t

        # 归一化后抑制灰色，再重新归一化
        prob = acc.cpu().numpy()
        total = prob.sum()
        prob = prob / (total + 1e-8)
        prob = suppress_gray(prob, bins=bins, thresh=2, factor=0.1)

        np.save(out_dir / f"{name}.npy", prob.astype(np.float32))


if __name__ == "__main__":
    build_stats(
        wikiart_root=r"data\wikiart_images",
        out_dir="wikiart_stats",
        bins=33,
    )
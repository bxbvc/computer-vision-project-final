import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TVF


# ========================== 1. 颜色迁移模型 ==========================

class StyleColorPredictor(nn.Module):
    def __init__(self, bins=33):
        super().__init__()
        self.bins = bins
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, bins ** 3)
        )

    def forward(self, style_img):
        x = (style_img + 1.0) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        feat = self.features(x).view(x.size(0), -1)
        logits = self.fc(feat)
        log_prob = F.log_softmax(logits, dim=-1)
        prob = log_prob.exp().view(-1, self.bins, self.bins, self.bins)
        return prob, log_prob


# ========================== 2. 纹理分割模型 ==========================

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


class MultiScaleStem(nn.Module):
    def __init__(self, out_dim=18):
        super().__init__()
        assert out_dim == 18
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
        return torch.cat(feats, dim=1)


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


# ========================== 3. 颜色迁移工具 ==========================

SPLAT_OFFSETS = torch.tensor([
    [0,0,0], [1,0,0], [0,1,0], [1,1,0],
    [0,0,1], [1,0,1], [0,1,1], [1,1,1]
], dtype=torch.long)


def splat_3d(x, bins):
    x = x.clamp(0, 1)
    B, C, H, W = x.shape
    N = H * W
    x = x.permute(0, 2, 3, 1).reshape(B, N, 3)
    pos = x * (bins - 1)
    pos = pos.clamp(0, bins - 1 - 1e-4)
    base = pos.floor().long()
    frac = pos - base.float()

    out = torch.zeros(B, bins * bins * bins, device=x.device, dtype=x.dtype)
    offsets = SPLAT_OFFSETS.to(x.device).view(1, 1, 8, 3)
    idx_all = base.unsqueeze(2) + offsets
    idx_all = idx_all.clamp(0, bins - 1)
    flat_all = idx_all[..., 0] * bins * bins + idx_all[..., 1] * bins + idx_all[..., 2]

    w_all = torch.stack([
        (1 - frac[..., 0]) * (1 - frac[..., 1]) * (1 - frac[..., 2]),
        frac[..., 0]       * (1 - frac[..., 1]) * (1 - frac[..., 2]),
        (1 - frac[..., 0]) * frac[..., 1]       * (1 - frac[..., 2]),
        frac[..., 0]       * frac[..., 1]       * (1 - frac[..., 2]),
        (1 - frac[..., 0]) * (1 - frac[..., 1]) * frac[..., 2],
        frac[..., 0]       * (1 - frac[..., 1]) * frac[..., 2],
        (1 - frac[..., 0]) * frac[..., 1]       * frac[..., 2],
        frac[..., 0]       * frac[..., 1]       * frac[..., 2],
    ], dim=2)

    out = torch.scatter_add(out, 1, flat_all.reshape(B, N * 8), w_all.reshape(B, N * 8))
    return out.view(B, bins, bins, bins) / N


def compute_kl(content_01, prob, bins=33):
    H = splat_3d(content_01, bins).squeeze(0)
    H = H + 1e-8
    P = prob + 1e-8
    return (H * (H.log() - P.log())).sum().item()


def transfer_color_lut_final(content, prob, style_img, bins=33, n_iter=800, lr=0.001, gamma=10):
    device = content.device
    B, C, H, W = content.shape
    assert B == 1

    with torch.no_grad():
        style_01 = (style_img + 1.0) / 2.0
        P_style = splat_3d(style_01, bins).squeeze(0)
        P_target = 0.5 * prob + 0.5 * P_style

    t = torch.linspace(0, 1, bins, device=device)
    r = t.view(bins, 1, 1, 1).expand(bins, bins, bins, 1)
    g = t.view(1, bins, 1, 1).expand(bins, bins, bins, 1)
    b = t.view(1, 1, bins, 1).expand(bins, bins, bins, 1)
    identity = torch.cat([r, g, b], dim=-1)

    delta = torch.zeros_like(identity, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    content_01 = (content + 1.0) / 2.0
    grid = content_01.permute(0, 2, 3, 1).unsqueeze(1) * 2.0 - 1.0

    for _ in range(n_iter):
        optimizer.zero_grad()
        lut = (identity + delta).clamp(0, 1)
        lut_gs = lut.permute(3, 2, 1, 0).unsqueeze(0)
        out = F.grid_sample(lut_gs, grid, mode='bilinear',
                            padding_mode='border', align_corners=True)
        out = out.squeeze(2)
        H_dist = splat_3d(out, bins).squeeze(0)
        H_safe = H_dist + 1e-8
        P_safe = P_target + 1e-8
        kl = (H_safe * (H_safe.log() - P_safe.log())).sum()

        tv_r = torch.abs(lut[1:,:,:,:] - lut[:-1,:,:,:]).mean()
        tv_g = torch.abs(lut[:,1:,:,:] - lut[:,:-1,:,:]).mean()
        tv_b = torch.abs(lut[:,:,1:,:] - lut[:,:,:-1,:]).mean()
        tv_lut = tv_r + tv_g + tv_b

        loss = kl + gamma * tv_lut
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        lut_final = (identity + delta).clamp(0, 1)
        lut_gs = lut_final.permute(3, 2, 1, 0).unsqueeze(0)
        out = F.grid_sample(lut_gs, grid, mode='bilinear',
                            padding_mode='border', align_corners=True).squeeze(2)
        return out * 2.0 - 1.0


# ========================== 4. 纹理聚类工具（内部使用）==========================

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


# ========================== 5. 保边滤波 & 平铺工具 ==========================

def guided_filter(guide, src, radius=2, eps=1e-2):
    mean_I = F.avg_pool2d(guide, 2*radius+1, stride=1, padding=radius)
    mean_p = F.avg_pool2d(src, 2*radius+1, stride=1, padding=radius)
    mean_Ip = F.avg_pool2d(guide * src, 2*radius+1, stride=1, padding=radius)
    mean_II = F.avg_pool2d(guide * guide, 2*radius+1, stride=1, padding=radius)
    var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = F.avg_pool2d(a, 2*radius+1, stride=1, padding=radius)
    mean_b = F.avg_pool2d(b, 2*radius+1, stride=1, padding=radius)
    q = mean_a * guide + mean_b
    return q


def guided_filter_rgb(guide, src, radius=2, eps=1e-2):
    C = src.shape[1]
    outs = []
    for c in range(C):
        q = guided_filter(guide, src[:, c:c+1], radius, eps)
        outs.append(q)
    return torch.cat(outs, dim=1)


def extract_bbox(mask):
    ys, xs = torch.where(mask)
    if len(ys) == 0:
        return None
    return ys.min().item(), xs.min().item(), ys.max().item(), xs.max().item()


def tile_patch(patch, target_h, target_w):
    C, h, w = patch.shape
    if h == 0 or w == 0:
        return torch.zeros(C, target_h, target_w, device=patch.device, dtype=patch.dtype)
    repeat_h = (target_h + h - 1) // h
    repeat_w = (target_w + w - 1) // w
    tiled = patch.repeat(1, repeat_h, repeat_w)
    return tiled[:, :target_h, :target_w]


# ========================== 6. 纹理迁移核心 ==========================

def transfer_texture_rgb(content, style, texture_model,
                         n_clusters=8, gf_radius=2, gf_eps=0.01,
                         use_feather=True, feather_sigma=1.5):
    device = content.device
    B, C, H, W = content.shape
    assert B == 1

    guide_c = content.mean(dim=1, keepdim=True)
    guide_s = style.mean(dim=1, keepdim=True)

    base_c = guided_filter_rgb(guide_c, content, radius=gf_radius, eps=gf_eps)
    base_s = guided_filter_rgb(guide_s, style, radius=gf_radius, eps=gf_eps)

    detail_s = style - base_s

    with torch.no_grad():
        z_c = texture_model(content)
        z_s = texture_model(style)

    labels_c = kmeans_cluster(z_c[0], K=n_clusters, max_iter=10, merge_thresh=0.95)
    labels_s = kmeans_cluster(z_s[0], K=n_clusters, max_iter=10, merge_thresh=0.95)
    K_c = labels_c.max().item() + 1
    K_s = labels_s.max().item() + 1

    centers_c = []
    centers_s = []
    for k in range(K_c):
        mask = labels_c == k
        centers_c.append(z_c[0][:, mask].mean(dim=1) if mask.any() else torch.zeros(18, device=device))
    for k in range(K_s):
        mask = labels_s == k
        centers_s.append(z_s[0][:, mask].mean(dim=1) if mask.any() else torch.zeros(18, device=device))

    centers_c = torch.stack(centers_c)
    centers_s = torch.stack(centers_s)

    c_norm = F.normalize(centers_c, p=2, dim=1)
    s_norm = F.normalize(centers_s, p=2, dim=1)
    sim = c_norm @ s_norm.T
    match = sim.argmax(dim=1)

    detail_out = torch.zeros_like(base_c)

    for k in range(K_c):
        m = match[k].item()
        mask_c = (labels_c == k)
        mask_s = (labels_s == m)
        if not mask_c.any() or not mask_s.any():
            continue

        y1s, x1s, y2s, x2s = extract_bbox(mask_s)
        patch_s = detail_s[0, :, y1s:y2s+1, x1s:x2s+1].clone()
        patch_s = patch_s - patch_s.mean(dim=(1, 2), keepdim=True)

        y1c, x1c, y2c, x2c = extract_bbox(mask_c)
        h_c = y2c - y1c + 1
        w_c = x2c - x1c + 1

        tiled = tile_patch(patch_s, h_c, w_c)

        sub_mask = mask_c[y1c:y2c+1, x1c:x2c+1].unsqueeze(0).float()
        if use_feather:
            pad = int(feather_sigma * 3) | 1
            sub_mask = TVF.gaussian_blur(sub_mask, kernel_size=max(pad, 3), sigma=feather_sigma)

        region = detail_out[0, :, y1c:y2c+1, x1c:x2c+1]
        region_new = torch.where(sub_mask.bool().expand_as(region), tiled, region)
        detail_out[0, :, y1c:y2c+1, x1c:x2c+1] = region_new

    result = base_c + detail_out
    return result.clamp(-1, 1), labels_c, labels_s


# ========================== 7. 带中文标注的可视化 ==========================

def get_chinese_font(size=16):
    candidates = [
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\simkai.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_label(img_array, text, font):
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    x, y = 5, 5
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 0, 0))
    return np.array(img)


# ========================== 8. 主流程：2×4 参数对比网格 ==========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    coco_root    = r"data\coco\train2017"
    wikiart_root = r"data\wikiart_images"
    ckpt_color   = Path("checkpoints/latest.pth")
    ckpt_texture = Path("checkpoints/encoder.pth")
    out_dir      = Path("param_grid")
    out_dir.mkdir(exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # ---- 加载模型 ----
    color_model = StyleColorPredictor(bins=33).to(device)
    if not ckpt_color.exists():
        raise FileNotFoundError(f"Color checkpoint not found: {ckpt_color}")
    color_model.load_state_dict(torch.load(ckpt_color, map_location=device)["model"])
    color_model.eval()

    texture_model = TextureSegmentor(dim=18, window_size=4).to(device)
    if not ckpt_texture.exists():
        raise FileNotFoundError(f"Texture checkpoint not found: {ckpt_texture}")
    ckpt_tex = torch.load(ckpt_texture, map_location=device)
    texture_model.load_state_dict(ckpt_tex["model_state_dict"])
    texture_model.eval()

    # ---- 搜集图片 ----
    coco_root_p = Path(coco_root)
    coco_paths = [p for p in coco_root_p.iterdir()
                  if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")]
    wiki_paths = []
    for split in ["train", "test"]:
        sp = Path(wikiart_root) / split
        if not sp.exists():
            continue
        for d in sp.iterdir():
            if not d.is_dir():
                continue
            for p in d.iterdir():
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                    wiki_paths.append(p)

    if not coco_paths or not wiki_paths:
        raise RuntimeError("Empty image folders")

    # 随机选择
    content_path = random.choice(coco_paths)
    style_path = random.choice(wiki_paths)
    print(f"Random content: {content_path.name}")
    print(f"Random style  : {style_path.name}")

    style_img   = transform(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)
    content_img = transform(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)
    content_01  = (content_img + 1.0) / 2.0

    # 获取颜色概率分布
    with torch.no_grad():
        prob, _ = color_model(style_img)
        P = prob[0]

    # 参数配置
    lr_list = [1e-5, 1e-3, 0.1]
    gamma_list = [0.1, 10, 1000]
    base_gamma = 10
    base_lr = 1e-3

    # ---- 第一行：变化学习率（固定 gamma=10）----
    print("\n[Row 1] Searching learning rates...")
    row1_color = []
    for lr in lr_list:
        print(f"  lr={lr} ...")
        c = transfer_color_lut_final(
            content_img, P, style_img,
            bins=33, n_iter=800, lr=lr, gamma=base_gamma
        )
        row1_color.append(c)

    # ---- 第二行：变化 gamma（固定 lr=1e-3）----
    print("\n[Row 2] Searching gammas...")
    row2_color = []
    for gamma in gamma_list:
        if gamma == base_gamma:
            row2_color.append(row1_color[1])
            print(f"  γ={gamma} (reused) ...")
        else:
            print(f"  γ={gamma} ...")
            c = transfer_color_lut_final(
                content_img, P, style_img,
                bins=33, n_iter=800, lr=base_lr, gamma=gamma
            )
            row2_color.append(c)

    # 纹理迁移
    print("\nRunning texture transfer...")
    row1_results = [content_img[0]]
    for i, c in enumerate(row1_color):
        tex, _, _ = transfer_texture_rgb(
            c, style_img, texture_model,
            n_clusters=8, gf_radius=2, gf_eps=0.01,
            use_feather=True, feather_sigma=1.5
        )
        row1_results.append(tex[0])
        print(f"  lr={lr_list[i]} done.")

    row2_results = [style_img[0]]
    for i, c in enumerate(row2_color):
        tex, _, _ = transfer_texture_rgb(
            c, style_img, texture_model,
            n_clusters=8, gf_radius=2, gf_eps=0.01,
            use_feather=True, feather_sigma=1.5
        )
        row2_results.append(tex[0])
        print(f"  γ={gamma_list[i]} done.")

    # 绘制 2×4 画布
    font = get_chinese_font(size=16)

    def to_pil(t):
        t = ((t.detach() + 1.0) / 2.0).clamp(0, 1)
        arr = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)

    imgs = [to_pil(r) for r in row1_results + row2_results]
    labels = [
        "内容图像", "lr=1e-5", "lr=1e-3", "lr=0.1",
        "风格图像", "γ=0.1", "γ=10", "γ=1000",
    ]

    for i in range(len(imgs)):
        arr = np.array(imgs[i])
        arr = draw_label(arr, labels[i], font)
        imgs[i] = Image.fromarray(arr)

    W, H = imgs[0].size
    canvas = Image.new("RGB", (W * 4, H * 2), (30, 30, 30))

    for row in range(2):
        for col in range(4):
            idx = row * 4 + col
            x = col * W
            y = row * H
            canvas.paste(imgs[idx], (x, y))

    rand_name = f"{random.randint(10000000, 99999999)}.png"
    save_path = out_dir / rand_name
    canvas.save(save_path)
    print(f"\nSaved 2×4 param grid to {save_path}")

    kl_before = compute_kl(content_01, P, bins=33)
    print(f"KL before (content vs target): {kl_before:.4f}")


if __name__ == "__main__":
    main()
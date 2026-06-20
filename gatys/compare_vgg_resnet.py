import os
import random
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from tqdm import tqdm


# ==================== 加速 & 配置 ====================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_PAIRS = 100
NUM_STEPS = 500
STYLE_WEIGHT = 1e7
CONTENT_WEIGHT = 1.0
LR = 0.01
IMG_SIZE = 128

COCO_ROOT = r"data\coco\train2017"
WIKIART_ROOT = r"data\wikiart_images"
CKPT_VGG = Path("checkpoints/vgg.pth")
CKPT_RESNET = Path("checkpoints/resnet18.pth")
OUT_DIR = Path("vgg_vs_resnet")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================== 工具函数 ====================
def get_chinese_font(size=20):
    candidates = [
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simkai.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def imagenet_normalize(x):
    x = (x + 1.0) / 2.0
    mean = torch.tensor([0.485, 0.456, 0.406]).to(x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def gram_matrix(tensor):
    b, c, h, w = tensor.shape
    feat = tensor.view(b, c, h * w)
    G = torch.bmm(feat, feat.transpose(1, 2))
    return G / (c * h * w)


def tensor_to_pil(tensor):
    img = tensor.squeeze(0).cpu().clamp(-1, 1)
    img = (img + 1.0) / 2.0
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


def create_comparison_row(images, labels, font):
    n = len(images)
    w, h = images[0].size
    grid = Image.new("RGB", (w * n, h))
    for i, (img, label) in enumerate(zip(images, labels)):
        annotated = img.copy()
        draw = ImageDraw.Draw(annotated)
        draw.text((6, 6), label, fill=(0, 0, 0), font=font)
        draw.text((5, 5), label, fill=(255, 0, 0), font=font)
        grid.paste(annotated, (i * w, 0))
    return grid


# ==================== KL / splat_3d (修正为原始三线性插值版本) ====================
SPLAT_OFFSETS = torch.tensor([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
], dtype=torch.long)


def splat_3d(x, bins):
    """
    三线性插值体素 splatting（与原始代码一致）
    x: [B, 3, H, W] in [0, 1]
    return: [B, bins, bins, bins]
    """
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
        frac[..., 0] * (1 - frac[..., 1]) * (1 - frac[..., 2]),
        (1 - frac[..., 0]) * frac[..., 1] * (1 - frac[..., 2]),
        frac[..., 0] * frac[..., 1] * (1 - frac[..., 2]),
        (1 - frac[..., 0]) * (1 - frac[..., 1]) * frac[..., 2],
        frac[..., 0] * (1 - frac[..., 1]) * frac[..., 2],
        (1 - frac[..., 0]) * frac[..., 1] * frac[..., 2],
        frac[..., 0] * frac[..., 1] * frac[..., 2],
    ], dim=2)
    out = torch.scatter_add(out, 1, flat_all.reshape(B, N * 8), w_all.reshape(B, N * 8))
    return out.view(B, bins, bins, bins) / N


def compute_kl_between(img_a_01, img_b_01, bins=33):
    H_a = splat_3d(img_a_01, bins).squeeze(0) + 1e-8
    H_b = splat_3d(img_b_01, bins).squeeze(0) + 1e-8
    return (H_a * (H_a.log() - H_b.log())).sum().item()


def compute_vgg_style_loss(img, style_grams, vgg_extractor, style_layers):
    with torch.no_grad():
        feats = vgg_extractor(imagenet_normalize(img))
        loss = sum(
            F.mse_loss(gram_matrix(feats[layer]), style_grams[layer])
            for layer in style_layers
        )
    return loss.item()


# ==================== 特征提取器 ====================
class FeatureExtractor(nn.Module):
    def __init__(self, model, target_layers):
        super().__init__()
        self.model = model
        self.target_layers = target_layers
        self.features = {}
        self.hooks = []
        for name, module in self.model.named_modules():
            if name in target_layers:
                hook = module.register_forward_hook(self._save_hook(name))
                self.hooks.append(hook)

    def _save_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def forward(self, x):
        self.features = {}
        self.model(x)
        return self.features

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


# ==================== 加载/保存 Checkpoint ====================
def load_or_save_checkpoint(model_fn, ckpt_path):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        model = model_fn(weights="DEFAULT")
    except TypeError:
        model = model_fn(pretrained=True)

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    else:
        torch.save(model.state_dict(), ckpt_path)
    return model


# ==================== Gatys 风格迁移 ====================
def gatys_transfer(content_img, style_img, extractor, content_layers, style_layers,
                   num_steps, content_weight, style_weight, device, desc="Optim"):
    content_norm = imagenet_normalize(content_img)
    style_norm = imagenet_normalize(style_img)

    with torch.no_grad():
        c_feats = extractor(content_norm)
        s_feats = extractor(style_norm)
        c_targets = {layer: c_feats[layer] for layer in content_layers}
        s_grams = {layer: gram_matrix(s_feats[layer]) for layer in style_layers}

    input_img = content_img.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([input_img], lr=LR)

    for step in tqdm(range(num_steps), desc=desc, leave=False, ncols=80):
        optimizer.zero_grad()
        in_norm = imagenet_normalize(input_img)
        in_feats = extractor(in_norm)

        c_loss = sum(
            F.mse_loss(in_feats[layer], c_targets[layer])
            for layer in content_layers
        )
        s_loss = sum(
            F.mse_loss(gram_matrix(in_feats[layer]), s_grams[layer])
            for layer in style_layers
        )
        total_loss = content_weight * c_loss + style_weight * s_loss
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            input_img.clamp_(-1.0, 1.0)

    return input_img.detach()


# ==================== 主函数 ====================
def main():
    print(f"Device: {DEVICE}")
    font = get_chinese_font(20)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # VGG19
    print("Loading VGG19...")
    vgg_full = load_or_save_checkpoint(models.vgg19, CKPT_VGG)
    vgg_features = vgg_full.features.to(DEVICE).eval()
    for p in vgg_features.parameters():
        p.requires_grad = False

    vgg_content_layers = ["21"]
    vgg_style_layers = ["0", "5", "10", "19", "28"]
    vgg_extractor = FeatureExtractor(
        vgg_features, vgg_content_layers + vgg_style_layers
    ).to(DEVICE).eval()

    # ResNet18
    print("Loading ResNet18...")
    resnet_full = load_or_save_checkpoint(models.resnet18, CKPT_RESNET)
    resnet_full = resnet_full.to(DEVICE).eval()
    for p in resnet_full.parameters():
        p.requires_grad = False

    resnet_style_layers = ["layer1"]
    resnet_content_layers = ["layer3"]
    resnet_extractor = FeatureExtractor(
        resnet_full, resnet_style_layers + resnet_content_layers
    ).to(DEVICE).eval()

    # 搜集图片
    print("Gathering images...")
    coco_root_p = Path(COCO_ROOT)
    coco_paths = [
        p for p in coco_root_p.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    ]

    wiki_paths = []
    for split in ["train", "test"]:
        sp = Path(WIKIART_ROOT) / split
        if not sp.exists():
            continue
        for d in sp.iterdir():
            if not d.is_dir():
                continue
            for p in d.iterdir():
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                    wiki_paths.append(p)

    if not coco_paths or not wiki_paths:
        raise RuntimeError("Empty image folders!")

    print(f"Found {len(coco_paths)} COCO images, {len(wiki_paths)} WikiArt images")

    # CSV 分离
    loss_csv = open(OUT_DIR / "loss.csv", 'w', newline='')
    loss_writer = csv.writer(loss_csv)
    loss_writer.writerow(['content_loss', 'vgg_loss', 'resnet_loss'])

    time_csv = open(OUT_DIR / "time.csv", 'w', newline='')
    time_writer = csv.writer(time_csv)
    time_writer.writerow(['time_vgg', 'time_resnet'])

    kl_csv = open(OUT_DIR / "kl.csv", 'w', newline='')
    kl_writer = csv.writer(kl_csv)
    kl_writer.writerow(['kl_vgg', 'kl_resnet'])

    labels = ["内容图", "风格图", "VGG19", "ResNet18"]

    for pair_idx in range(NUM_PAIRS):
        print(f"\n--- Pair {pair_idx+1}/{NUM_PAIRS} ---")
        style_path = random.choice(wiki_paths)
        content_path = random.choice(coco_paths)
        print(f"  Style : {style_path.name}  ({style_path.parent.name})")
        print(f"  Content: {content_path.name}")

        style_img = transform(Image.open(style_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        content_img = transform(Image.open(content_path).convert("RGB")).unsqueeze(0).to(DEVICE)

        # 预计算 VGG 风格 Gram
        with torch.no_grad():
            s_feats = vgg_extractor(imagenet_normalize(style_img))
            vgg_style_grams = {layer: gram_matrix(s_feats[layer]) for layer in vgg_style_layers}

        # 内容图 VGG style loss
        content_loss = compute_vgg_style_loss(content_img, vgg_style_grams, vgg_extractor, vgg_style_layers)

        # VGG19 Gatys
        t0 = time.time()
        vgg_result = gatys_transfer(
            content_img, style_img, vgg_extractor,
            vgg_content_layers, vgg_style_layers,
            NUM_STEPS, CONTENT_WEIGHT, STYLE_WEIGHT, DEVICE,
            desc=f"VGG {pair_idx+1}/{NUM_PAIRS}"
        )
        time_vgg = time.time() - t0
        vgg_loss = compute_vgg_style_loss(vgg_result, vgg_style_grams, vgg_extractor, vgg_style_layers)

        # ResNet18 Gatys
        t0 = time.time()
        resnet_result = gatys_transfer(
            content_img, style_img, resnet_extractor,
            resnet_content_layers, resnet_style_layers,
            NUM_STEPS, CONTENT_WEIGHT, STYLE_WEIGHT, DEVICE,
            desc=f"ResNet {pair_idx+1}/{NUM_PAIRS}"
        )
        time_resnet = time.time() - t0
        resnet_loss = compute_vgg_style_loss(resnet_result, vgg_style_grams, vgg_extractor, vgg_style_layers)

        # 计算 KL (结果 vs 内容图, bins=33)
        content_01 = (content_img + 1.0) / 2.0
        vgg_01 = (vgg_result + 1.0) / 2.0
        resnet_01 = (resnet_result + 1.0) / 2.0

        kl_vgg = compute_kl_between(vgg_01, content_01, bins=33)
        kl_resnet = compute_kl_between(resnet_01, content_01, bins=33)

        # 保存图片
        content_pil = tensor_to_pil(content_img)
        style_pil = tensor_to_pil(style_img)
        vgg_pil = tensor_to_pil(vgg_result)
        resnet_pil = tensor_to_pil(resnet_result)

        grid = create_comparison_row(
            [content_pil, style_pil, vgg_pil, resnet_pil],
            labels,
            font
        )
        out_path = OUT_DIR / f"transfer_compare_{pair_idx+1:03d}.png"
        grid.save(out_path)

        loss_writer.writerow([f"{content_loss:.4g}", f"{vgg_loss:.4g}", f"{resnet_loss:.4g}"])
        time_writer.writerow([f"{time_vgg:.4g}", f"{time_resnet:.4g}"])
        kl_writer.writerow([f"{kl_vgg:.4g}", f"{kl_resnet:.4g}"])

        print(f"  loss: c={content_loss:.4g} v={vgg_loss:.4g} r={resnet_loss:.4g} | "
            f"time: v={time_vgg:.4g}s r={time_resnet:.4g}s | "
            f"kl: v={kl_vgg:.4g} r={kl_resnet:.4g}")
        
    loss_csv.close()
    time_csv.close()
    kl_csv.close()
    print(f"\nDone. All files saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
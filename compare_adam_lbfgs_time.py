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
from PIL import Image
from pathlib import Path
from tqdm import tqdm


# ==================== 加速 & 配置 ====================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_PAIRS = 50
NUM_STEPS = 500
CONTENT_WEIGHT = 1.0
BETA = 1e7
GAMMA = 1e-4
LR_ADAM = 0.01
LR_LBFGS = 1.0
IMG_SIZE = 128

COCO_ROOT = r"data\coco\train2017"
WIKIART_ROOT = r"data\wikiart_images"
CKPT_VGG = Path("checkpoints/vgg.pth")
OUT_DIR = Path("vgg_vs_resnet")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================== 工具函数 ====================
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


def tv_loss(img):
    diff_h = img[:, :, 1:, :] - img[:, :, :-1, :]
    diff_w = img[:, :, :, 1:] - img[:, :, :, :-1]
    return (diff_h ** 2).mean() + (diff_w ** 2).mean()


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


# ==================== 加载 Checkpoint ====================
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


# ==================== Gatys Adam ====================
def gatys_adam(content_img, style_img, extractor, content_layers, style_layers,
               num_steps, content_weight, style_weight, tv_weight, device):
    content_norm = imagenet_normalize(content_img)
    style_norm = imagenet_normalize(style_img)

    with torch.no_grad():
        c_feats = extractor(content_norm)
        s_feats = extractor(style_norm)
        c_targets = {layer: c_feats[layer] for layer in content_layers}
        s_grams = {layer: gram_matrix(s_feats[layer]) for layer in style_layers}

    input_img = content_img.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([input_img], lr=LR_ADAM)

    for _ in range(num_steps):
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
        t_loss = tv_loss(input_img)

        total_loss = content_weight * c_loss + style_weight * s_loss + tv_weight * t_loss
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            input_img.clamp_(-1.0, 1.0)

    return input_img.detach()


# ==================== Gatys L-BFGS ====================
def gatys_lbfgs(content_img, style_img, extractor, content_layers, style_layers,
                num_steps, content_weight, style_weight, tv_weight, device):
    content_norm = imagenet_normalize(content_img)
    style_norm = imagenet_normalize(style_img)

    with torch.no_grad():
        c_feats = extractor(content_norm)
        s_feats = extractor(style_norm)
        c_targets = {layer: c_feats[layer] for layer in content_layers}
        s_grams = {layer: gram_matrix(s_feats[layer]) for layer in style_layers}

    input_img = content_img.clone().detach().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [input_img],
        lr=LR_LBFGS,
        max_iter=num_steps,
        history_size=10,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-5,
        tolerance_change=1e-9,
    )

    def closure():
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
        t_loss = tv_loss(input_img)

        total_loss = content_weight * c_loss + style_weight * s_loss + tv_weight * t_loss
        total_loss.backward()

        with torch.no_grad():
            input_img.clamp_(-1.0, 1.0)

        return total_loss

    optimizer.step(closure)
    return input_img.detach()


# ==================== 主函数 ====================
def main():
    print(f"Device: {DEVICE}")

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

    # 固定 50 对图片
    random.seed(42)
    selected_pairs = []
    for _ in range(NUM_PAIRS):
        selected_pairs.append((random.choice(coco_paths), random.choice(wiki_paths)))

    # 预加载 tensor
    content_imgs = []
    style_imgs = []
    for c_path, s_path in selected_pairs:
        c = transform(Image.open(c_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        s = transform(Image.open(s_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        content_imgs.append(c)
        style_imgs.append(s)

    # CSV
    csv_path = OUT_DIR / "optimizer_time.csv"
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['idx', 'content_name', 'style_name', 'time_adam', 'time_lbfgs'])

    times_adam = []
    times_lbfgs = []

    for pair_idx in tqdm(range(NUM_PAIRS), desc="Benchmarking", position=0):
        c_path, s_path = selected_pairs[pair_idx]

        # Adam
        torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
        t0 = time.perf_counter()
        gatys_adam(
            content_imgs[pair_idx], style_imgs[pair_idx],
            vgg_extractor, vgg_content_layers, vgg_style_layers,
            NUM_STEPS, CONTENT_WEIGHT, BETA, GAMMA, DEVICE
        )
        torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
        t_adam = time.perf_counter() - t0
        times_adam.append(t_adam)

        # L-BFGS
        torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
        t0 = time.perf_counter()
        gatys_lbfgs(
            content_imgs[pair_idx], style_imgs[pair_idx],
            vgg_extractor, vgg_content_layers, vgg_style_layers,
            NUM_STEPS, CONTENT_WEIGHT, BETA, GAMMA, DEVICE
        )
        torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
        t_lbfgs = time.perf_counter() - t0
        times_lbfgs.append(t_lbfgs)

        csv_writer.writerow([
            pair_idx,
            Path(c_path).name,
            Path(s_path).name,
            f"{t_adam:.4g}",
            f"{t_lbfgs:.4g}",
        ])

    csv_file.close()

    # 汇总
    print(f"\n{'='*50}")
    print(f"Optimizer time comparison ({NUM_PAIRS} pairs, {NUM_STEPS} steps)")
    print(f"Fixed: β={BETA:.4g}, γ={GAMMA:.4g}")
    print(f"{'='*50}")
    print(f"  Adam   : {np.mean(times_adam):.4g} ± {np.std(times_adam):.4g} s")
    print(f"  L-BFGS : {np.mean(times_lbfgs):.4g} ± {np.std(times_lbfgs):.4g} s")
    print(f"  Ratio  : {np.mean(times_lbfgs)/np.mean(times_adam):.4g}×")
    print(f"{'='*50}")
    print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()
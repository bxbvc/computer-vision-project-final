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
NUM_PAIRS = 10
NUM_STEPS = 500          # 对应 LBFGS 的 max_iter
CONTENT_WEIGHT = 1.0
LR = 1.0                 # LBFGS 通常设为 1.0
IMG_SIZE = 128

COCO_ROOT = r"data\coco\train2017"
WIKIART_ROOT = r"data\wikiart_images"
CKPT_VGG = Path("checkpoints/vgg.pth")
OUT_DIR = Path("vgg_vs_resnet_lbfgs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 10×10 超参数网格
BETAS = np.logspace(0, 10, 10)
GAMMAS = np.logspace(0, 10, 10)

# 可视化 4×4 子网格
VIZ_BETA_IDX = [0, 3, 6, 9]
VIZ_GAMMA_IDX = [0, 3, 6, 9]


# ==================== 工具函数 ====================
def get_chinese_font(size=12):
    candidates = [
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
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


def tv_loss(img):
    """全变分损失 (L2)"""
    diff_h = img[:, :, 1:, :] - img[:, :, :-1, :]
    diff_w = img[:, :, :, 1:] - img[:, :, :, :-1]
    return (diff_h ** 2).mean() + (diff_w ** 2).mean()


def tensor_to_pil(tensor):
    img = tensor.squeeze(0).cpu().clamp(-1, 1)
    img = (img + 1.0) / 2.0
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


def create_grid_2x9(images, labels, font):
    assert len(images) == 18 and len(labels) == 18
    w, h = images[0].size
    grid = Image.new("RGB", (w * 9, h * 2))
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // 9
        col = idx % 9
        annotated = img.copy()
        draw = ImageDraw.Draw(annotated)
        draw.text((4, 4), label, fill=(0, 0, 0), font=font)
        draw.text((3, 3), label, fill=(255, 0, 0), font=font)
        grid.paste(annotated, (col * w, row * h))
    return grid


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


# ==================== Gatys 风格迁移 (L-BFGS 拟牛顿法) ====================
def gatys_transfer_lbfgs(content_img, style_img, extractor, content_layers, style_layers,
                         num_steps, content_weight, style_weight, tv_weight, device, desc="LBFGS"):
    content_norm = imagenet_normalize(content_img)
    style_norm = imagenet_normalize(style_img)

    with torch.no_grad():
        c_feats = extractor(content_norm)
        s_feats = extractor(style_norm)
        c_targets = {layer: c_feats[layer] for layer in content_layers}
        s_grams = {layer: gram_matrix(s_feats[layer]) for layer in style_layers}

    input_img = content_img.clone().detach().requires_grad_(True)

    # L-BFGS 优化器（拟牛顿法）
    optimizer = torch.optim.LBFGS(
        [input_img],
        lr=LR,
        max_iter=num_steps,
        history_size=10,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-5,
        tolerance_change=1e-9,
    )

    pbar = tqdm(total=num_steps, desc=desc, leave=False, ncols=80)
    n_evals = 0

    def closure():
        nonlocal n_evals
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

        # 进度更新（按 closure 调用次数估算）
        n_evals += 1
        if n_evals <= num_steps:
            pbar.update(1)

        # 裁剪到合法范围
        with torch.no_grad():
            input_img.clamp_(-1.0, 1.0)

        return total_loss

    # 执行拟牛顿优化
    optimizer.step(closure)
    pbar.close()

    # 最终统计各项原始 loss（不加权）
    with torch.no_grad():
        in_norm = imagenet_normalize(input_img)
        in_feats = extractor(in_norm)
        c_loss = sum(F.mse_loss(in_feats[layer], c_targets[layer]) for layer in content_layers)
        s_loss = sum(F.mse_loss(gram_matrix(in_feats[layer]), s_grams[layer]) for layer in style_layers)
        t_loss = tv_loss(input_img)

    return input_img.detach(), c_loss.item(), s_loss.item(), t_loss.item()


# ==================== 主函数 ====================
def main():
    print(f"Device: {DEVICE}")
    font = get_chinese_font(12)

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

    # 固定 30 对图片
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
    csv_path = OUT_DIR / "vgg_hyperparam_lbfgs.csv"
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'beta', 'gamma',
        'content_loss_mean', 'content_loss_std',
        'style_loss_mean', 'style_loss_std',
        'tv_loss_mean', 'tv_loss_std'
    ])

    # 可视化缓存
    viz_cache = {}

    total_iters = len(BETAS) * len(GAMMAS) * NUM_PAIRS
    print(f"\nTotal optimization runs: {total_iters} ({len(BETAS)}×{len(GAMMAS)}×{NUM_PAIRS})")
    print("Optimizer: L-BFGS (Quasi-Newton)")

    outer_pbar = tqdm(total=len(BETAS)*len(GAMMAS), desc="Hyperparam grid", position=0)

    for b_idx, beta in enumerate(BETAS):
        for g_idx, gamma in enumerate(GAMMAS):
            c_losses = []
            s_losses = []
            t_losses = []

            for pair_idx in range(NUM_PAIRS):
                desc = f"LBFGS β={beta:.1e} γ={gamma:.1e} [{pair_idx+1}/{NUM_PAIRS}]"
                result, c_l, s_l, t_l = gatys_transfer_lbfgs(
                    content_imgs[pair_idx], style_imgs[pair_idx],
                    vgg_extractor, vgg_content_layers, vgg_style_layers,
                    NUM_STEPS, CONTENT_WEIGHT, beta, gamma, DEVICE,
                    desc=desc
                )
                c_losses.append(c_l)
                s_losses.append(s_l)
                t_losses.append(t_l)

                # 缓存可视化图片
                if b_idx in VIZ_BETA_IDX and g_idx in VIZ_GAMMA_IDX:
                    viz_cache[(pair_idx, b_idx, g_idx)] = tensor_to_pil(result)

            # 统计并写入 CSV
            csv_writer.writerow([
                f"{beta:.6e}", f"{gamma:.6e}",
                f"{np.mean(c_losses):.4g}", f"{np.std(c_losses):.4g}",
                f"{np.mean(s_losses):.4g}", f"{np.std(s_losses):.4g}",
                f"{np.mean(t_losses):.4g}", f"{np.std(t_losses):.4g}",
            ])
            outer_pbar.update(1)

    outer_pbar.close()
    csv_file.close()

    # ==================== 生成 2×9 可视化大图 ====================
    print("\nGenerating 2×9 visualization grids...")
    viz_dir = OUT_DIR / "viz_grids"
    viz_dir.mkdir(exist_ok=True)

    for pair_idx in range(NUM_PAIRS):
        images = []
        labels = []

        # 第一行前两张：内容图 + 风格图
        content_pil = tensor_to_pil(content_imgs[pair_idx])
        style_pil = tensor_to_pil(style_imgs[pair_idx])
        images.extend([content_pil, style_pil])
        labels.extend(["内容图", "风格图"])

        # 收集 16 张可视化结果
        viz_imgs = []
        viz_labs = []
        for b_idx in VIZ_BETA_IDX:
            for g_idx in VIZ_GAMMA_IDX:
                key = (pair_idx, b_idx, g_idx)
                if key in viz_cache:
                    viz_imgs.append(viz_cache[key])
                    beta = BETAS[b_idx]
                    gamma = GAMMAS[g_idx]
                    viz_labs.append(f"β={beta:.0e}\nγ={gamma:.0e}")

        # 补足 16 张
        while len(viz_imgs) < 16:
            viz_imgs.append(Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128)))
            viz_labs.append("N/A")

        # 前 7 张放入第一行（凑满 9 张）
        images.extend(viz_imgs[:7])
        labels.extend(viz_labs[:7])

        # 后 9 张放入第二行
        images.extend(viz_imgs[7:16])
        labels.extend(viz_labs[7:16])

        assert len(images) == 18 and len(labels) == 18
        grid = create_grid_2x9(images, labels, font)
        grid.save(viz_dir / f"grid_pair_{pair_idx+1:03d}.png")

    print(f"\nDone. CSV -> {csv_path}")
    print(f"Viz grids -> {viz_dir}")


if __name__ == "__main__":
    main()
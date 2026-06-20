import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm


# ==================== 配置 ====================
torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONTENT_PATH = Path("./content.png")
STYLE_PATH = Path("./style.png")
OUT_PDF = Path("style_transfer_result.pdf")

BETA = 1e7
GAMMA = 0.0
CONTENT_WEIGHT = 1.0
NUM_STEPS = 1000
IMG_SIZE = 512
LR = 1.0
SAVE_EVERY = 100

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 工具函数 ====================
def get_font(size=28):
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
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
    diff_h = img[:, :, 1:, :] - img[:, :, :-1, :]
    diff_w = img[:, :, :, 1:] - img[:, :, :, :-1]
    return (diff_h ** 2).mean() + (diff_w ** 2).mean()


def tensor_to_pil(tensor):
    img = tensor.squeeze(0).detach().cpu().clamp(-1, 1)
    img = (img + 1.0) / 2.0
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


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


# ==================== 风格迁移核心 ====================
def gatys_transfer_with_history(content_img, style_img, extractor,
                                content_layers, style_layers,
                                num_steps, content_weight, style_weight, tv_weight,
                                device, save_every=100):
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
        lr=LR,
        max_iter=num_steps,
        history_size=10,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-5,
        tolerance_change=1e-9,
    )

    history_pils = []
    history_steps = []
    all_steps = []
    all_c_losses = []
    all_s_losses = []
    all_t_losses = []

    step_counter = [0]
    font = get_font(size=28)

    pbar = tqdm(total=num_steps, desc="L-BFGS 优化", unit="step", ncols=80)

    def closure():
        optimizer.zero_grad()
        in_norm = imagenet_normalize(input_img)
        in_feats = extractor(in_norm)

        c_loss = sum(F.mse_loss(in_feats[layer], c_targets[layer]) for layer in content_layers)
        s_loss = sum(F.mse_loss(gram_matrix(in_feats[layer]), s_grams[layer]) for layer in style_layers)
        t_loss = tv_loss(input_img)

        total_loss = content_weight * c_loss + style_weight * s_loss + tv_weight * t_loss
        total_loss.backward()

        step = step_counter[0]
        all_steps.append(step)
        all_c_losses.append(c_loss.item())
        all_s_losses.append(s_loss.item())
        all_t_losses.append(t_loss.item())

        if step % save_every == 0:
            pil_img = tensor_to_pil(input_img)
            draw = ImageDraw.Draw(pil_img)
            label = str(step)
            draw.text((6, 6), label, fill=(0, 0, 0), font=font)
            draw.text((5, 5), label, fill=(255, 0, 0), font=font)
            history_pils.append(pil_img)
            history_steps.append(step)

        with torch.no_grad():
            input_img.clamp_(-1.0, 1.0)

        step_counter[0] += 1
        if step < num_steps:
            pbar.update(1)

        return total_loss

    optimizer.step(closure)
    pbar.close()

    if not history_steps or history_steps[-1] != num_steps:
        pil_img = tensor_to_pil(input_img)
        draw = ImageDraw.Draw(pil_img)
        label = str(num_steps)
        draw.text((6, 6), label, fill=(0, 0, 0), font=font)
        draw.text((5, 5), label, fill=(255, 0, 0), font=font)
        history_pils.append(pil_img)
        history_steps.append(num_steps)

        with torch.no_grad():
            in_norm = imagenet_normalize(input_img)
            in_feats = extractor(in_norm)
            c_loss = sum(F.mse_loss(in_feats[layer], c_targets[layer]) for layer in content_layers)
            s_loss = sum(F.mse_loss(gram_matrix(in_feats[layer]), s_grams[layer]) for layer in style_layers)
            t_loss = tv_loss(input_img)

        if not all_steps or all_steps[-1] != num_steps:
            all_steps.append(num_steps)
            all_c_losses.append(c_loss.item())
            all_s_losses.append(s_loss.item())
            all_t_losses.append(t_loss.item())

    return input_img.detach(), history_pils, history_steps, all_steps, all_c_losses, all_s_losses, all_t_losses


# ==================== 可视化 ====================
def get_curve_image(all_steps, all_c_losses, all_s_losses, all_t_losses, target_width, dpi=150):
    """
    三张子图各自为正方形（ax.set_box_aspect(1)），横向均匀分布。
    曲线图总宽度与长图对齐，高度由正方形边长+标签空间决定。
    """
    colors = ['#2878B5', '#C82423', '#FF8884']

    fig_w_inch = target_width / dpi
    # 每个子图正方形边长约 fig_w/3，额外留出 2.5 英寸给标题/轴标签
    fig_h_inch = fig_w_inch / 3 + 2.5

    fig, axes = plt.subplots(1, 3, figsize=(fig_w_inch, fig_h_inch), dpi=dpi, constrained_layout=True)

    titles = ['内容损失 (Content Loss)', '风格损失 (Style Loss)', '全变分损失 (TV Loss)']
    data_list = [all_c_losses, all_s_losses, all_t_losses]

    for ax, title, y_data, color in zip(axes, titles, data_list, colors):
        ax.plot(all_steps, y_data, color=color, linewidth=8)
        ax.set_title(title, fontsize=54, fontweight='bold')
        ax.set_xlabel('迭代步数', fontsize=32)
        ax.set_ylabel('损失值', fontsize=32)
        ax.tick_params(labelsize=28)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_box_aspect(1)  # 关键：每个子图显示区域为正方形

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def build_long_image(images):
    n = len(images)
    w, h = images[0].size
    total_w = w * n
    long_img = Image.new("RGB", (total_w, h))
    for i, img in enumerate(images):
        long_img.paste(img, (i * w, 0))
    return long_img


# ==================== 主函数 ====================
def main():
    print(f"Device: {DEVICE}")
    if not CONTENT_PATH.exists() or not STYLE_PATH.exists():
        raise FileNotFoundError(f"请确保 {CONTENT_PATH} 和 {STYLE_PATH} 存在")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    content_img = transform(Image.open(CONTENT_PATH).convert("RGB")).unsqueeze(0).to(DEVICE)
    style_img = transform(Image.open(STYLE_PATH).convert("RGB")).unsqueeze(0).to(DEVICE)

    print("Loading VGG19...")
    vgg_full = models.vgg19(weights="DEFAULT")
    vgg_features = vgg_full.features.to(DEVICE).eval()
    for p in vgg_features.parameters():
        p.requires_grad = False

    content_layers = ["21"]
    style_layers = ["0", "5", "10", "19", "28"]
    extractor = FeatureExtractor(vgg_features, content_layers + style_layers).to(DEVICE).eval()

    print(f"Running L-BFGS optimization: {NUM_STEPS} steps, save every {SAVE_EVERY}...")
    final_img, history_pils, history_steps, all_steps, all_c, all_s, all_t = gatys_transfer_with_history(
        content_img, style_img, extractor,
        content_layers, style_layers,
        NUM_STEPS, CONTENT_WEIGHT, BETA, GAMMA,
        DEVICE, save_every=SAVE_EVERY
    )

    print(f"Recorded {len(history_steps)} snapshots at steps: {history_steps}")
    print(f"Total loss evaluations: {len(all_steps)}")

    # 风格图放在长图最左边
    style_pil = tensor_to_pil(style_img)
    images_for_long = [style_pil] + history_pils
    long_img = build_long_image(images_for_long)
    long_w, long_h = long_img.size

    # 曲线图：宽度与长图对齐，每张子图为正方形
    curve_img = get_curve_image(all_steps, all_c, all_s, all_t, target_width=long_w, dpi=150)
    curve_w, curve_h = curve_img.size

    margin = 30
    page_w = max(curve_w, long_w)
    page_h = curve_h + long_h + margin

    page = Image.new("RGB", (page_w, page_h), (255, 255, 255))
    page.paste(curve_img, ((page_w - curve_w) // 2, 0))
    page.paste(long_img, ((page_w - long_w) // 2, curve_h + margin))

    dpi = 150
    fig = plt.figure(figsize=(page_w / dpi, page_h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(page)
    ax.axis('off')
    fig.savefig(OUT_PDF, format='pdf', dpi=dpi)
    plt.close(fig)

    extractor.remove_hooks()
    print(f"\nDone. PDF saved to: {OUT_PDF.resolve()}")


if __name__ == "__main__":
    main()
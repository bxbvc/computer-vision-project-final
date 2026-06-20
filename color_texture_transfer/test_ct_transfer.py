import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms, models
from tqdm import tqdm


# ========================== 常量配置区（直接修改，拒绝命令行） ==========================
STYLE_LAYERS   = [1, 2, 3]       # 风格层列表 (1~5)
CONTENT_LAYERS = [2, 3, 4]       # 内容层列表 (1~5)

COCO_ROOT     = r"data\coco\train2017"
WIKIART_ROOT  = r"data\wikiart_images"
CKPT_PATH     = Path("checkpoints/ct.pth")
VGG_CKPT_PATH = Path(r"D:\WorkSpace\class-pjs\computer-vision\checkpoints\vgg.pth")

MAX_STEPS      = 500
SAVE_EVERY     = 50
LR             = 0.01
CONTENT_WEIGHT = 1.0
STYLE_WEIGHT   = 1e7

OUTPUT_DIR = Path("rubbish")
IMG_SIZE   = 128
# ===================================================================================


# ========================== 模型定义 ==========================

class AttentionLayer(nn.Module):
    def __init__(self, in_channels: int, transform: bool):
        super().__init__()
        self.transform = transform

        if transform:
            out_channels = in_channels * 2
            stride = 2
        else:
            out_channels = in_channels
            stride = 1

        self.gelu = nn.GELU()

        self.q_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.q_bn1 = nn.BatchNorm2d(out_channels)
        self.q_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.q_bn2 = nn.BatchNorm2d(out_channels)

        self.v_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.v_bn1 = nn.BatchNorm2d(out_channels)
        self.v_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.v_bn2 = nn.BatchNorm2d(out_channels)

        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_down = nn.BatchNorm2d(out_channels)

    def _window_attn(self, Q, V, h, w, token_len, shift):
        B = Q.size(0)
        num_win_h = h // 4
        num_win_w = w // 4

        if shift > 0:
            Q = torch.roll(Q, shifts=(shift, shift), dims=(2, 3))
            V = torch.roll(V, shifts=(shift, shift), dims=(2, 3))

        def to_windows(t):
            t = t.view(B, token_len, num_win_h, 4, num_win_w, 4)
            t = t.permute(0, 2, 4, 3, 5, 1)
            t = t.reshape(B, num_win_h * num_win_w, 16, token_len)
            return t

        Q_win = to_windows(Q)
        V_win = to_windows(V)

        attn = torch.matmul(Q_win, Q_win.transpose(-2, -1)) / math.sqrt(token_len)
        A = F.softmax(attn, dim=-1)
        out = torch.matmul(A, V_win)

        out = out.view(B, num_win_h, num_win_w, 4, 4, token_len)
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(B, token_len, h, w)

        if shift > 0:
            out = torch.roll(out, shifts=(-shift, -shift), dims=(2, 3))

        return out

    def forward(self, x, return_qv=False):
        B, C, H, W = x.shape

        if self.transform:
            token_len = C * 2
            h = H // 2
            w = W // 2
        else:
            token_len = C
            h = H
            w = W

        assert h % 4 == 0 and w % 4 == 0, f"特征图尺寸 ({h},{w}) 必须能被 4 整除"

        Q = self.gelu(self.q_bn1(self.q_conv1(x)))
        Q = self.q_bn2(self.q_conv2(Q))

        V = self.gelu(self.v_bn1(self.v_conv1(x)))
        V = self.v_bn2(self.v_conv2(V))

        x = self.down_conv(x)

        out1 = self._window_attn(Q, V, h, w, token_len, shift=0)
        out2 = self._window_attn(Q, V, h, w, token_len, shift=2)

        out = x + out1 + out2
        out = self.gelu(self.bn_down(out))

        if return_qv:
            return out, Q, V
        return out

    def get_window_pooled_features(self, x):
        """
        沿窗口维度做平均池化，压缩成 (B, 16, token_len) 的二维特征。
        16 是窗口内固定的 4×4 空间位置。
        """
        B, C, H, W = x.shape

        if self.transform:
            token_len = C * 2
            h = H // 2
            w = W // 2
        else:
            token_len = C
            h = H
            w = W

        assert h % 4 == 0 and w % 4 == 0

        # 计算 Q, V（与 forward 一致）
        Q = self.gelu(self.q_bn1(self.q_conv1(x)))
        Q = self.q_bn2(self.q_conv2(Q))
        V = self.gelu(self.v_bn1(self.v_conv1(x)))
        V = self.v_bn2(self.v_conv2(V))

        num_win_h = h // 4
        num_win_w = w // 4

        def to_windows(t):
            t = t.view(B, token_len, num_win_h, 4, num_win_w, 4)
            t = t.permute(0, 2, 4, 3, 5, 1)
            t = t.reshape(B, num_win_h * num_win_w, 16, token_len)
            return t

        Q_win = to_windows(Q)  # (B, num_win, 16, token_len)
        V_win = to_windows(V)  # (B, num_win, 16, token_len)

        # 沿窗口维度平均池化 -> (B, 16, token_len)
        Q_pooled = Q_win.mean(dim=1)
        V_pooled = V_win.mean(dim=1)

        return Q_pooled, V_pooled


class FinalAttentionLayer(nn.Module):
    def __init__(self, in_channels, num_classes=100):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits


class AttentionResNet(nn.Module):
    def __init__(self, channels=64, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.layer1 = AttentionLayer(channels, True)
        self.layer2 = AttentionLayer(channels * 2, False)
        self.layer3 = AttentionLayer(channels * 2, True)
        self.layer4 = AttentionLayer(channels * 4, False)
        self.layer5 = AttentionLayer(channels * 4, True)

        self.final = FinalAttentionLayer(channels * 8, num_classes)

    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        logits = self.final(x)
        return logits

    def extract_for_transfer(self, x, style_layers, content_layers):
        """
        按序前向传播，在指定层提取窗口平均池化后的 Q（风格）和 V（内容）。
        style_layers / content_layers 为列表，元素范围 1~5。
        返回: {'style': {layer_idx: Q_pooled}, 'content': {layer_idx: V_pooled}}
        Q_pooled / V_pooled 形状: (B, 16, token_len)
        """
        style_set = set(style_layers)
        content_set = set(content_layers)

        features = {'style': {}, 'content': {}}
        x = F.gelu(self.bn1(self.conv1(x)))

        layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]
        for i, layer in enumerate(layers, 1):
            need_style = i in style_set
            need_content = i in content_set

            if need_style or need_content:
                Q_pooled, V_pooled = layer.get_window_pooled_features(x)
                if need_style:
                    features['style'][i] = Q_pooled
                if need_content:
                    features['content'][i] = V_pooled

            x = layer(x)

        return features


# ========================== VGG 风格迁移参考 ==========================

class VGGStyleTransfer(nn.Module):
    def __init__(self, checkpoint_path, device):
        super().__init__()
        vgg = models.vgg19(pretrained=False)
        state_dict = torch.load(checkpoint_path, map_location=device)
        vgg.load_state_dict(state_dict)
        self.features = vgg.features.to(device).eval()
        for param in self.features.parameters():
            param.requires_grad = False

        self.style_layers = [1, 6, 11, 20, 29]
        self.content_layers = [22]

    def extract_features(self, x):
        style_features = []
        content_features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.style_layers:
                style_features.append(x)
            if i in self.content_layers:
                content_features.append(x)
        return {'style': style_features, 'content': content_features}


# ========================== 工具函数 ==========================

def gram_matrix_vgg(x):
    b, c, h, w = x.shape
    x = x.view(b, c, h * w)
    gram = torch.matmul(x, x.transpose(1, 2)) / (c * h * w)
    return gram


def save_tensor_image(tensor, path):
    img = tensor.squeeze(0).detach().cpu()
    img = (img + 1.0) / 2.0
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).astype("uint8")
    Image.fromarray(img).save(path)


def save_vgg_image(tensor, path):
    img = tensor.squeeze(0).detach().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype("uint8")
    Image.fromarray(img).save(path)


def make_square_grid(image_paths, output_path, labels=None, font_size=20):
    if not image_paths:
        return

    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    w, h = imgs[0].size
    imgs = [img.resize((w, h)) for img in imgs]

    n = len(imgs)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    label_height = font_size + 10 if labels else 0
    grid = Image.new("RGB", (cols * w, rows * (h + label_height)), (255, 255, 255))

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(grid)
    for idx, img in enumerate(imgs):
        row, col = divmod(idx, cols)
        x, y = col * w, row * (h + label_height)
        grid.paste(img, (x, y))
        if labels and idx < len(labels):
            text = str(labels[idx])
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            draw.text((x + (w - text_w) // 2, y + h + 5), text, fill=(0, 0, 0), font=font)

    grid.save(output_path)
    print(f"\nGrid saved: {output_path}  ({cols}×{rows}, total {n} images)")


# ========================== 主函数 ==========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 搜集图片 ----
    coco_root_p = Path(COCO_ROOT)
    coco_paths = [p for p in coco_root_p.iterdir()
                  if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")]

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
        raise RuntimeError("Empty image folders")

    style_path   = random.choice(wiki_paths)
    content_path = random.choice(coco_paths)

    print(f"\nStyle : {style_path.name}  ({style_path.parent.name})")
    print(f"Content: {content_path.name}")

    # ---- 两套预处理 ----
    transform_attn = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    transform_vgg = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    style_img_attn   = transform_attn(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)
    content_img_attn = transform_attn(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)

    style_img_vgg   = transform_vgg(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)
    content_img_vgg = transform_vgg(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)

    save_tensor_image(style_img_attn, OUTPUT_DIR / "style.png")
    save_tensor_image(content_img_attn, OUTPUT_DIR / "content.png")
    print(f"Saved original style.png & content.png -> {OUTPUT_DIR}")

    # ============================================================
    # 1) Attention-based（窗口平均池化，直接 MSE）
    # ============================================================
    attn_dir = OUTPUT_DIR / "attention"
    attn_dir.mkdir(exist_ok=True)

    model = AttentionResNet(channels=48, num_classes=100).to(device)
    if CKPT_PATH.exists():
        model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        print(f"Loaded Attention checkpoint: {CKPT_PATH}")
    else:
        print(f"Warning: Attention checkpoint not found, using random weights!")
    model.eval()

    with torch.no_grad():
        style_feat   = model.extract_for_transfer(style_img_attn, STYLE_LAYERS, CONTENT_LAYERS)
        content_feat = model.extract_for_transfer(content_img_attn, STYLE_LAYERS, CONTENT_LAYERS)

    input_img = content_img_attn.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([input_img], lr=LR)

    print(f"\n>>> [Attention] Style Transfer Begin")
    print(f"    Style layers  : {STYLE_LAYERS}")
    print(f"    Content layers: {CONTENT_LAYERS}")
    print(f"    Steps         : {MAX_STEPS} | LR: {LR}")

    attn_paths = []
    for step in tqdm(range(MAX_STEPS + 1), desc="Attention"):
        optimizer.zero_grad()

        feat = model.extract_for_transfer(input_img, STYLE_LAYERS, CONTENT_LAYERS)

        # 风格损失：多层窗口平均池化后的 Q 直接 MSE
        loss_style = 0
        for layer_idx in STYLE_LAYERS:
            loss_style += F.mse_loss(feat['style'][layer_idx], style_feat['style'][layer_idx])

        # 内容损失：多层窗口平均池化后的 V 直接 MSE
        loss_content = 0
        for layer_idx in CONTENT_LAYERS:
            loss_content += F.mse_loss(feat['content'][layer_idx], content_feat['content'][layer_idx])

        loss = STYLE_WEIGHT * loss_style + CONTENT_WEIGHT * loss_content
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            input_img.clamp_(-1, 1)

        if step % SAVE_EVERY == 0 or step == MAX_STEPS:
            save_path = attn_dir / f"step{step:04d}.png"
            save_tensor_image(input_img, save_path)
            attn_paths.append(save_path)
            tqdm.write(f"[Attn] Step {step:04d} | Style: {loss_style.item():.4f} | "
                       f"Content: {loss_content.item():.4f} | Total: {loss.item():.4f}")

    # ============================================================
    # 2) VGG19 标准风格迁移（相同 Adam 参数）
    # ============================================================
    vgg_dir = OUTPUT_DIR / "vgg"
    vgg_dir.mkdir(exist_ok=True)

    vgg_model = VGGStyleTransfer(VGG_CKPT_PATH, device)

    vgg_style_weights = [1.0, 0.5, 0.25, 0.125, 0.0625]

    with torch.no_grad():
        vgg_style_feat   = vgg_model.extract_features(style_img_vgg)
        vgg_content_feat = vgg_model.extract_features(content_img_vgg)
        target_grams = [gram_matrix_vgg(f) for f in vgg_style_feat['style']]

    input_img_vgg = content_img_vgg.clone().requires_grad_(True)
    optimizer_vgg = torch.optim.Adam([input_img_vgg], lr=LR)

    print(f"\n>>> [VGG19] Style Transfer Begin")
    print(f"    Steps: {MAX_STEPS} | LR: {LR}")

    vgg_paths = []
    for step in tqdm(range(MAX_STEPS + 1), desc="VGG19"):
        optimizer_vgg.zero_grad()

        feat = vgg_model.extract_features(input_img_vgg)

        loss_style = 0
        for i, (f, target, w) in enumerate(zip(feat['style'], target_grams, vgg_style_weights)):
            loss_style += w * F.mse_loss(gram_matrix_vgg(f), target)

        loss_content = F.mse_loss(feat['content'][0], vgg_content_feat['content'][0])

        loss = STYLE_WEIGHT * loss_style + CONTENT_WEIGHT * loss_content
        loss.backward()
        optimizer_vgg.step()

        if step % SAVE_EVERY == 0 or step == MAX_STEPS:
            save_path = vgg_dir / f"step{step:04d}.png"
            save_vgg_image(input_img_vgg, save_path)
            vgg_paths.append(save_path)
            tqdm.write(f"[VGG]  Step {step:04d} | Style: {loss_style.item():.4f} | "
                       f"Content: {loss_content.item():.4f} | Total: {loss.item():.4f}")

    # ============================================================
    # 3) 拼接网格图
    # ============================================================
    font_size = max(IMG_SIZE // 8, 12)

    grid_attn_paths = [OUTPUT_DIR / "style.png", OUTPUT_DIR / "content.png"] + attn_paths
    grid_attn_labels = ["Style", "Content"] + [f"A-{int(p.stem[-4:])}" for p in attn_paths]
    make_square_grid(grid_attn_paths, OUTPUT_DIR / "grid_attention.png",
                     labels=grid_attn_labels, font_size=font_size)

    grid_vgg_paths = [OUTPUT_DIR / "style.png", OUTPUT_DIR / "content.png"] + vgg_paths
    grid_vgg_labels = ["Style", "Content"] + [f"V-{int(p.stem[-4:])}" for p in vgg_paths]
    make_square_grid(grid_vgg_paths, OUTPUT_DIR / "grid_vgg.png",
                     labels=grid_vgg_labels, font_size=font_size)

    print(f"\nDone! All results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
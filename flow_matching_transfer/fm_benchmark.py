import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from diffusers import AutoencoderTiny
from PIL import Image
import random
from pathlib import Path
import time
import csv
from glob import glob
import numpy as np

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

SEED = 42
N_STEPS = 20
DEPTH_FACTOR = 2
LAMBDA = 1.0
FM_CKPT_NAME = 'fm_gram.pth'
NUM_PER_STYLE = 30
BATCH_SIZE = 10

CACHE_DIR = Path('cache_gram')
COCO_CACHE = CACHE_DIR / 'coco_files.pkl'
WIKI_CACHE = CACHE_DIR / 'wiki_classes.pkl'
FEAT_CACHE_DIR = CACHE_DIR / 'features'

C0 = 512
C1 = 1024
C2 = 2048
C3 = 4096


def fmt4(x):
    return f"{x:.4g}"


def make_divisible(v, divisor=32):
    return max(divisor, int(v + divisor // 2) // divisor * divisor)


class DSConv(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, k, s, p, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.gn = nn.GroupNorm(min(32, c_out), c_out)

    def forward(self, x):
        return self.gn(self.pw(self.dw(x)))


class TimeEmbed(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, 512), nn.SiLU(), nn.Linear(512, dim))
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        dev = t.device
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=dev))
            * torch.arange(half, device=dev)
            / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class GAB_TypeA(nn.Module):
    def __init__(self):
        super().__init__()
        self.z_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), DSConv(4, C0, 3, 1, 1)
        )
        self.c_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="nearest"), DSConv(256, 64, 3, 1, 1)
        )
        self.expand = DSConv(64, C0, 1, 1, 0)
        self.out_gn = nn.GroupNorm(min(32, C0), C0)

    def forward(self, z_raw, t_emb, cond_s_raw, cond_c_raw):
        z = self.z_up(z_raw) + t_emb
        v = self.c_up(cond_c_raw)
        q = cond_s_raw

        B, _, H, W = q.shape
        N = H * W
        qf = q.view(B, 64, N)
        vf = v.view(B, 64, N)

        gram = torch.bmm(qf, qf.transpose(1, 2)) / (N ** 0.5)
        gram = F.softmax(gram + 1e-6, dim=-1)
        out = torch.bmm(gram, vf).view(B, 64, H, W)

        out = self.expand(out)
        return F.silu(self.out_gn(out + z)), q, v


class GABlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1, has_qv_skip=False):
        super().__init__()
        self.has_qv_skip = has_qv_skip
        self.stride = stride

        if stride != 1:
            self.compress = DSConv(c_in, c_out, 3, stride, 1)
        else:
            self.compress = DSConv(c_in, c_out, 1, 1, 0)

        cur_c = 64 // 2 if has_qv_skip else 64
        self.split_q = DSConv(c_out, cur_c, 1, 1, 0)
        self.split_v = DSConv(c_out, cur_c, 1, 1, 0)

        if has_qv_skip:
            self.skip_q = DSConv(64, cur_c, 1, 1, 0)
            self.skip_v = DSConv(64, cur_c, 1, 1, 0)

        self.expand = DSConv(64, c_out, 1, 1, 0)

        if stride != 1 or c_in != c_out:
            self.res = DSConv(c_in, c_out, 3, stride, 1)
        else:
            self.res = None
        self.out_gn = nn.GroupNorm(min(32, c_out), c_out)

    def forward(self, x, t_emb, q_prev=None, v_prev=None):
        x_comp = self.compress(x)

        q_cur = self.split_q(x_comp + t_emb)
        v_cur = self.split_v(x_comp + t_emb)

        if self.has_qv_skip and q_prev is not None:
            if q_prev.shape[-2:] != q_cur.shape[-2:]:
                q_prev = F.interpolate(q_prev, size=q_cur.shape[-2:], mode="nearest")
                v_prev = F.interpolate(v_prev, size=v_cur.shape[-2:], mode="nearest")
            q_prev = self.skip_q(q_prev)
            v_prev = self.skip_v(v_prev)
            q = torch.cat([q_cur, q_prev], dim=1)
            v = torch.cat([v_cur, v_prev], dim=1)
        else:
            q = q_cur
            v = v_cur

        B, C, H, W = q.shape
        N = H * W
        qf = q.view(B, C, N)
        vf = v.view(B, C, N)

        gram = torch.bmm(qf, qf.transpose(1, 2)) / (N ** 0.5)
        gram = F.softmax(gram + 1e-6, dim=-1)
        out = torch.bmm(gram, vf).view(B, C, H, W)

        out = self.expand(out)
        res = self.res(x) if self.res is not None else x
        return F.silu(self.out_gn(out + res)), q, v


class ResNetFeat(nn.Module):
    def __init__(self):
        super().__init__()
        rn = models.resnet18(weights=None)
        self.conv1 = rn.conv1
        self.bn1 = rn.bn1
        self.relu = rn.relu
        self.maxpool = rn.maxpool
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        f1 = self.layer1(x)
        x = self.layer2(f1)
        f3 = self.layer3(x)
        return f1, f3


class ScalableFlowMatchingNet(nn.Module):
    def __init__(self, depth_factor=2.0):
        super().__init__()
        self.depth_factor = depth_factor

        self.n_enc0 = max(2, int(round(2 * depth_factor)))
        self.n_enc1 = max(2, int(round(2 * depth_factor)))
        self.n_enc2 = max(4, int(round(4 * depth_factor)))
        self.n_enc3 = max(4, int(round(4 * depth_factor)))
        self.n_dec3 = max(3, int(round(3 * depth_factor)))
        self.n_dec2 = max(3, int(round(3 * depth_factor)))
        self.n_dec1 = max(3, int(round(3 * depth_factor)))
        self.n_dec0 = max(3, int(round(3 * depth_factor)))

        self.time_embed = TimeEmbed(128)

        time_dims = []
        time_dims += [C0] * self.n_enc0
        time_dims += [C1] * self.n_enc1
        time_dims += [C2] * self.n_enc2
        time_dims += [C3] * self.n_enc3
        time_dims += [C3] * self.n_dec3
        time_dims += [C2] * self.n_dec2
        time_dims += [C1] * self.n_dec1
        time_dims += [C0] * self.n_dec0
        self.time_projs = nn.ModuleList([nn.Linear(128, d) for d in time_dims])

        self.latent_aug = DSConv(4, C0, 1, 1, 0)

        self.enc0 = nn.ModuleList()
        self.enc0.append(GAB_TypeA())
        for _ in range(1, self.n_enc0):
            self.enc0.append(GABlock(C0, C0, has_qv_skip=True))

        self.enc1 = nn.ModuleList()
        self.enc1.append(GABlock(C0, C1, stride=2, has_qv_skip=True))
        for _ in range(1, self.n_enc1):
            self.enc1.append(GABlock(C1, C1, has_qv_skip=True))

        self.enc2 = nn.ModuleList()
        self.enc2.append(GABlock(C1, C2, stride=2, has_qv_skip=True))
        for _ in range(1, self.n_enc2):
            self.enc2.append(GABlock(C2, C2, has_qv_skip=True))

        self.enc3 = nn.ModuleList()
        self.enc3.append(GABlock(C2, C3, stride=2, has_qv_skip=True))
        for _ in range(1, self.n_enc3):
            self.enc3.append(GABlock(C3, C3, has_qv_skip=True))

        self.dec3 = nn.ModuleList()
        self.dec3.append(GABlock(C3 * 2, C3, has_qv_skip=True))
        for _ in range(1, self.n_dec3):
            self.dec3.append(GABlock(C3, C3, has_qv_skip=True))

        self.dec2 = nn.ModuleList()
        self.dec2.append(GABlock(C3 + C2, C2, has_qv_skip=True))
        for _ in range(1, self.n_dec2):
            self.dec2.append(GABlock(C2, C2, has_qv_skip=True))

        self.dec1 = nn.ModuleList()
        self.dec1.append(GABlock(C2 + C1, C1, has_qv_skip=True))
        for _ in range(1, self.n_dec1):
            self.dec1.append(GABlock(C1, C1, has_qv_skip=True))

        self.dec0 = nn.ModuleList()
        self.dec0.append(GABlock(C1 + C0, C0, has_qv_skip=True))
        for _ in range(1, self.n_dec0):
            self.dec0.append(GABlock(C0, C0, has_qv_skip=True))

        self.to_latent = DSConv(C0, 4, 3, 2, 1)

    def forward(self, z_raw, t, cond_s_raw, cond_c_raw):
        B = z_raw.shape[0]
        h4, w4 = z_raw.shape[2] * 2, z_raw.shape[3] * 2

        t_vec = self.time_embed(t)
        t_embs = [proj(t_vec)[:, :, None, None] for proj in self.time_projs]

        t_idx = 0

        def get_t_emb():
            nonlocal t_idx
            te = t_embs[t_idx]
            t_idx += 1
            return te

        z_aug = self.latent_aug(z_raw)
        z_aug_up = F.interpolate(z_aug, size=(h4, w4), mode="nearest")

        skips = {}
        qv_skips = {}

        for i, blk in enumerate(self.enc0):
            t_emb = get_t_emb().expand(B, -1, h4, w4)
            if i == 0:
                x, q_prev, v_prev = blk(z_raw, t_emb, cond_s_raw, cond_c_raw)
            else:
                x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)
            skips["enc0"] = x
        qv_skips["enc0"] = (q_prev, v_prev)

        for i, blk in enumerate(self.enc1):
            t_emb = get_t_emb().expand(B, -1, h4 // 2, w4 // 2)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)
            if i == len(self.enc1) - 1:
                skips["enc1"] = x
        qv_skips["enc1"] = (q_prev, v_prev)

        for i, blk in enumerate(self.enc2):
            t_emb = get_t_emb().expand(B, -1, h4 // 4, w4 // 4)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)
            if i == len(self.enc2) - 1:
                skips["enc2"] = x
        qv_skips["enc2"] = (q_prev, v_prev)

        for i, blk in enumerate(self.enc3):
            t_emb = get_t_emb().expand(B, -1, h4 // 8, w4 // 8)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)
            if i == len(self.enc3) - 1:
                skips["enc3"] = x
        qv_skips["enc3"] = (q_prev, v_prev)

        x = torch.cat([x, skips["enc3"]], dim=1)
        t_emb = get_t_emb().expand(B, -1, h4 // 8, w4 // 8)
        x, q_prev, v_prev = self.dec3[0](x, t_emb, qv_skips["enc3"][0], qv_skips["enc3"][1])
        for blk in self.dec3[1:]:
            t_emb = get_t_emb().expand(B, -1, h4 // 8, w4 // 8)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)

        x = F.interpolate(x, size=(h4 // 4, w4 // 4), mode="nearest")
        x = torch.cat([x, skips["enc2"]], dim=1)
        t_emb = get_t_emb().expand(B, -1, h4 // 4, w4 // 4)
        x, q_prev, v_prev = self.dec2[0](x, t_emb, qv_skips["enc2"][0], qv_skips["enc2"][1])
        for blk in self.dec2[1:]:
            t_emb = get_t_emb().expand(B, -1, h4 // 4, w4 // 4)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)

        x = F.interpolate(x, size=(h4 // 2, w4 // 2), mode="nearest")
        x = torch.cat([x, skips["enc1"]], dim=1)
        t_emb = get_t_emb().expand(B, -1, h4 // 2, w4 // 2)
        x, q_prev, v_prev = self.dec1[0](x, t_emb, qv_skips["enc1"][0], qv_skips["enc1"][1])
        for blk in self.dec1[1:]:
            t_emb = get_t_emb().expand(B, -1, h4 // 2, w4 // 2)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)

        x = F.interpolate(x, size=(h4, w4), mode="nearest")
        skip0_fused = skips["enc0"] + z_aug_up
        x = torch.cat([x, skip0_fused], dim=1)
        t_emb = get_t_emb().expand(B, -1, h4, w4)
        x, q_prev, v_prev = self.dec0[0](x, t_emb, qv_skips["enc0"][0], qv_skips["enc0"][1])
        for blk in self.dec0[1:]:
            t_emb = get_t_emb().expand(B, -1, h4, w4)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)

        return self.to_latent(x)


class VGGStyleLoss(nn.Module):
    def __init__(self, vgg_state_dict_path):
        super().__init__()
        vgg = models.vgg19()
        vgg.load_state_dict(torch.load(vgg_state_dict_path, map_location="cpu"))
        vgg.eval()
        self.features = vgg.features
        for p in self.features.parameters():
            p.requires_grad = False

        self.style_layers = [1, 6, 11, 20]

    def _normalize(self, x):
        x = (x + 1.0) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def _gram_matrix(self, feat):
        b, c, h, w = feat.shape
        f = feat.view(b, c, h * w)
        G = torch.bmm(f, f.transpose(1, 2))
        return G / (c * h * w)

    def forward(self, generated, style):
        gen = self._normalize(generated)
        sty = self._normalize(style)

        x = gen
        gen_feats = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.style_layers:
                gen_feats[i] = x

        style_loss = 0
        x = sty
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.style_layers:
                style_loss += F.mse_loss(self._gram_matrix(gen_feats[i]), self._gram_matrix(x))

        return style_loss


def fast_list_images(root, cache_file=None, exts=None):
    import pickle
    if exts is None:
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    if cache_file and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    root = str(root)
    files = []
    if os.path.isdir(root):
        try:
            files = [os.path.join(root, f) for f in os.listdir(root)
                     if os.path.splitext(f)[1].lower() in exts]
        except PermissionError:
            files = []
        if not files:
            patterns = [os.path.join(root, '**', f'*{ext}') for ext in exts]
            for pat in patterns:
                files.extend(glob(pat, recursive=True))
    files = sorted(set(files))
    if cache_file:
        os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(files, f)
    return files


def get_wikiart_by_classes(wiki_root, cache_file=None):
    import pickle
    if cache_file and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    wiki_root = Path(wiki_root)
    classes = {}
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    for split in ['train', 'test']:
        sp = wiki_root / split
        if not sp.exists():
            continue
        for d in sp.iterdir():
            if not d.is_dir():
                continue
            class_name = d.name
            files = [str(f) for f in d.iterdir() if f.suffix.lower() in exts]
            if files:
                if class_name not in classes:
                    classes[class_name] = []
                classes[class_name].extend(files)

    for k in classes:
        classes[k] = sorted(set(classes[k]))

    if cache_file:
        os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(classes, f)
    return classes


@torch.no_grad()
def get_or_cache_features(img_path, resnet, taesd, transform, device,
                          cache_dir=FEAT_CACHE_DIR):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mtime = os.path.getmtime(img_path)
    key = f"{Path(img_path).stem}_{mtime:.0f}_gram.pt"
    cache_file = cache_dir / key
    if cache_file.exists():
        data = torch.load(cache_file, map_location=device)
        return data['tensor_01'], data['cond_s_raw'], data['cond_c_raw'], data['latent']

    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    tensor_01 = (tensor + 1.0) / 2.0

    cond_s_raw, cond_c_raw = resnet(tensor_01)
    latent = taesd.encode(tensor_01).latents

    torch.save({
        'tensor_01': tensor_01.cpu(),
        'cond_s_raw': cond_s_raw.cpu(),
        'cond_c_raw': cond_c_raw.cpu(),
        'latent': latent.cpu(),
    }, cache_file)

    return tensor_01, cond_s_raw, cond_c_raw, latent


@torch.no_grad()
def forward_ode(model, z_data, cond_s, cond_c, n_steps=20):
    dt = 1.0 / n_steps
    z = z_data.clone()
    for i in range(n_steps):
        t = torch.full((z.shape[0],), 1.0 - i * dt, device=z.device, dtype=z.dtype)
        v = model(z, t, cond_s, cond_c)
        z = z - dt * v
    return z


@torch.no_grad()
def reverse_ode(model, z_noise, cond_s, cond_c, n_steps=20):
    dt = 1.0 / n_steps
    z = z_noise.clone()
    for i in range(n_steps):
        t = torch.full((z.shape[0],), i * dt, device=z.device, dtype=z.dtype)
        v = model(z, t, cond_s, cond_c)
        z = z + dt * v
    return z


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {device}, Seed: {SEED}, ODE steps: {N_STEPS}, lambda={LAMBDA}, Num per style: {NUM_PER_STYLE}, Batch: {BATCH_SIZE}")

    out_dir = Path("fm_benchmark_simple")
    out_dir.mkdir(exist_ok=True)

    vgg_path = Path("checkpoints/vgg.pth")
    if not vgg_path.exists():
        print("Downloading VGG19...")
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(vgg.state_dict(), vgg_path)
    vgg_style = VGGStyleLoss(str(vgg_path)).to(device)

    ckpt_dir = Path('checkpoints')
    resnet_path = ckpt_dir / 'resnet18.pth'
    rn = models.resnet18(weights=None)
    if resnet_path.exists():
        rn.load_state_dict(torch.load(resnet_path, map_location='cpu'))
    else:
        rn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        torch.save(rn.state_dict(), resnet_path)
    resnet = ResNetFeat().to(device)
    resnet.load_state_dict(rn.state_dict(), strict=False)
    resnet.eval()
    for p in resnet.parameters():
        p.requires_grad = False

    taesd_dir = ckpt_dir / 'taesd'
    taesd = AutoencoderTiny.from_pretrained(str(taesd_dir), local_files_only=True)
    taesd = taesd.to(device)
    taesd.eval()
    for p in taesd.parameters():
        p.requires_grad = False

    model = ScalableFlowMatchingNet(depth_factor=DEPTH_FACTOR).to(device)
    fm_ckpt_path = ckpt_dir / FM_CKPT_NAME
    if not fm_ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {fm_ckpt_path}')
    ckpt = torch.load(fm_ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f'Loaded ScalableFlowMatchingNet from epoch {ckpt.get("epoch", "?")}')

    coco_root = r'data\coco\train2017'
    wiki_root = r'data\wikiart_images'

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    coco_imgs = fast_list_images(coco_root, cache_file=str(COCO_CACHE))
    if len(coco_imgs) == 0:
        raise RuntimeError("No COCO images found")
    random.shuffle(coco_imgs)

    wiki_classes = get_wikiart_by_classes(wiki_root, cache_file=str(WIKI_CACHE))
    if not wiki_classes:
        raise RuntimeError("No WikiArt classes found")

    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["style_name", "total_sec", "direct_loss", "stylized_loss"])

    coco_ptr = 0
    for class_name, style_files in sorted(wiki_classes.items()):
        print("kimi 傻逼")
        if not style_files:
            continue

        if coco_ptr + NUM_PER_STYLE > len(coco_imgs):
            random.shuffle(coco_imgs)
            coco_ptr = 0
        content_batch = coco_imgs[coco_ptr:coco_ptr + NUM_PER_STYLE]
        coco_ptr += NUM_PER_STYLE

        print(f"\n{'='*60}")
        print(f"Style: {class_name} | Processing {NUM_PER_STYLE} content images")
        print(f"{'='*60}")

        # 加载风格特征（单张，共享）
        style_path = style_files[0]
        style_01, cond_s_style, cond_c_style, _ = get_or_cache_features(
            style_path, resnet, taesd, transform, device
        )

        # 加载内容特征（30张）
        content_items = []
        for i in range(NUM_PER_STYLE):
            cp = content_batch[i]
            c01, cs, cc, z = get_or_cache_features(cp, resnet, taesd, transform, device)
            content_items.append({
                'c01': c01, 'cond_s': cs, 'cond_c': cc, 'z': z
            })

        # 计算 direct_loss（单张，不计时）
        direct_losses = []
        for item in content_items:
            direct_losses.append(vgg_style(item['c01'] * 2.0 - 1.0, style_01 * 2.0 - 1.0).item())

        # Batch 推理
        for b_start in range(0, NUM_PER_STYLE, BATCH_SIZE):
            print("kimi 真傻逼")
            b_end = min(b_start + BATCH_SIZE, NUM_PER_STYLE)
            actual_bs = b_end - b_start
            batch_items = content_items[b_start:b_end]

            z_batch = torch.cat([it['z'] for it in batch_items], dim=0)
            cond_s_batch = torch.cat([it['cond_s'] for it in batch_items], dim=0)
            cond_c_batch = torch.cat([it['cond_c'] for it in batch_items], dim=0)

            cond_s_mixed = LAMBDA * cond_s_style + (1.0 - LAMBDA) * cond_s_batch

            t_start = time.perf_counter()

            z_noise = forward_ode(model, z_batch, cond_s_batch, cond_c_batch, n_steps=N_STEPS)
            z_stylized = reverse_ode(model, z_noise, cond_s_mixed, cond_c_batch, n_steps=N_STEPS)
            generated_batch = taesd.decode(z_stylized).sample

            t_end = time.perf_counter()
            per_img_sec = (t_end - t_start) / actual_bs

            # 逐张计算 stylized_loss
            for j in range(actual_bs):
                idx = b_start + j
                sty_loss = vgg_style(generated_batch[j:j+1] * 2.0 - 1.0, style_01 * 2.0 - 1.0).item()
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        class_name,
                        fmt4(per_img_sec),
                        fmt4(direct_losses[idx]),
                        fmt4(sty_loss)
                    ])

            print(f"  Batch [{b_start:02d}-{b_end:02d}] PerImg={fmt4(per_img_sec)}s")

    print(f"\nAll done. Results saved to: {csv_path}")


if __name__ == '__main__':
    main()
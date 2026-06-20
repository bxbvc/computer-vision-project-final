import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from torchvision.utils import save_image
from diffusers import AutoencoderTiny
from PIL import Image
import random
import time
from pathlib import Path
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# 可选：8-bit Adam 节省显存
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ==================== 结构级可配置参数 ====================
C0 = 512
C1 = 1024
C2 = 2048
C3 = 4096
DEPTH_FACTOR = 2
BATCH_SIZE = 8
USE_AMP = True
USE_8BIT_ADAM = True
MAX_EPOCHS = 6          # 总共训练 6 个 epoch
BASE_LR = 2e-4          # 用户要求的新学习率
ACCUMULATION_STEPS = 1

# ==================== 工具函数 ====================

def make_divisible(v, divisor=32):
    return max(divisor, int(v + divisor // 2) // divisor * divisor)


# ==================== 模型定义 ====================

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

        gram = torch.bmm(qf, qf.transpose(1, 2)) / (N**0.5)
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

        gram = torch.bmm(qf, qf.transpose(1, 2)) / (N**0.5)
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

        print(f"[ScalableFM] depth={depth_factor}")
        print(f"  Channels: C0={C0}, C1={C1}, C2={C2}, C3={C3}")
        print(
            f"  Depths: enc0={self.n_enc0}, enc1={self.n_enc1}, enc2={self.n_enc2}, enc3={self.n_enc3}"
        )
        print(
            f"          dec3={self.n_dec3}, dec2={self.n_dec2}, dec1={self.n_dec1}, dec0={self.n_dec0}"
        )

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
        x, q_prev, v_prev = self.dec3[0](
            x, t_emb, qv_skips["enc3"][0], qv_skips["enc3"][1]
        )
        for blk in self.dec3[1:]:
            t_emb = get_t_emb().expand(B, -1, h4 // 8, w4 // 8)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)

        x = F.interpolate(x, size=(h4 // 4, w4 // 4), mode="nearest")
        x = torch.cat([x, skips["enc2"]], dim=1)
        t_emb = get_t_emb().expand(B, -1, h4 // 4, w4 // 4)
        x, q_prev, v_prev = self.dec2[0](
            x, t_emb, qv_skips["enc2"][0], qv_skips["enc2"][1]
        )
        for blk in self.dec2[1:]:
            t_emb = get_t_emb().expand(B, -1, h4 // 4, w4 // 4)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)

        x = F.interpolate(x, size=(h4 // 2, w4 // 2), mode="nearest")
        x = torch.cat([x, skips["enc1"]], dim=1)
        t_emb = get_t_emb().expand(B, -1, h4 // 2, w4 // 2)
        x, q_prev, v_prev = self.dec1[0](
            x, t_emb, qv_skips["enc1"][0], qv_skips["enc1"][1]
        )
        for blk in self.dec1[1:]:
            t_emb = get_t_emb().expand(B, -1, h4 // 2, w4 // 2)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)

        x = F.interpolate(x, size=(h4, w4), mode="nearest")
        skip0_fused = skips["enc0"] + z_aug_up
        x = torch.cat([x, skip0_fused], dim=1)
        t_emb = get_t_emb().expand(B, -1, h4, w4)
        x, q_prev, v_prev = self.dec0[0](
            x, t_emb, qv_skips["enc0"][0], qv_skips["enc0"][1]
        )
        for blk in self.dec0[1:]:
            t_emb = get_t_emb().expand(B, -1, h4, w4)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)

        return self.to_latent(x)


class MixedDataset(Dataset):
    def __init__(self, coco_root, wikiart_root, transform, coco_max=20000):
        self.transform = transform
        self.paths = []

        coco_p = Path(coco_root)
        coco_all = [
            p
            for p in coco_p.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        ]
        random.shuffle(coco_all)
        self.coco_paths = coco_all[:coco_max]
        print(f"COCO: selected {len(self.coco_paths)} / {len(coco_all)}")
        self.paths.extend(self.coco_paths)

        wiki_root = Path(wikiart_root)
        for split in ["train", "test"]:
            sp = wiki_root / split
            if not sp.exists():
                continue
            for d in sp.iterdir():
                if not d.is_dir():
                    continue
                for p in d.iterdir():
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                        self.paths.append(p)
        print(f"WikiArt + COCO total: {len(self.paths)} images")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    coco_root = r"data\coco\train2017"
    wikiart_root = r"data\wikiart_images"

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    # ---- ResNet18 ----
    resnet_path = ckpt_dir / "resnet18.pth"
    if resnet_path.exists():
        print(f"Loading ResNet18 from {resnet_path}")
        rn = models.resnet18(weights=None)
        rn.load_state_dict(torch.load(resnet_path, map_location="cpu"))
    else:
        print("Downloading ResNet18 pretrained weights...")
        rn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        torch.save(rn.state_dict(), resnet_path)
        print(f"Saved ResNet18 to {resnet_path}")
    resnet = ResNetFeat().to(device)
    resnet.load_state_dict(rn.state_dict(), strict=False)
    resnet.eval()
    for p in resnet.parameters():
        p.requires_grad = False

    # ---- TAESD ----
    taesd_dir = ckpt_dir / "taesd"
    if (taesd_dir / "config.json").exists():
        print(f"Loading TAESD from {taesd_dir}")
        taesd = AutoencoderTiny.from_pretrained(str(taesd_dir), local_files_only=True)
    else:
        print("Downloading TAESD...")
        try:
            taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd")
            taesd.save_pretrained(taesd_dir)
            print(f"Saved TAESD to {taesd_dir}")
        except Exception as e:
            print(f"TAESD download failed: {e}")
            raise
    taesd = taesd.to(device)
    taesd.eval()
    for p in taesd.parameters():
        p.requires_grad = False

    # ---- 训练 ----
    model = ScalableFlowMatchingNet(depth_factor=DEPTH_FACTOR).to(device)

    if USE_8BIT_ADAM and HAS_BNB:
        print("Using 8-bit Adam optimizer")
        opt = bnb.optim.Adam8bit(model.parameters(), lr=BASE_LR)
    else:
        if USE_8BIT_ADAM and not HAS_BNB:
            print("Warning: bitsandbytes not installed, falling back to standard Adam")
        opt = torch.optim.Adam(model.parameters(), lr=BASE_LR)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params / 1e6:.2f}M")

    ds = MixedDataset(coco_root, wikiart_root, transform, coco_max=20000)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
    )

    scaler = GradScaler("cuda", enabled=USE_AMP) if USE_AMP else None

    fm_ckpt_path = ckpt_dir / "fm_gram.pth"
    start_epoch = 0
    
    # 线性学习率调度（默认，会在 resume 后重建）
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda epoch: 1.0 - epoch / MAX_EPOCHS
    )
    
    if fm_ckpt_path.exists():
        print(f"Loading checkpoint from {fm_ckpt_path}")
        # 加载到 CPU，避免 GPU 显存临时翻倍
        ckpt = torch.load(fm_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        
        if "opt" in ckpt:
            try:
                opt.load_state_dict(ckpt["opt"])
                # 强制覆盖为新学习率
                for param_group in opt.param_groups:
                    param_group["lr"] = BASE_LR
                print(f"Overridden LR to {BASE_LR:.2e}")
            except Exception:
                print("Optimizer state incompatible, reinitializing.")
        
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch}")
        
        # 释放 checkpoint 内存
        del ckpt
        torch.cuda.empty_cache()
        
        # 重建 scheduler，确保基于新的 BASE_LR
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda epoch: 1.0 - epoch / MAX_EPOCHS
        )
        # 恢复 scheduler 已走过的步数
        for _ in range(start_epoch):
            scheduler.step()

    last_save = time.time()
    global_step = 0
    loss_sum = 0.0

    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        epoch_loss_sum = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{MAX_EPOCHS}", unit="batch")
        for batch in pbar:
            batch = batch.to(device)
            B = batch.shape[0]

            with torch.no_grad():
                x01 = (batch + 1.0) / 2.0
                z1 = taesd.encode(x01).latents
                cond_s_raw, cond_c_raw = resnet(x01)

            z0 = torch.randn_like(z1)
            t = torch.rand(B, device=device)
            zt = (1 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1

            # 梯度累计
            if USE_AMP and scaler is not None:
                with autocast("cuda"):
                    v_pred = model(zt, t, cond_s_raw, cond_c_raw)
                    loss = F.mse_loss(v_pred.float(), z1 - z0)
                scaler.scale(loss / ACCUMULATION_STEPS).backward()
            else:
                v_pred = model(zt, t, cond_s_raw, cond_c_raw)
                loss = F.mse_loss(v_pred, z1 - z0)
                (loss / ACCUMULATION_STEPS).backward()

            epoch_loss_sum += loss.item()
            loss_sum += loss.item()
            n_batches += 1
            global_step += 1
            pbar.set_postfix(loss=loss.item())

            if global_step % ACCUMULATION_STEPS == 0:
                if USE_AMP and scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad()

            if global_step % 100 == 0:
                avg_loss = loss_sum / 100.0
                print(f"\n[Step {global_step}] Avg Loss (last 100): {avg_loss:.6f}")
                loss_sum = 0.0

            if time.time() - last_save > 30:
                last_save = time.time()
                model.eval()
                with torch.no_grad():
                    rand_idx = random.randint(0, len(ds) - 1)
                    ex = ds[rand_idx].unsqueeze(0).to(device)

                    ex01 = (ex + 1.0) / 2.0
                    ex_z1 = taesd.encode(ex01).latents
                    ex_cond_s_raw, ex_cond_c_raw = resnet(ex01)

                    direct_recon = taesd.decode(ex_z1).sample

                    z = torch.randn_like(ex_z1)
                    dt = 1.0 / 20.0
                    for i in range(20):
                        ti = torch.full((1,), i * dt, device=device)
                        if USE_AMP:
                            with autocast("cuda"):
                                v = model(z, ti, ex_cond_s_raw, ex_cond_c_raw)
                        else:
                            v = model(z, ti, ex_cond_s_raw, ex_cond_c_raw)
                        z = z + dt * v

                    recon = taesd.decode(z).sample
                    comp = torch.cat([ex01, direct_recon, recon], dim=0)
                    save_image(
                        comp,
                        "fm_example.png",
                        nrow=3,
                        normalize=True,
                        value_range=(0, 1),
                    )
                model.train()

        # 每个 epoch 结束后：降学习率、打印统计
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg_loss = epoch_loss_sum / n_batches
        print(
            f"\n>>> Epoch {epoch} finished | Avg Loss: {avg_loss:.6f} | LR: {current_lr:.2e}\n"
        )

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
            },
            ckpt_dir / "fm_gram.pth",
        )
        print(f"Saved checkpoints/fm_gram.pth")

    print(f"Training complete. Total epochs: {MAX_EPOCHS}")


if __name__ == "__main__":
    main()
# ============================================================
# SOUTH INDIAN DISH CLASSIFIER (FINAL VERSION)
# EfficientNet-B2 + Progressive Resizing + CutMix + EMA
# ============================================================

import os
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt

# ---------------- PATHS ----------------

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "SouthIndian_Split_Data"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True, parents=True)

# ---------------- CONFIG ----------------

BATCH_SIZE = 32
HEAD_EPOCHS = 5
FT_EPOCHS = 55
LR_HEAD = 1e-3
LR_FULL = 1e-4
WD = 1e-4
LABEL_SMOOTH = 0.05
PATIENCE = 12
CUTMIX_PROB = 0.6
EMA_DECAY = 0.999
SEED = 42

STAGE1_SIZE = 160
STAGE2_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- UTILITIES ----------------

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_topk(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).float().sum() / batch) for k in topk]


def rand_bbox(W, H, lam):
    cut_ratio = np.sqrt(1 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2


def apply_cutmix(x, y, alpha=1.0):
    if np.random.rand() > CUTMIX_PROB:
        return x, y, None, None, False

    lam = np.random.beta(alpha, alpha)
    batch, C, H, W = x.size()
    index = torch.randperm(batch, device=x.device)

    x1, y1, x2, y2 = rand_bbox(W, H, lam)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x, y, y[index], lam, True


def update_ema(model, ema_model, decay=EMA_DECAY):
    with torch.no_grad():
        for p, ema_p in zip(model.parameters(), ema_model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


# ---------------- DATA LOADING ----------------

def get_transforms(img_size):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        normalize
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    return train_tfms, val_tfms


def get_loaders(img_size):
    train_tfms, val_tfms = get_transforms(img_size)

    train_data = datasets.ImageFolder(DATA_ROOT / "train", train_tfms)
    val_data = datasets.ImageFolder(DATA_ROOT / "val", val_tfms)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    return train_data, train_loader, val_loader


# ---------------- MODEL ----------------

def build_model(num_classes):
    weights = EfficientNet_B2_Weights.IMAGENET1K_V1
    model = efficientnet_b2(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


# ---------------- TRAINING LOOP ----------------

def train_one_epoch(model, loader, opt, criterion, scaler, ema_model):
    model.train()
    total, correct1 = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        x_cm, y1, y2, lam, used = apply_cutmix(x, y)

        opt.zero_grad()

        use_amp = (device.type == "cuda")
        with autocast(device_type=device.type, enabled=use_amp):
            out = model(x_cm)
            loss = (lam * criterion(out, y1) + (1 - lam) * criterion(out, y2)) if used else criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        update_ema(model, ema_model)

        acc1 = accuracy_topk(out, y, (1,))[0]
        correct1 += acc1 * x.size(0)
        total += x.size(0)

    return correct1 / total


def validate(model, loader, criterion):
    model.eval()
    total, correct1 = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            acc1 = accuracy_topk(out, y, (1,))[0]
            correct1 += acc1 * x.size(0)
            total += x.size(0)

    return correct1 / total


# ---------------- TRAINING ENTRYPOINT ----------------

def train():
    set_seed(SEED)

    print("📌 Loading data at 160px...")
    data, train_loader, val_loader = get_loaders(STAGE1_SIZE)
    num_classes = len(data.classes)

    model = build_model(num_classes).to(device)
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = GradScaler()

    best_acc = 0
    no_improve = 0
    train_hist, val_hist = [], []

    # -------- STAGE 1: Train at 160px --------
    print("\n🚀 Stage 1 (160px)")
    opt = optim.AdamW(model.parameters(), lr=LR_HEAD, weight_decay=WD)

    for epoch in range(1, HEAD_EPOCHS + 1):
        tr_acc = train_one_epoch(model, train_loader, opt, criterion, scaler, ema_model)
        va_acc = validate(ema_model, val_loader, criterion)

        train_hist.append(tr_acc)
        val_hist.append(va_acc)

        print(f"[Stage1] Epoch {epoch} | Train={tr_acc:.3f} | Val={va_acc:.3f}")

        if va_acc > best_acc:
            best_acc = va_acc
            no_improve = 0
            torch.save(ema_model.state_dict(), MODEL_DIR / "best_efficientnet_b2.pth")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    # -------- STAGE 2: Train at 224px --------
    print("\n🚀 Stage 2 (224px)")
    data, train_loader, val_loader = get_loaders(STAGE2_SIZE)

    opt = optim.AdamW(model.parameters(), lr=LR_FULL, weight_decay=WD)

    for epoch in range(1, FT_EPOCHS + 1):
        tr_acc = train_one_epoch(model, train_loader, opt, criterion, scaler, ema_model)
        va_acc = validate(ema_model, val_loader, criterion)

        train_hist.append(tr_acc)
        val_hist.append(va_acc)

        print(f"[Stage2] Epoch {epoch} | Train={tr_acc:.3f} | Val={va_acc:.3f}")

        if va_acc > best_acc:
            best_acc = va_acc
            no_improve = 0
            torch.save(ema_model.state_dict(), MODEL_DIR / "best_efficientnet_b2.pth")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    print("\n🏆 BEST VAL ACCURACY:", f"{best_acc:.3f}")

    # Plot
    plt.plot(train_hist, label="Train")
    plt.plot(val_hist, label="Val")
    plt.legend()
    plt.title("Training Curve")
    plt.show()


if __name__ == "__main__":
    train()

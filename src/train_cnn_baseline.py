from pathlib import Path
import random
import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =========================
# 1. 基础配置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = PROJECT_ROOT / "data" / "splits" / "train"
VAL_DIR = PROJECT_ROOT / "data" / "splits" / "val"
TEST_DIR = PROJECT_ROOT / "data" / "splits" / "test"

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
WEIGHTS_DIR = PROJECT_ROOT / "results" / "weights"

SEED = 42
IMAGE_SIZE = 64
BATCH_SIZE = 64
NUM_WORKERS = 0   # Windows + PyCharm 先用 0，最稳
LEARNING_RATE = 1e-3
EPOCHS = 2        # 笔记本先做 smoke test；主机正式训练再改成 8~15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 2. 工具函数
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15, fill=0),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


def get_dataloaders():
    train_transform, eval_transform = get_transforms()

    train_dataset = datasets.ImageFolder(root=str(TRAIN_DIR), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(VAL_DIR), transform=eval_transform)
    test_dataset = datasets.ImageFolder(root=str(TEST_DIR), transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


# =========================
# 3. 简单 CNN 基线模型
# =========================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 64 -> 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 16 -> 8

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================
# 4. 训练与验证
# =========================
def run_one_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc, all_labels, all_preds


def train_model(model, train_loader, val_loader, criterion, optimizer):
    history = []
    best_val_acc = 0.0
    best_model_path = WEIGHTS_DIR / "baseline_cnn_best.pth"

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss, train_acc, _, _ = run_one_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc, _, _ = run_one_epoch(
            model, val_loader, criterion, optimizer=None
        )

        epoch_time = time.time() - start_time

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "epoch_time_sec": epoch_time
        })

        print(
            f"Epoch [{epoch}/{EPOCHS}] | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    print(f"\n最佳验证集准确率：{best_val_acc:.4f}")
    print(f"最佳模型已保存：{best_model_path}")

    return history, best_model_path


# =========================
# 5. 测试与结果保存
# =========================
def evaluate_on_test(model, test_loader, class_names):
    test_loss, test_acc, y_true, y_pred = run_one_epoch(
        model, test_loader, criterion=nn.CrossEntropyLoss(), optimizer=None
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    metrics_df = pd.DataFrame([{
        "test_loss": test_loss,
        "accuracy": test_acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1
    }])

    metrics_path = TABLES_DIR / "baseline_cnn_test_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Baseline CNN Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = FIGURES_DIR / "baseline_cnn_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n测试集结果：")
    print(f"test_loss = {test_loss:.4f}")
    print(f"accuracy  = {test_acc:.4f}")
    print(f"precision = {precision:.4f}")
    print(f"recall    = {recall:.4f}")
    print(f"f1_macro  = {f1:.4f}")
    print(f"测试指标已保存：{metrics_path}")
    print(f"混淆矩阵已保存：{cm_path}")


def save_history_and_plot(history):
    history_df = pd.DataFrame(history)
    history_path = TABLES_DIR / "baseline_cnn_history.csv"
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline CNN Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_df["epoch"], history_df["train_accuracy"], label="Train Acc")
    plt.plot(history_df["epoch"], history_df["val_accuracy"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Baseline CNN Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    curve_path = FIGURES_DIR / "baseline_cnn_curves.png"
    plt.savefig(curve_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"训练历史已保存：{history_path}")
    print(f"训练曲线已保存：{curve_path}")


# =========================
# 6. 主流程
# =========================
def main():
    set_seed(SEED)
    ensure_dirs()

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_dataloaders()
    class_names = train_dataset.classes

    print("数据加载完成")
    print(f"Device: {DEVICE}")
    print(f"类别数：{len(class_names)}")
    print(f"训练集：{len(train_dataset)}")
    print(f"验证集：{len(val_dataset)}")
    print(f"测试集：{len(test_dataset)}")

    model = SimpleCNN(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history, best_model_path = train_model(model, train_loader, val_loader, criterion, optimizer)
    save_history_and_plot(history)

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    evaluate_on_test(model, test_loader, class_names)

    print("\n基线模型训练完成")


if __name__ == "__main__":
    main()
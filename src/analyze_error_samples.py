from pathlib import Path
import random

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = PROJECT_ROOT / "data" / "splits" / "test"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
WEIGHTS_DIR = PROJECT_ROOT / "results" / "weights"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def get_transform():
    weights = ResNet18_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_model(num_classes):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def main():
    set_seed(SEED)
    ensure_dirs()

    test_dataset = datasets.ImageFolder(root=str(TEST_DIR), transform=get_transform())
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    class_names = test_dataset.classes
    weight_path = WEIGHTS_DIR / "resnet18_best.pth"

    if not weight_path.exists():
        raise FileNotFoundError(f"未找到权重文件：{weight_path}")

    model = build_model(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()

    error_rows = []

    sample_paths = [path for path, _ in test_dataset.samples]

    global_index = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = preds[i].item()

                if true_label != pred_label:
                    error_rows.append({
                        "image_path": sample_paths[global_index + i],
                        "true_class": class_names[true_label],
                        "pred_class": class_names[pred_label],
                    })

            global_index += len(labels)

    error_df = pd.DataFrame(error_rows)
    csv_path = TABLES_DIR / "error_samples_resnet18.csv"
    error_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"错误样本表已保存：{csv_path}")
    print(f"错误样本数量：{len(error_df)}")

    # 取前 6 张错误样本做展示
    if len(error_df) == 0:
        print("没有错误样本，未生成错误样本图。")
        return

    show_df = error_df.head(6)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for idx, row in enumerate(show_df.itertuples(index=False)):
        image = Image.open(row.image_path).convert("RGB")
        axes[idx].imshow(image)
        axes[idx].set_title(
            f"True: {row.true_class}\nPred: {row.pred_class}",
            fontsize=10
        )
        axes[idx].axis("off")

    plt.suptitle("ResNet18 Error Samples", fontsize=14)
    plt.tight_layout()

    fig_path = FIGURES_DIR / "error_samples_resnet18.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"错误样本图已保存：{fig_path}")


if __name__ == "__main__":
    main()
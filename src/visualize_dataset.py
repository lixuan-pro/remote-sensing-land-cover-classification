from pathlib import Path
from collections import defaultdict
import math

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import ImageFolder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "EuroSAT_RGB"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> ImageFolder:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"数据目录不存在：{DATA_DIR}")
    return ImageFolder(root=str(DATA_DIR))


def collect_class_info(dataset: ImageFolder):
    class_counts = defaultdict(int)
    sample_image_paths = {}

    for image_path, label in dataset.samples:
        class_name = dataset.classes[label]
        class_counts[class_name] += 1

        if class_name not in sample_image_paths:
            sample_image_paths[class_name] = image_path

    return class_counts, sample_image_paths


def save_class_counts_csv(class_counts: dict) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "class_name": list(class_counts.keys()),
            "sample_count": list(class_counts.values()),
        }
    ).sort_values("class_name").reset_index(drop=True)

    output_path = TABLES_DIR / "class_counts.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df


def plot_class_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.bar(df["class_name"], df["sample_count"])
    plt.title("EuroSAT RGB Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Sample Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = FIGURES_DIR / "class_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sample_grid(dataset: ImageFolder, sample_image_paths: dict) -> None:
    class_names = dataset.classes
    n_classes = len(class_names)
    ncols = 5
    nrows = math.ceil(n_classes / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 7))
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for idx, class_name in enumerate(class_names):
        image_path = sample_image_paths[class_name]
        image = Image.open(image_path).convert("RGB")

        axes[idx].imshow(image)
        axes[idx].set_title(class_name, fontsize=10)
        axes[idx].axis("off")

    plt.suptitle("EuroSAT RGB Sample Images", fontsize=14)
    plt.tight_layout()

    output_path = FIGURES_DIR / "sample_grid.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    ensure_output_dirs()

    dataset = load_dataset()
    class_counts, sample_image_paths = collect_class_info(dataset)

    df = save_class_counts_csv(class_counts)
    plot_class_distribution(df)
    plot_sample_grid(dataset, sample_image_paths)

    print("数据集可视化完成")
    print(f"数据目录：{DATA_DIR}")
    print(f"总类别数：{len(dataset.classes)}")
    print(f"总样本数：{len(dataset.samples)}")
    print("类别统计表已保存：results/tables/class_counts.csv")
    print("类别分布图已保存：results/figures/class_distribution.png")
    print("样本示例图已保存：results/figures/sample_grid.png")


if __name__ == "__main__":
    main()
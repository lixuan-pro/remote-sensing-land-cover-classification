from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "data" / "splits" / "train"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

SEED = 42


def get_one_sample_image():
    class_dirs = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError("训练集目录为空，请先运行 prepare_splits.py")

    random.seed(SEED)
    chosen_class_dir = random.choice(class_dirs)

    image_paths = sorted([
        p for p in chosen_class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])
    if not image_paths:
        raise FileNotFoundError(f"类别目录没有图片：{chosen_class_dir}")

    return chosen_class_dir.name, image_paths[0]


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    class_name, image_path = get_one_sample_image()
    image = Image.open(image_path).convert("RGB")

    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])

    augmented_images = [image]
    titles = ["Original"]

    for i in range(4):
        aug_img = train_transform(image)
        augmented_images.append(aug_img)
        titles.append(f"Augmented {i+1}")

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    for ax, img, title in zip(axes, augmented_images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.suptitle(f"Augmentation Preview - Class: {class_name}", fontsize=14)
    plt.tight_layout()

    output_path = FIGURES_DIR / "augmentation_preview.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("增强预览图已生成：results/figures/augmentation_preview.png")
    print(f"示例类别：{class_name}")
    print(f"示例图片：{image_path.name}")


if __name__ == "__main__":
    main()
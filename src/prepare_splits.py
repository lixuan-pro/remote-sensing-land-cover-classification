from pathlib import Path
from collections import defaultdict
import random
import shutil

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "EuroSAT_RGB"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


def ensure_dirs():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        (SPLIT_DIR / split_name).mkdir(parents=True, exist_ok=True)


def clear_old_split_dirs():
    for split_name in ["train", "val", "test"]:
        split_path = SPLIT_DIR / split_name
        if split_path.exists():
            shutil.rmtree(split_path)
        split_path.mkdir(parents=True, exist_ok=True)


def get_class_image_paths():
    class_to_paths = {}
    class_names = sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])

    for class_name in class_names:
        class_dir = RAW_DIR / class_name
        image_paths = sorted([
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])
        class_to_paths[class_name] = image_paths

    return class_to_paths


def split_one_class(image_paths, rng):
    image_paths = image_paths[:]
    rng.shuffle(image_paths)

    n = len(image_paths)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    train_paths = image_paths[:n_train]
    val_paths = image_paths[n_train:n_train + n_val]
    test_paths = image_paths[n_train + n_val:]

    assert len(train_paths) + len(val_paths) + len(test_paths) == n
    return train_paths, val_paths, test_paths


def copy_files(split_name, class_name, paths):
    target_dir = SPLIT_DIR / split_name / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for src_path in paths:
        dst_path = target_dir / src_path.name
        shutil.copy2(src_path, dst_path)


def save_split_counts(split_counts):
    rows = []
    for split_name, class_dict in split_counts.items():
        for class_name, count in class_dict.items():
            rows.append({
                "split": split_name,
                "class_name": class_name,
                "sample_count": count,
            })

    df = pd.DataFrame(rows).sort_values(["split", "class_name"]).reset_index(drop=True)
    output_path = TABLES_DIR / "split_counts.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"原始数据目录不存在：{RAW_DIR}")

    ensure_dirs()
    clear_old_split_dirs()

    rng = random.Random(SEED)
    class_to_paths = get_class_image_paths()

    split_counts = defaultdict(dict)

    total_train = total_val = total_test = 0

    for class_name, image_paths in class_to_paths.items():
        train_paths, val_paths, test_paths = split_one_class(image_paths, rng)

        copy_files("train", class_name, train_paths)
        copy_files("val", class_name, val_paths)
        copy_files("test", class_name, test_paths)

        split_counts["train"][class_name] = len(train_paths)
        split_counts["val"][class_name] = len(val_paths)
        split_counts["test"][class_name] = len(test_paths)

        total_train += len(train_paths)
        total_val += len(val_paths)
        total_test += len(test_paths)

    save_split_counts(split_counts)

    print("数据划分完成")
    print(f"原始数据目录：{RAW_DIR}")
    print(f"训练集样本数：{total_train}")
    print(f"验证集样本数：{total_val}")
    print(f"测试集样本数：{total_test}")
    print("划分统计表已保存：results/tables/split_counts.csv")
    print("划分结果目录已生成：data/splits/train, val, test")


if __name__ == "__main__":
    main()
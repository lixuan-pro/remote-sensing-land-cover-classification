from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def main():
    cnn_path = TABLES_DIR / "baseline_cnn_test_metrics.csv"
    resnet_path = TABLES_DIR / "resnet18_test_metrics.csv"

    cnn_df = pd.read_csv(cnn_path)
    resnet_df = pd.read_csv(resnet_path)

    comparison_df = pd.DataFrame([
        {
            "model": "Baseline CNN",
            "accuracy": cnn_df.loc[0, "accuracy"],
            "precision_macro": cnn_df.loc[0, "precision_macro"],
            "recall_macro": cnn_df.loc[0, "recall_macro"],
            "f1_macro": cnn_df.loc[0, "f1_macro"],
        },
        {
            "model": "ResNet18",
            "accuracy": resnet_df.loc[0, "accuracy"],
            "precision_macro": resnet_df.loc[0, "precision_macro"],
            "recall_macro": resnet_df.loc[0, "recall_macro"],
            "f1_macro": resnet_df.loc[0, "f1_macro"],
        }
    ])

    comparison_path = TABLES_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    plot_df = comparison_df.set_index("model")[["accuracy", "precision_macro", "recall_macro", "f1_macro"]]
    plot_df.plot(kind="bar", figsize=(10, 6))
    plt.title("Model Comparison: Baseline CNN vs ResNet18")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.tight_layout()

    fig_path = FIGURES_DIR / "model_comparison_bar.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("模型对比表已保存：results/tables/model_comparison.csv")
    print("模型对比图已保存：results/figures/model_comparison_bar.png")
    print("\n模型对比结果：")
    print(comparison_df)


if __name__ == "__main__":
    main()
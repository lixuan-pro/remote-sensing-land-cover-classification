# 模型结果总结

## 1. Baseline CNN
- Accuracy: 0.9116
- Precision(macro): 0.9118
- Recall(macro): 0.9098
- F1(macro): 0.9093

## 2. ResNet18
- Accuracy: 0.9820
- Precision(macro): 0.9819
- Recall(macro): 0.9809
- F1(macro): 0.9813

## 3. 初步结论
与简单 CNN 相比，ResNet18 在 Accuracy 和 F1 等指标上均有明显提升，说明迁移学习模型在遥感图像分类任务中具有更强的特征提取能力和更好的泛化表现，因此更适合作为本项目的最终主模型。
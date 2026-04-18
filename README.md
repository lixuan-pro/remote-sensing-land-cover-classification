# 基于卷积神经网络与迁移学习的遥感图像地表覆盖分类与对比分析

## 1. 项目简介

本项目是第二次翻转课堂的课程型深度学习分类项目，任务是基于 **EuroSAT RGB** 数据集完成遥感图像地表覆盖分类，并对 **简单 CNN 基线模型** 与 **ResNet18 迁移学习主模型** 进行对比分析。

项目目标不是单纯把模型跑出来，而是完成一条从数据准备、预处理、模型训练、模型比较、结果解释到展示准备的完整链路，并形成适合 PPT 展示和答辩讲解的结果材料。

---

## 2. 项目目标

本项目主要完成以下内容：

- 使用 EuroSAT RGB 数据集构建遥感图像分类任务
- 完成数据集检查、类别统计、训练集/验证集/测试集划分
- 设计基础数据增强方案
- 训练简单 CNN 作为 baseline
- 训练 ResNet18 迁移学习模型作为主模型
- 对两种模型进行指标比较
- 结合混淆矩阵与错误样本进行结果分析

---

## 3. 数据集说明

- 数据集名称：**EuroSAT RGB**
- 任务类型：**10 类遥感图像分类**
- 总样本数：**27000**
- 数据目录：`data/raw/EuroSAT_RGB/`

项目中使用的是 RGB 版本，不使用多波段版本，优先保证课程项目主线完整与结果可解释。

---

## 4. 模型方案

### 4.1 Baseline 模型
- 模型名称：Simple CNN
- 用途：作为第一版可运行分类基线
- 作用：用于说明最基础卷积神经网络能达到什么水平

### 4.2 主模型
- 模型名称：ResNet18
- 方式：迁移学习
- 用途：作为最终主模型
- 作用：用于和 baseline 进行比较，说明更高级方法带来的性能提升

---

## 5. 项目目录结构

```text
remote-sensing-land-cover-classification/
├─ data/
│  ├─ raw/
│  │  └─ EuroSAT_RGB/
│  ├─ processed/
│  └─ splits/
│     ├─ train/
│     ├─ val/
│     └─ test/
├─ reports/
├─ results/
│  ├─ figures/
│  ├─ tables/
│  └─ weights/
├─ src/
│  ├─ data_check.py
│  ├─ visualize_dataset.py
│  ├─ prepare_splits.py
│  ├─ preview_augmentation.py
│  ├─ train_cnn_baseline.py
│  ├─ train_resnet18_transfer.py
│  ├─ compare_models.py
│  └─ analyze_error_samples.py
├─ .gitignore
├─ README.md
└─ requirements.txt
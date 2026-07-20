# FM-VMamba：喉部病变图像分类模型

本仓库为论文实验代码的简要实现，主要用于喉部内镜图像的多类别分类任务。核心训练代码为 `train.py`，该文件实现了基于 VMamba 主干网络的 FM-VMamba 模型训练流程，并结合 FARM 频域增强模块与 Mona Adapter 参数高效微调模块，以提升喉癌及相关喉部病变图像的分类性能。

## 1. 项目简介

喉部病变图像分类对喉癌早期筛查和辅助诊断具有重要意义。针对喉部内镜图像中病灶区域细节弱、类别间差异小、局部纹理信息复杂等问题，本项目在 VMamba 视觉骨干网络基础上引入频域增强和轻量化适配器结构，构建 FM-VMamba 模型，用于实现喉部病变图像的自动分类。

模型主要包含以下部分：

- **FARM（Frequency Adaptive Restoration Module）**：通过傅里叶变换提取并调制图像频域信息，增强纹理、边缘及病灶细节表达。
- **VMamba Backbone**：基于视觉状态空间模型的主干网络，用于提取图像全局与局部特征。
- **Mona Adapter**：插入到 VSSBlock 中的轻量化适配器模块，用于参数高效微调。
- **五折交叉验证**：采用 Stratified K-Fold 保证各类别样本分布相对均衡。
- **多指标评估**：输出 Accuracy、Precision、Recall、F1-score、AUC、混淆矩阵和 ROC 曲线等结果。

## 2. 代码结构

```text
HuGuan-main/
├── train.py                         # FM-VMamba 主训练代码
├── train_laryngeal.py                # 基础 VMamba 训练代码
├── Train_FARM_Mona.py                # FARM + Mona 相关训练实验代码
├── FARM.py                           # FARM 频域增强模块测试/可视化代码
├── offline_augment.py                # 离线数据增强脚本
├── count.py                          # 数据统计脚本
├── test_image/                       # 测试图像示例
├── FARM_result_image/                # FARM 可视化结果
├── farm_effect_results/              # FARM 图像增强效果评价结果
├── farm_before_after_results/        # FARM 增强前后对比图
└── results_fm_vmamba_mona_5fold/     # 五折交叉验证结果
```

## 3. 环境要求

建议使用 Python 3.8 及以上版本，并配置支持 CUDA 的 PyTorch 环境。

主要依赖包括：

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib scikit-learn tqdm pillow
pip install timm opencv-python albumentations packaging
```

此外，`train.py` 中需要导入本地 VMamba 模型文件：

```python
from vmamba import VSSM
```

因此需要保证 `vmamba.py` 或 VMamba 相关源码已经放在项目目录下，或已正确安装到 Python 环境中。

## 4. 数据集格式
数据增广由 offline_augment.py 在正式模型训练前离线完成。
完成数据准备后，将实际用于五折实验的图像按照类别统一整理至 PreparedData/all/<class_name>。train.py 不再实施在线数据增广，而是直接读取该目录，并使用StratifiedKFold完成五折交叉验证。

训练代码使用 `torchvision.datasets.ImageFolder` 读取数据，因此数据集需要按照类别文件夹组织：

```text
PreparedData/all/
├── laryngocarcinoma/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── normal vocal fold/
│   ├── 001.jpg
│   └── ...
├── polyp/
├── leukoplakia/
├── sulcus vocalis/
└── chorditis vocalis/
```

每个子文件夹名称会被自动识别为类别标签。

## 5. 训练配置

训练参数在 `train.py` 的 `CONFIG` 字典中设置，运行前需要根据本地路径修改以下内容：

```python
CONFIG = {
    "all_data_dir": "/path/to/PreparedData/all",
    "pretrained_vssm_path": "/path/to/vssm1_tiny_0230s_ckpt_epoch_264.pth",
    "batch_size": 24,
    "input_size": 224,
    "n_splits": 5,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "num_epochs": 300,
    "results_dir": "./results_fm_vmamba_mona_5fold",
}
```

其中：

- `all_data_dir`：全部训练数据路径；
- `pretrained_vssm_path`：VMamba 预训练权重路径；
- `n_splits`：交叉验证折数，默认 5；
- `results_dir`：训练结果保存目录。

注意：`.pth` 权重文件通常较大，不建议直接上传到 GitHub，可在 README 中说明权重获取方式或单独提供下载链接。

## 6. 运行训练

修改好数据集路径和预训练权重路径后，执行：

```bash
python train.py
```

程序会自动进行五折交叉验证训练，并在每一折中保存最佳模型和评估结果。

## 7. 输出结果

训练结束后，结果会保存在：

```text
results_fm_vmamba_mona_5fold/
```

每一折会生成如下文件：

```text
fold_1/
├── best_checkpoint.pth          # 当前折最佳模型权重
├── run_config.json              # 当前折训练配置
├── training_log.csv             # 训练日志
├── training_curves.png          # Loss、Accuracy、Macro-F1 曲线
├── confusion_matrix.png         # 混淆矩阵图
├── confusion_matrix.csv         # 混淆矩阵数值
├── multiclass_roc.png           # 多类别 ROC 曲线
├── classification_report.txt    # 分类报告
├── y_true.npy                   # 真实标签
├── y_pred.npy                   # 预测标签
└── y_prob.npy                   # 预测概率
```

五折总结果会保存为：

```text
five_fold_results.csv
five_fold_summary.txt
```

当前代码包中的五折交叉验证示例结果如下：

```text
Accuracy:        93.4478 ± 0.7166
Precision Macro: 91.9896 ± 1.2724
Recall Macro:    89.2342 ± 1.0024
F1 Macro:        90.4379 ± 1.0683
AUC Macro:       0.9791 ± 0.0037
```

## 8. 模型训练流程

`train.py` 的整体流程如下：

1. 设置随机种子，保证实验可复现；
2. 使用 `ImageFolder` 读取喉部病变图像数据；
3. 采用 `StratifiedKFold` 构建五折交叉验证；
4. 构建 FM-VMamba 模型；
5. 加载 VMamba 预训练权重；
6. 注入 Mona Adapter，并根据配置冻结主干网络；
7. 使用 AdamW 优化器进行训练；
8. 采用验证集 Macro-F1 作为最佳模型保存指标；
9. 使用 Early Stopping 防止过拟合；
10. 输出分类报告、混淆矩阵、ROC 曲线和五折统计结果。

## 9. 注意事项

- 训练前请确保数据路径和预训练权重路径正确。
- GitHub 不建议上传虚拟环境文件夹，如 `vmamba_env/`。
- GitHub 普通仓库不适合直接上传较大的 `.pth`、`.pt`、`.ckpt` 模型权重文件。
- 建议在 `.gitignore` 中加入：

```text
vmamba_env/
**/vmamba_env/
*.pth
*.pt
*.ckpt
__pycache__/
*.pyc
.DS_Store
```

## 10. 引用说明

如果本代码用于论文、课程设计或后续研究，请说明模型来源与实验设置，并注明本项目基于 FM-VMamba 框架完成喉部病变图像分类实验。

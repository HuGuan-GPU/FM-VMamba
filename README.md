# FM-VMamba

**FM-VMamba: A Frequency-Reweighted Visual State Space Model for Laryngeal Lesion Classification in Electronic Laryngoscopy**

## 1. 方法简介

FM-VMamba 面向电子喉镜图像六分类任务，由以下三部分组成：

- **FARM（Fourier Adaptive Reweighting Module）**：在频域生成可学习权重，对输入图像的频率响应进行自适应重加权；
- **VMamba Backbone**：利用视觉状态空间模型和二维选择性扫描建模局部纹理与整体结构信息；
- **Mona Adapter**：在冻结大部分 VMamba 主干参数的条件下完成参数高效任务适配。

模型仅更新 FARM、Mona 适配器和分类头参数，以减少微调阶段的参数更新量。

## 2. 论文实验结果

在质量控制后的 2721 张电子喉镜图像上，采用图像级分层五折交叉验证进行内部评估。FM-VMamba 的主要结果如下：

| Metric | Mean ± Std |
|---|---:|
| Accuracy | 93.4478 ± 0.7166 % |
| Macro-Precision | 91.9896 ± 1.2724 % |
| Macro-Recall | 89.2342 ± 1.0024 % |
| Macro-F1 | 90.4379 ± 1.0683 % |
| Macro-AUC | 0.9791 ± 0.0037 |
| Trainable parameters | 1.64 M |
| Trainable parameter ratio | 5.26 % |

上述结果为五折交叉验证的均值与标准差，除 AUC 外，其余分类指标均以百分数表示。

## 3. 正式实验协议

论文中的正式实验采用以下流程：

1. 对原始电子喉镜图像进行重复与近重复质量控制；
2. 在质量控制后的原始图像上进行图像级分层五折划分，随机种子为 42；
3. 每一折仅对当前训练集执行离线数据增广；
4. 当前折验证集始终使用未增广的原始图像；
5. 每折独立训练，并根据验证集 Macro-F1 保存最佳模型；
6. 汇总五折 Accuracy、Macro-Precision、Macro-Recall、Macro-F1 和 Macro-AUC。

离线数据增广是独立的数据准备步骤，不嵌入 `train.py`。
## 4. 数据集说明

本研究数据来源于重庆医科大学附属第一医院。由于医学数据隐私、伦理审批和使用授权限制，原始电子喉镜图像不随本仓库公开。
质量控制后的六类图像数量如下：

| Class | Number of images |
|---|---:|
| Laryngeal carcinoma | 184 |
| Vocal fold polyp | 1246 |
| Vocal fold leukoplakia | 198 |
| Chorditis vocalis | 488 |
| Normal vocal folds | 496 |
| Sulcus vocalis | 109 |
| **Total** | **2721** |

用户使用自有数据时，应确保已获得相应的伦理审批、知情同意或豁免，并完成去标识化处理。

## 5. 仓库结构

```text
FM-VMamba-main/
├── train.py                         # FM-VMamba 五折训练与评估主脚本
├── train_laryngeal.py               # 基础 VMamba 训练脚本
├── Train_FARM_Mona.py               # FARM + Mona 组合实验脚本
├── FARM.py                           # FARM 模块及可视化代码
├── offline_augment.py                # 独立离线增广/固定划分辅助脚本
├── check_duplicate_images.py         # 重复与近重复图像质量控制
├── prepare_grouped_dataset.py        # 数据整理辅助脚本
├── count.py                          # 类别数量统计
├── duplicate_audit/                  # 重复与近重复审计结果
├── FARM_result_image/                # FARM 代表性可视化结果
├── farm_effect_results/              # FARM 辅助分析结果
├── results_fm_vmamba_mona_5fold/     # 五折实验汇总结果
├── figure6.svg                       # 数据类别分布图
├── figure7.svg                       # Macro-F1 与可训练参数量关系图
└── README.md
```

其中，`train_laryngeal.py`、`Train_FARM_Mona.py` 和部分辅助脚本保留用于不同阶段的模型验证与补充实验；论文主结果以正式五折实验流程为准。

## 6. 环境配置

建议使用 Python 3.9 或更高版本，并安装支持 CUDA 的 PyTorch 环境。

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib scikit-learn tqdm pillow
pip install timm opencv-python albumentations packaging
```

主要依赖包括：

- PyTorch
- torchvision
- NumPy
- pandas
- scikit-learn
- matplotlib
- timm
- OpenCV
- Albumentations

不同 PyTorch、CUDA 和 VMamba 源码版本可能导致结果存在轻微差异。

## 7. VMamba 源码与预训练权重

`train.py` 通过以下语句导入 VMamba：

```python
from vmamba import VSSM
```

运行前需要将与本项目兼容的 `vmamba.py` 及相关依赖放入项目目录，或将 VMamba 源码加入 Python 搜索路径。

同时需要在 `train.py` 中设置 VMamba 预训练权重路径：

```python
"pretrained_vssm_path": "/path/to/vssm_pretrained_checkpoint.pth"
```

预训练权重文件通常较大，本仓库不直接提供。请从 VMamba 官方项目获取与当前网络配置相匹配的预训练权重。

## 8. 数据组织

`train.py` 使用 `torchvision.datasets.ImageFolder` 读取图像，基础目录格式如下：

```text
PreparedData/all/
├── laryngeal_carcinoma/
├── vocal_fold_polyp/
├── vocal_fold_leukoplakia/
├── chorditis_vocalis/
├── normal_vocal_folds/
└── sulcus_vocalis/
```

每个类别文件夹中存放对应图像，文件夹名称将自动映射为类别标签。

对于论文正式折内增广实验，应按相同的固定五折清单分别准备每折训练集与验证集：

```text
PreparedData/folds/
├── fold_1/
│   ├── train/        # 原始训练图像 + 当前折训练集增广图像
│   └── val/          # 当前折未增广原始验证图像
├── fold_2/
├── fold_3/
├── fold_4/
└── fold_5/
```

## 9. 主要训练配置

`train.py` 中的主要配置包括：

```python
CONFIG = {
    "all_data_dir": "/path/to/PreparedData/all",
    "pretrained_vssm_path": "/path/to/vssm_pretrained_checkpoint.pth",
    "batch_size": 24,
    "input_size": 224,
    "n_splits": 5,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "num_epochs": 300,
    "seed": 42,
    "results_dir": "./results_fm_vmamba_mona_5fold",
}
```

论文实验使用：

- Input size: 224 × 224
- Optimizer: AdamW
- Initial learning rate: 2 × 10⁻⁴
- Weight decay: 1 × 10⁻⁴
- Batch size: 24
- Maximum epochs: 300
- Early stopping patience: 8
- Learning-rate scheduler factor: 0.5
- Scheduler patience: 3
- Random seed: 42
- Loss: weighted cross-entropy

## 10. 运行方式

### 10.1 数据质量控制

```bash
python check_duplicate_images.py
```

该脚本用于筛查完全重复和近重复图像，结果保存在 `duplicate_audit/`。

### 10.2 离线数据增广

```bash
python offline_augment.py
```


增广操作包括：

- 随机水平翻转；
- 轻度缩放和平移；
- 随机旋转；
- 亮度与对比度调整；
- CLAHE；
- 高斯噪声；
- 高斯模糊。

### 10.3 模型训练

修改数据路径和预训练权重路径后运行：

```bash
python train.py
```

`train.py` 会执行模型构建、预训练权重加载、Mona 注入、参数冻结、五折训练、最佳模型保存和多指标评估。

## 11. 输出文件

训练结果默认保存在：

```text
results_fm_vmamba_mona_5fold/
```

典型输出包括：

```text
fold_1/
├── best_checkpoint.pth
├── run_config.json
├── training_log.csv
├── training_curves.png
├── confusion_matrix.csv
├── confusion_matrix.png
├── multiclass_roc.png
├── classification_report.txt
├── y_true.npy
├── y_pred.npy
└── y_prob.npy
```

五折汇总结果包括：

```text
five_fold_results.csv
five_fold_summary.txt
```

## 12. 可训练参数

在参数高效微调设置下，VMamba 主干的大部分参数被冻结，仅更新：

- FARM；
- Mona adapters；
- Classification head。

论文统计结果为：

```text
Total parameters:       approximately 31.12 M
Trainable parameters:   approximately 1.64 M
Trainable ratio:        approximately 5.26 %
```

## 13. 复现注意事项

1. 论文正式结果来自图像级分层五折内部验证，不是患者级或外部验证结果；
2. 原始数据未做患者级划分，因此不能完全排除同一患者图像跨折分布的风险；
3. 复现实验时应固定随机种子和折划分；
4. 训练折可增广，验证折不得增广；
5. 不得将同一原图及其增广版本分配到不同折；
6. GitHub 中的示例结果用于展示代码输出格式，实际结果可能因环境、权重和数据划分而略有差异。

## 14. 隐私与数据使用

仓库中不提供原始临床数据。使用者应自行保证：

- 数据来源合法；
- 已获得必要的伦理审批和数据授权；
- 图像、文件名和元数据不包含可识别患者身份的信息；
- 公开可视化图像前已完成去标识化并获得相应许可。
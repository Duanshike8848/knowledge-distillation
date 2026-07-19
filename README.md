# 面向边缘部署的轻量级文本模型优化系统（Codex 开发文档）

## 项目背景

本项目来源于本科毕业设计：

**《面向边缘部署的轻量级文本模型优化方法研究》**

目标是设计并实现一个基于知识蒸馏（Knowledge Distillation）的轻量级文本模型，通过模型压缩和量化技术降低模型规模，并最终完成边缘设备部署。

---

# 开发原则

1. 搭建项目结构
2. 数据集处理
3. 教师模型训练
4. 学生模型设计
5. 知识蒸馏模块
6. 模型量化模块
7. ONNX 导出
8. 边缘部署支持
9. 性能评估
10. 可视化分析
11. README 文档



# 项目目标

开发一个完整的轻量级 NLP 模型优化系统。

要求实现：

* 教师模型训练
* 学生模型构建
* 知识蒸馏
* 动态温度机制
* 动态损失权重
* 中间特征对齐
* INT8 量化
* ONNX 导出
* 边缘部署准备
* 模型性能测试
* 实验结果可视化

---

# 技术栈

```text
Python 3.11

PyTorch
Transformers
Datasets
TorchMetrics
Scikit-learn
NumPy
Pandas
Matplotlib

ONNX
ONNXRuntime

torch.quantization

Optimum
```

---

# 项目目录结构

```text
EdgeLiteNLP/

├── data/
├── teacher/
├── student/
├── distillation/
├── quantization/
├── deploy/
├── evaluation/
├── export/
├── figures/
├── models/
├── outputs/

├── train_teacher.py
├── train_student.py
├── distill.py
├── quantize.py
├── export_onnx.py
├── benchmark.py

├── requirements.txt
└── README.md
```

---

# 数据集

优先使用公开数据集：

## 第一选择

AG News

## 可替换

* SST-2
* IMDb
* THUCNews

要求：

自动下载

自动划分：

* Train
* Validation
* Test

---
Teacher：
bert-base-chinese

Dataset：
THUCNews
或者 AG News

Student：
自己设计 TinyEdgeBERT
4层Transformer

Distillation：
✓ Soft Label
✓ Dynamic Temperature
✓ Feature Alignment
✓ Dynamic Loss Weight

Compression：
✓ QAT
✓ INT8

Deploy：
PyTorch
↓
ONNX
↓
SNPE
↓
Qualcomm QCS6490

# 教师模型

使用：

```text
bert-base-uncased
```

或者：

```text
bert-base-chinese
```

要求：

训练教师模型并保存：

```text
teacher_model.pt
```

输出指标：

* Accuracy
* Precision
* Recall
* F1 Score

---

# 学生模型

构建轻量化 Transformer。

建议：

Teacher：

```text
12 Layers
768 Hidden Size
12 Heads
```

Student：

```text
4 Layers
312 Hidden Size
6 Heads
```

保存：

```text
student_model.pt
```

---

# 知识蒸馏

## 软标签

教师模型输出 logits。

生成 soft label。

---

## 温度缩放

支持：

```python
T = 4
```

并支持动态温度：

```text
Epoch1-3:

T = 6

Epoch4-6:

T = 4

Epoch7-10:

T = 2
```

---

## 联合损失函数

实现：

```text
Loss = α × DistillationLoss
      + β × CrossEntropyLoss
```

支持动态权重：

训练初期：

```text
α=0.8
β=0.2
```

训练后期：

```text
α=0.3
β=0.7
```

---

## 中间层特征对齐

增加：

Feature Alignment Loss

采用 MSE Loss 对教师模型和学生模型中间层进行约束。

---

# 模型量化

实现：

## PTQ

Post Training Quantization

## QAT

Quantization Aware Training

比较：

* FP32
* INT8

输出：

* Accuracy
* 模型大小
* 推理速度

---

# ONNX 导出

自动导出：

```text
model.onnx
```

支持：

ONNX Runtime 推理。

---

# 边缘部署支持

生成部署流程：

```text
PyTorch

↓

ONNX

↓

SNPE DLC

↓

Qualcomm QCS6490
```

自动生成：

```bash
snpe-onnx-to-dlc
```

相关部署命令。

---

# 性能评估

统计：

## 模型精度

* Accuracy
* Precision
* Recall
* F1

---

## 模型资源消耗

* 参数量
* 模型大小
* 内存占用
* 推理延迟

---

## 输出实验表格

| Model     | Accuracy | Size | Latency |
| --------- | -------- | ---- | ------- |
| Teacher   |          |      |         |
| Student   |          |      |         |
| Distilled |          |      |         |
| INT8      |          |      |         |

---

# 可视化

自动绘制：

## 训练曲线

* Loss
* Accuracy

## 对比图

* 模型大小
* 推理时间
* 精度变化

---

# README

README 应包含：

* 项目介绍
* 环境配置
* 数据集下载
* 教师模型训练
* 蒸馏训练
* 模型量化
* ONNX 导出
* 性能测试
* 边缘部署流程

---

# 最终交付内容

请确保最终项目至少包含：

✅ 教师模型训练

✅ 学生模型设计

✅ 知识蒸馏

✅ 动态温度机制

✅ 动态损失权重

✅ 特征对齐

✅ INT8 量化

✅ ONNX 导出

✅ 性能测试

✅ 实验图表

✅ 部署文档

✅ README

---

# 开发要求

代码必须：

* 可以直接运行
* 注释清晰
* 模块化设计
* 易于扩展
* 满足本科毕业设计要求

开发过程中优先保证：

**代码可运行 > 架构美观 > 功能扩展。**

每个阶段完成后先进行测试，不允许生成无法运行的大量占位代码。

---

# 当前第一版实现：Mac 可运行的知识蒸馏最小闭环

本仓库当前已实现一个便于迁移到 Windows + RTX 3070 的最小知识蒸馏流程。

## 已选开源基座模型

第一版优先保证跨平台可运行，因此使用英文公开数据集 AG News：

| 角色 | 模型 | 说明 |
| ---- | ---- | ---- |
| Teacher | `distilbert-base-uncased` | Hugging Face 上稳定的英文文本分类基座 |
| Student | `google/bert_uncased_L-2_H-128_A-2` | Google 发布的小型 BERT，适合做轻量学生模型 |
| Dataset | `ag_news` | 自动下载，训练速度快，适合先跑通工程 |

后续如果切换到中文任务，建议组合：

| 角色 | 建议 |
| ---- | ---- |
| Teacher | `bert-base-chinese` |
| Dataset | THUCNews 或自建中文新闻分类数据 |
| Student | 中文 TinyBERT / 自定义 TinyEdgeBERT |

注意：切换中文模型时，要确保 Teacher 和 Student 的 tokenizer 兼容；如果不兼容，需要在蒸馏数据集中分别为 Teacher 和 Student 生成输入。

## 目录说明

```text
configs/default.yaml       # 训练配置，迁移到 Windows 时主要改这里
data/ag_news.py            # AG News 下载、划分、tokenize
distillation/trainer.py    # 蒸馏损失、动态温度、动态权重、特征对齐
evaluation/metrics.py      # Accuracy / Precision / Recall / F1
train_teacher.py           # 教师模型微调
distill.py                 # 学生模型知识蒸馏
evaluate_model.py          # 单独评估保存后的模型
utils.py                   # 路径、随机种子、设备选择
requirements.txt           # Python 依赖
```

## Mac 运行方式

建议先创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

先训练教师模型：

```bash
python train_teacher.py --config configs/default.yaml
```

再训练蒸馏学生模型：

```bash
python distill.py --config configs/default.yaml
```

单独评估学生模型：

```bash
python evaluate_model.py --config configs/default.yaml --model models/student_distilled
```

## Windows + RTX 3070 迁移方式

项目代码不绑定 macOS 路径，复制到 Windows 后主要修改 `configs/default.yaml`：

```yaml
dataset:
  max_train_samples: null
  max_eval_samples: null

training:
  batch_size: 16
  eval_batch_size: 32
  fp16: true
  num_workers: 2
  epochs: 3
```

Windows 上建议先确认 CUDA 可用：

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

如果输出 RTX 3070，脚本会自动选择 `cuda`，不需要改代码。

## 当前实现的蒸馏能力

`distillation/trainer.py` 已实现：

* Soft Label 蒸馏：KL Divergence
* Dynamic Temperature：从高温逐步降低到低温
* Dynamic Loss Weight：从偏重教师软标签逐步转向真实标签
* Feature Alignment：学生 CLS hidden state 经过线性投影后对齐教师 CLS hidden state

整体损失：

```text
Loss = alpha * KL(student, teacher)
     + (1 - alpha) * CE(student, label)
     + feature_loss_weight * MSE(projected_student_feature, teacher_feature)
```

## 下一阶段建议

1. 确认 Mac 上可以跑完 `train_teacher.py` 和 `distill.py`。
2. 迁移到 Windows + RTX 3070，用全量 AG News 跑出第一组实验指标。
3. 加入模型大小、参数量、推理延迟统计。
4. 再做 ONNX 导出和 INT8 动态量化。
5. 最后切换中文 THUCNews，作为毕业设计最终实验。

---

# 第二阶段：模型统计、ONNX 导出与 INT8 量化

当前已补充以下脚本：

```text
benchmark.py      # PyTorch 模型参数量、模型大小、推理延迟
export_onnx.py    # 导出 ONNX，并用 ONNXRuntime 验证 logits 误差
quantize.py       # ONNXRuntime INT8 动态量化，并比较大小/延迟/误差
```

## 统计 PyTorch 模型参数量、大小和延迟

```bash
python benchmark.py --config configs/default.yaml --model models/student_distilled
```

输出文件：

```text
models/student_distilled/benchmark.json
```

Windows + RTX 3070 上可指定 CUDA：

```bash
python benchmark.py --config configs/default.yaml --model models/student_distilled --device cuda
```

## 导出 ONNX

```bash
python export_onnx.py --config configs/default.yaml --model models/student_distilled
```

输出文件：

```text
models/onnx/model.onnx
models/onnx/onnx_export_metrics.json
```

说明：

* ONNX 导出固定在 CPU 上执行，避免 MPS/CUDA 图导出差异。
* 导出后会自动用 ONNXRuntime 跑一次样例输入，比较 PyTorch 和 ONNX logits 误差。

## INT8 动态量化

```bash
python quantize.py --config configs/default.yaml --model models/student_distilled
```

输出文件：

```text
models/quantized/model_int8.onnx
models/quantized/quantization_metrics.json
```

说明：

* 当前使用 ONNXRuntime dynamic quantization。
* 不需要校准数据集。
* 适合作为第一版边缘部署前的 INT8 量化实验。
* 后续如果需要更贴近 Qualcomm SNPE，可以在 ONNX 稳定后再接 SNPE DLC 转换。

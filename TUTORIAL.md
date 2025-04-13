# XiaoZhi-KWS 使用教程

本教程将指导您如何使用 XiaoZhi-KWS 来训练和部署一个关键词唤醒系统。

## 1. 环境准备

首先，安装所需的依赖：

```bash
pip install -r requirements.txt
```

## 2. 数据准备

为了训练模型，您需要准备以下数据：

- 关键词音频样本（例如"小智"）
- 负样本（非关键词音频）
- 背景噪音（可选，用于数据增强）

### 2.1 使用合成数据快速测试

如果您想快速测试系统而不准备真实数据，可以使用数据准备工具生成合成数据：

```bash
python -m xiaozhi_kws.utils.data_preparation --output_dir data --synthetic --num_samples 200
```

这将在 `data` 目录下生成合成的训练数据，包括关键词样本和负样本。

### 2.2 准备实际数据

对于实际训练，您需要准备以下目录结构：

```
raw_data/
  小智/  # 关键词文件夹，使用实际关键词命名
    audio1.wav
    audio2.wav
    ...
  negative/  # 负样本文件夹
    neg1.wav
    neg2.wav
    ...
  background/  # 背景噪音文件夹（可选）
    bg1.wav
    bg2.wav
    ...
```

然后，使用数据准备工具处理数据：

```bash
python -m xiaozhi_kws.utils.data_preparation --input_dir raw_data --output_dir data --keywords 小智
```

这将在 `data` 目录下按照训练需要的结构组织数据，并自动拆分为训练集和验证集。

## 3. 训练模型

### 3.1 修改配置文件

根据您的需要调整 `config/train_config.yaml` 中的配置，例如关键词、训练参数等。

### 3.2 开始训练

运行以下命令开始训练：

```bash
python -m xiaozhi_kws.train.train --config config/train_config.yaml
```

训练过程中会显示进度和指标，并将最佳模型保存在 `checkpoints` 目录中。

### 3.3 恢复训练

如果需要从检查点恢复训练：

```bash
python -m xiaozhi_kws.train.train --config config/train_config.yaml --resume checkpoints/model_best.pth
```

## 4. 模型测试与使用

### 4.1 实时测试

使用麦克风进行实时测试：

```bash
python -m xiaozhi_kws.inference.demo --model_path checkpoints/model_best.pth
```

系统会监听麦克风输入，当检测到关键词时会输出提示。

### 4.2 测试音频文件

测试单个音频文件：

```bash
python -m xiaozhi_kws.inference.demo --model_path checkpoints/model_best.pth --audio test.wav
```

或测试整个目录中的音频：

```bash
python -m xiaozhi_kws.inference.demo --model_path checkpoints/model_best.pth --audio test_dir/
```

### 4.3 调整检测阈值

可以通过 `--threshold` 参数调整检测灵敏度：

```bash
python -m xiaozhi_kws.inference.demo --model_path checkpoints/model_best.pth --threshold 0.8
```

值越高要求越严格，降低误触发；值越低灵敏度越高，但可能增加误触发率。

## 5. 注意事项

- 关键词音频应尽量清晰，避免背景噪音过大
- 增加负样本数量和多样性可以减少误触发
- 模型推理要求 CPU 资源较少，可在边缘设备上运行
- 对于实际应用，建议收集至少 100 个以上的关键词样本进行训练

## 6. 自定义开发

如果您想将该系统集成到自己的应用中，可以参考以下示例代码：

```python
import torch
from xiaozhi_kws.inference import KeywordDetector

# 加载检测器
detector = KeywordDetector.from_checkpoint("checkpoints/model_best.pth")

# 定义回调函数
def on_keyword_detected(confidence):
    print(f"检测到关键词，执行操作！置信度: {confidence:.4f}")
    # 这里执行您的自定义操作

# 开始实时检测
detector.start_detection(callback=on_keyword_detected)
```

## 7. 故障排除

### 音频设备问题

如果遇到麦克风访问错误，请检查：

- 麦克风是否已连接并工作正常
- 应用是否有麦克风访问权限
- 尝试使用 `sounddevice` 模块的 `query_devices()` 函数列出可用设备

### 模型性能问题

如果检测效果不佳：

- 增加训练数据数量和多样性
- 尝试调整模型参数，如隐藏层大小
- 调整检测阈值找到最佳平衡点
- 确保实际使用环境的噪声水平与训练数据相似 
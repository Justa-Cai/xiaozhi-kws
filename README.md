# XiaoZhi-KWS：关键词唤醒系统

这是一个基于Python的语音关键词唤醒系统，参考了乐鑫的设计方案。

## 项目结构
- `data/`: 存放训练和测试数据
- `models/`: 网络模型定义
- `utils/`: 工具函数和特征提取代码
- `train/`: 训练相关代码
- `inference/`: 推理和演示代码

## 安装依赖
```bash
pip install -r requirements.txt
```

## 训练模型
```bash
python train/train.py --config config/train_config.yaml
```

## 运行演示
```bash
python inference/demo.py --model_path checkpoints/model_best.pth
```

## 特性
- 基于MFCC特征的关键词检测
- 支持自定义关键词
- 通过计算帧平均确保识别稳定性 

# 特征提取对齐工具

这个工具集用于解决Python训练和C++推理之间的特征提取差异问题，确保模型推理结果与训练时一致。

## 问题描述

当使用Python训练关键词唤醒模型，然后在C++中部署时，由于特征提取算法实现差异，可能导致相同音频输入产生不同的特征，进而使推理结果与预期不符。主要差异来源于：

1. Mel滤波器计算方式不同
2. DCT计算方式不同
3. 功率谱计算方式不同
4. Delta特征计算方式不同
5. 窗口函数实现方式不同

## 工具说明

本项目包含以下工具：

1. `compare_features.py` - 对比Python和C++代码的特征提取结果
2. `align_features.py` - 修改C++代码，使其特征提取方式与Python/librosa一致
3. `process_audio_files.py` - 批量处理音频文件并检测唤醒词

## 使用步骤

### 1. 配置环境

确保安装了所需的Python依赖：

```bash
pip install numpy librosa matplotlib soundfile
```

对于C++代码，需要安装以下依赖：

```bash
sudo apt-get install libfftw3-dev nlohmann-json3-dev
```

### 2. 编译C++特征提取器测试工具

```bash
cd xiaozhi_kws_cpp
mkdir -p build && cd build
cmake ..
make feature_extractor_test
```

### 3. 比较特征提取结果

选择一个典型的测试音频文件，比较Python和C++的特征提取结果：

```bash
python compare_features.py --audio data/samples/test.wav --cpp-extractor xiaozhi_kws_cpp/build/feature_extractor_test
```

这将生成一个特征比较报告和可视化图表，显示两种实现的差异。

### 4. 对齐C++特征提取代码

使用比较结果修改C++特征提取器代码，使其与Python/librosa实现对齐：

```bash
python align_features.py --cpp-file xiaozhi_kws_cpp/src/feature_extractor.cpp --comparison-report feature_comparison.json --backup
```

### 5. 重新编译并测试

修改完C++代码后，重新编译并测试：

```bash
cd xiaozhi_kws_cpp/build
make
python compare_features.py --audio data/samples/test.wav --cpp-extractor xiaozhi_kws_cpp/build/feature_extractor_test
```

如果特征提取结果已经对齐，差异值应当变得非常小。

### 6. 重新测试关键词检测

使用修正后的特征提取代码重新测试关键词检测：

```bash
python process_audio_files.py --audio-dir data/test_samples --recursive
```

## 特征提取配置

为确保Python和C++代码使用相同的特征提取参数，我们使用了统一的配置文件 `config/feature_config.json`。关键参数包括：

```json
{
    "sample_rate": 16000,
    "window_size_ms": 25,
    "window_stride_ms": 10,
    "n_mfcc": 13,
    "n_fft": 512,
    "n_mels": 40,
    "use_delta": true,
    "use_delta2": true
}
```

## 常见问题

### Q: 修复后仍有小的数值差异是否正常？

A: 是的，由于浮点数运算精度和微小实现差异，即使完全对齐算法，也可能存在很小的数值差异。只要平均差异小于0.01，一般不会影响检测结果。

### Q: 为什么Python和C++有不同的参数命名？

A: 为了保持各自代码库的命名一致性，两者有不同的命名风格。在配置文件中，我们提供了两种命名的参数，以确保兼容性。 
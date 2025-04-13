#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xiaozhi_kws.utils.dataset import KeywordDataset
from xiaozhi_kws.utils.feature_extractor import FeatureExtractor
from xiaozhi_kws.models import get_model

# 添加ONNX Runtime支持
import onnxruntime as ort


class SimplifiedDetector:
    """简化的检测器类，专门用于模型验证"""
    
    def __init__(self, model, config, device=None, is_onnx=False):
        """初始化检测器"""
        self.model = model
        self.config = config
        self.is_onnx = is_onnx
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        if not is_onnx:
            self.model = self.model.to(self.device)
            self.model.eval()
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor(config["feature"])
        
        # 检测阈值
        self.detection_threshold = config["inference"]["detection_threshold"]
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None, use_onnx=True):
        """从检查点加载检测器，支持PyTorch和ONNX模型"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 判断是否为ONNX模型
        is_onnx = checkpoint_path.endswith('.onnx')
        
        if is_onnx and use_onnx:
            print(f"加载ONNX模型: {checkpoint_path}")
            # 创建ONNX运行时会话
            ort_options = ort.SessionOptions()
            ort_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # 创建执行提供者，根据设备类型选择
            if device.type == 'cuda':
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            model = ort.InferenceSession(checkpoint_path, ort_options, providers=providers)
            
            # 加载配置文件
            # 如果与模型同名的yaml配置文件存在，则加载
            config_path = checkpoint_path.replace('.onnx', '.yaml')
            if not os.path.exists(config_path):
                # 尝试寻找默认配置文件
                config_path = os.path.join(os.path.dirname(checkpoint_path), '../config/train_config.yaml')
            
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                raise FileNotFoundError(f"找不到配置文件: {config_path}")
            
            return cls(model, config, device, is_onnx=True)
        else:
            print(f"加载PyTorch模型: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            config = checkpoint["config"]
            
            # 创建模型
            model = get_model(config)
            model.load_state_dict(checkpoint["model_state_dict"])
            
            return cls(model, config, device, is_onnx=False)
    
    def detect(self, features):
        """检测是否为关键词"""
        if self.is_onnx:
            # ONNX模型推理
            # 转换输入格式
            if isinstance(features, torch.Tensor):
                features_np = features.cpu().numpy()
            else:
                features_np = features
                
            # 获取模型输入名称
            input_name = self.model.get_inputs()[0].name
            
            # 打印模型输入详情，用于调试
            print(f"[Python-Debug] 输入特征形状: {features_np.shape}")
            print(f"[Python-Debug] 输入特征样本(第一帧前5个值): {features_np[0, 0, :5]}")
            
            # 检查特征中的无效值
            nan_count = np.isnan(features_np).sum()
            inf_count = np.isinf(features_np).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"[Python-Debug] 警告: 输入特征中包含 {nan_count} 个NaN和 {inf_count} 个Inf值")
                # 替换无效值
                features_np = np.nan_to_num(features_np, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 运行推理
            outputs = self.model.run(None, {input_name: features_np})
            
            # 处理输出
            logits = outputs[0]
            print(f"[Python-Debug] 原始logits输出: {logits[0]}")
            
            probs = F.softmax(torch.tensor(logits), dim=1).numpy()
            print(f"[Python-Debug] Softmax后概率: {probs[0]}")
            
            confidence = probs[0, 1].item()  # 第二个类别的概率（关键词）
            print(f"[Python-Debug] 置信度(索引1/关键词): {confidence}, 阈值: {self.detection_threshold}")
            
            is_keyword = confidence > self.detection_threshold
        else:
            # PyTorch模型推理
            with torch.no_grad():
                # 打印输入特征详情
                print(f"[Python-Debug] 输入特征形状: {features.shape}")
                print(f"[Python-Debug] 输入特征样本(第一帧前5个值): {features[0, 0, :5].cpu().numpy()}")
                
                outputs = self.model(features)
                print(f"[Python-Debug] 原始logits输出: {outputs[0].cpu().numpy()}")
                
                probs = F.softmax(outputs, dim=1)
                print(f"[Python-Debug] Softmax后概率: {probs[0].cpu().numpy()}")
                
                confidence = probs[0, 1].item()  # 第二个类别的概率（关键词）
                print(f"[Python-Debug] 置信度(索引1/关键词): {confidence}, 阈值: {self.detection_threshold}")
                
                is_keyword = confidence > self.detection_threshold
        
        return is_keyword, confidence


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="小智唤醒词模型验证工具")
    parser.add_argument('--checkpoint', type=str, default="checkpoints/model_best_new.onnx",
                      help='模型检查点路径，默认使用ONNX模型')
    parser.add_argument('--config', type=str, default="config/train_config.yaml",
                      help='配置文件路径')
    parser.add_argument('--dataset', type=str, default="val", choices=["train", "val"],
                      help='验证的数据集（train 或 val）')
    parser.add_argument('--device', type=str, default=None,
                      help='运行设备 (cuda或cpu)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='批处理大小')
    parser.add_argument('--threshold', type=float, default=None,
                      help='检测阈值，不设置则使用配置文件中的值')
    parser.add_argument('--audio-file', type=str, default=None,
                      help='单个音频文件路径进行测试')
    parser.add_argument('--audio-dir', type=str, default=None,
                      help='音频目录，验证目录中的所有音频文件')
    parser.add_argument('--use-pytorch', action='store_true',
                      help='使用PyTorch模型而非ONNX模型')
    parser.add_argument('--verbose', action='store_true',
                      help='是否打印详细调试信息')
    return parser.parse_args()


def validate_dataset(detector, config, dataset_type="val", batch_size=32, threshold=None):
    """验证数据集上的性能"""
    # 加载数据集
    if dataset_type == "train":
        dataset = KeywordDataset(config, mode="train", augment=False)
    else:
        dataset = KeywordDataset(config, mode="val", augment=False)
    
    # 手动设置阈值
    if threshold is not None:
        detector.detection_threshold = threshold
    
    all_labels = []
    all_preds = []
    all_confidences = []
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f"验证 {dataset_type} 数据集"):
            # 获取样本
            features, label = dataset[i]
            features = features.unsqueeze(0).to(detector.device)
            label = label.item()
            
            # 预测
            if detector.is_onnx:
                # ONNX模型推理
                input_name = detector.model.get_inputs()[0].name
                features_np = features.cpu().numpy()
                outputs = detector.model.run(None, {input_name: features_np})
                logits = outputs[0]
                probs = F.softmax(torch.tensor(logits), dim=1).numpy()
                confidence = probs[0, 1].item()
                pred = 1 if confidence > detector.detection_threshold else 0
            else:
                # PyTorch模型推理
                outputs = detector.model(features)
                probs = F.softmax(outputs, dim=1)
                confidence = probs[0, 1].item()
                pred = 1 if confidence > detector.detection_threshold else 0
            
            all_labels.append(label)
            all_preds.append(pred)
            all_confidences.append(confidence)
    
    # 计算评估指标
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # 打印混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("\n混淆矩阵:")
    print(cm)
    
    # 打印分类报告
    report = classification_report(all_labels, all_preds, target_names=["非关键词", "关键词"])
    print("\n分类报告:")
    print(report)
    
    # 计算准确率
    accuracy = (all_labels == all_preds).mean()
    print(f"准确率: {accuracy:.4f}")
    
    # 对于唤醒词任务，误唤醒率（False Alarm Rate）和漏检率（Miss Rate）很重要
    if cm.shape[0] == 2 and cm.shape[1] == 2:
        tn, fp, fn, tp = cm.ravel()
        
        # 误唤醒率（非关键词被误判为关键词的比例）
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        # 漏检率（关键词被误判为非关键词的比例）
        miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        print(f"误唤醒率(FAR): {far:.4f}")
        print(f"漏检率(Miss Rate): {miss_rate:.4f}")
    
    return accuracy, all_confidences, all_labels


def test_audio_file(detector, file_path, verbose=False):
    """测试单个音频文件"""
    import librosa
    
    try:
        # 加载音频
        audio, _ = librosa.load(file_path, sr=detector.config["feature"]["sample_rate"], mono=True)
        
        if verbose:
            # 打印音频信息
            print(f"[Python-Debug] 音频长度: {len(audio)}")
            print(f"[Python-Debug] 音频范围: [{np.min(audio)}, {np.max(audio)}]")
            print(f"[Python-Debug] 音频平均值: {np.mean(audio)}")
            print(f"[Python-Debug] 非零样本数: {np.count_nonzero(audio)}")
        
        # 提取特征
        features = detector.feature_extractor.extract_features(audio)
        
        if verbose:
            # 打印特征信息
            print(f"[Python-Debug] 特征形状: {features.shape}")
            print(f"[Python-Debug] 特征范围: [{np.min(features)}, {np.max(features)}]")
            print(f"[Python-Debug] 特征平均值: {np.mean(features)}")
            print(f"[Python-Debug] 特征NaN数量: {np.isnan(features).sum()}")
            print(f"[Python-Debug] 特征Inf数量: {np.isinf(features).sum()}")
        
        if detector.is_onnx:
            # 转换为ONNX需要的输入格式
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            features_np = features_tensor.numpy()
            
            # 获取模型输入名称
            input_name = detector.model.get_inputs()[0].name
            
            if verbose:
                print(f"[Python-Debug] ONNX输入形状: {features_np.shape}")
                print(f"[Python-Debug] ONNX输入名称: {input_name}")
            
            # 运行推理
            outputs = detector.model.run(None, {input_name: features_np})
            
            # 处理输出
            logits = outputs[0]
            
            if verbose:
                print(f"[Python-Debug] 原始logits输出: {logits[0]}")
            
            probs = F.softmax(torch.tensor(logits), dim=1).numpy()
            
            if verbose:
                print(f"[Python-Debug] Softmax后概率: {probs[0]}")
            
            confidence = probs[0, 1].item()
            
            if verbose:
                print(f"[Python-Debug] 置信度(关键词): {confidence}, 阈值: {detector.detection_threshold}")
            
            is_keyword = confidence > detector.detection_threshold
        else:
            # PyTorch模型推理
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(detector.device)
            is_keyword, confidence = detector.detect(features_tensor)
        
        return {
            "file": file_path,
            "is_keyword": bool(is_keyword),
            "confidence": float(confidence)
        }
    except Exception as e:
        print(f"处理音频文件时出错: {e}")
        return {
            "file": file_path,
            "error": str(e)
        }


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        print(f"错误: 配置文件不存在: {args.config}")
        return
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 从检查点加载检测器
    print(f"从检查点加载模型: {args.checkpoint}")
    detector = SimplifiedDetector.from_checkpoint(args.checkpoint, device, use_onnx=not args.use_pytorch)
    
    # 设置自定义阈值
    if args.threshold is not None:
        detector.detection_threshold = args.threshold
        print(f"使用自定义检测阈值: {args.threshold}")
    else:
        print(f"使用配置文件中的检测阈值: {detector.detection_threshold}")
    
    # 根据不同的测试类型执行相应操作
    if args.audio_file:
        # 测试单个音频文件
        print(f"测试音频文件: {args.audio_file}")
        result = test_audio_file(detector, args.audio_file, args.verbose)
        if "error" in result:
            print(f"错误: {result['error']}")
        else:
            print(f"结果: {'是' if result['is_keyword'] else '不是'}关键词")
            print(f"置信度: {result['confidence']:.4f}")
    
    elif args.audio_dir:
        # 测试目录中的所有音频文件
        print(f"测试目录: {args.audio_dir}")
        audio_files = []
        for root, _, files in os.walk(args.audio_dir):
            for file in files:
                if file.endswith((".wav", ".mp3", ".flac")):
                    audio_files.append(os.path.join(root, file))
        
        print(f"找到 {len(audio_files)} 个音频文件")
        
        if not audio_files:
            return
        
        results = []
        for audio_file in tqdm(audio_files):
            result = test_audio_file(detector, audio_file, args.verbose)
            results.append(result)
            
        # 打印统计信息
        kw_count = sum(1 for r in results if "is_keyword" in r and r["is_keyword"])
        print(f"\n检测结果: {kw_count} 个关键词, {len(results) - kw_count} 个非关键词")
        
    else:
        # 验证数据集
        print(f"验证 {args.dataset} 数据集...")
        validate_dataset(detector, config, args.dataset, args.batch_size, args.threshold)


if __name__ == "__main__":
    main() 
import torch
import numpy as np
import collections
import time
import os
import yaml
import sounddevice as sd


class KeywordDetector:
    """关键词检测器，用于实时音频流处理"""
    
    def __init__(self, model, config, device=None):
        """
        初始化检测器
        
        Args:
            model: 已训练的模型
            config: 配置字典
            device: 运行设备
        """
        self.model = model
        self.config = config
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 特征提取
        from utils.feature_extractor import FeatureExtractor
        self.feature_extractor = FeatureExtractor(config["feature"])
        
        # 检测参数
        self.detection_threshold = config["inference"]["detection_threshold"]
        self.smoothing_window = config["inference"]["smoothing_window"]
        
        # 状态变量
        self.recent_predictions = collections.deque(maxlen=self.smoothing_window)
        self.last_trigger_time = 0
        self.trigger_cooldown = 3.0  # 触发冷却时间（秒）
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None):
        """
        从检查点加载检测器
        
        Args:
            checkpoint_path: 检查点路径
            device: 运行设备
            
        Returns:
            detector: 关键词检测器实例
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]
        
        # 导入模型
        from models import get_model
        model = get_model(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return cls(model, config, device)
    
    def detect_single(self, audio):
        """
        检测单个音频样本
        
        Args:
            audio: 音频样本，NumPy数组
            
        Returns:
            is_keyword: 是否为关键词
            confidence: 置信度
        """
        # 提取特征
        features = self.feature_extractor.extract_features(audio)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # 检测
        with torch.no_grad():
            is_keyword, confidence = self.model.detect(features_tensor, self.detection_threshold)
        
        return is_keyword, confidence
    
    def process_audio_chunk(self, audio_chunk):
        """
        处理单个音频片段
        
        Args:
            audio_chunk: 音频片段，NumPy数组
            
        Returns:
            is_trigger: 是否触发关键词
            avg_confidence: 平均置信度
        """
        # 检测
        is_keyword, confidence = self.detect_single(audio_chunk)
        
        # 添加到最近预测队列
        self.recent_predictions.append(confidence)
        
        # 计算平均置信度
        avg_confidence = np.mean(self.recent_predictions)
        
        # 判断是否触发
        is_trigger = avg_confidence > self.detection_threshold
        
        # 检查冷却时间
        current_time = time.time()
        if is_trigger and (current_time - self.last_trigger_time) < self.trigger_cooldown:
            is_trigger = False
        
        # 更新最后触发时间
        if is_trigger:
            self.last_trigger_time = current_time
        
        return is_trigger, avg_confidence
    
    def start_detection(self, callback=None, window_size_ms=1000, hop_size_ms=500):
        """
        开始实时音频检测
        
        Args:
            callback: 触发回调函数，接收置信度参数
            window_size_ms: 窗口大小（毫秒）
            hop_size_ms: 窗口步长（毫秒）
        """
        sample_rate = self.config["feature"]["sample_rate"]
        window_samples = int(window_size_ms * sample_rate / 1000)
        hop_samples = int(hop_size_ms * sample_rate / 1000)
        
        # 音频缓冲区
        audio_buffer = np.zeros(window_samples, dtype=np.float32)
        
        # 音频回调函数
        def audio_callback(indata, frames, time, status):
            nonlocal audio_buffer
            if status:
                print(f"音频状态: {status}")
                
            # 更新缓冲区
            audio_buffer = np.roll(audio_buffer, -hop_samples)
            audio_buffer[-hop_samples:] = indata[:hop_samples, 0]
            
            # 处理音频
            is_trigger, confidence = self.process_audio_chunk(audio_buffer)
            
            # 输出状态
            bar_len = 30
            bar = "#" * int(confidence * bar_len) + "-" * (bar_len - int(confidence * bar_len))
            print(f"\r置信度: [{bar}] {confidence:.4f}", end="")
            
            # 触发回调
            if is_trigger:
                print(f"\n关键词检测到！置信度: {confidence:.4f}")
                if callback:
                    callback(confidence)
        
        # 开始音频流
        with sd.InputStream(
            channels=1,
            samplerate=sample_rate,
            blocksize=hop_samples,
            callback=audio_callback
        ):
            print(f"正在监听关键词: {', '.join(self.config['data']['keywords'])}")
            print(f"按 Ctrl+C 退出")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n已停止监听")
    
    def batch_detect(self, audio_files, verbose=True):
        """
        批量检测音频文件
        
        Args:
            audio_files: 音频文件路径列表
            verbose: 是否打印详细信息
            
        Returns:
            results: 检测结果列表
        """
        import librosa
        
        results = []
        
        for i, audio_file in enumerate(audio_files):
            if verbose:
                print(f"处理 [{i+1}/{len(audio_files)}]: {audio_file}")
                
            # 加载音频
            try:
                audio, _ = librosa.load(audio_file, sr=self.config["feature"]["sample_rate"], mono=True)
            except Exception as e:
                print(f"无法加载音频 {audio_file}: {e}")
                results.append({"file": audio_file, "error": str(e)})
                continue
            
            # 检测
            is_keyword, confidence = self.detect_single(audio)
            
            # 保存结果
            result = {
                "file": audio_file,
                "is_keyword": bool(is_keyword),
                "confidence": float(confidence)
            }
            results.append(result)
            
            if verbose:
                status = "✓" if is_keyword else "✗"
                print(f"  {status} 置信度: {confidence:.4f}")
        
        return results 
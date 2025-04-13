import numpy as np
import librosa
import torch


class FeatureExtractor:
    """MFCC特征提取器，用于关键词检测"""
    
    def __init__(self, config):
        """
        初始化特征提取器
        
        Args:
            config: 特征配置字典，包含以下字段：
                - sample_rate: 采样率
                - window_size_ms: 窗口大小（毫秒）
                - window_stride_ms: 窗口步长（毫秒）
                - n_mfcc: MFCC特征数量
                - n_fft: FFT点数
                - n_mels: Mel滤波器组数量
                - use_delta: 是否使用MFCC一阶差分
                - use_delta2: 是否使用MFCC二阶差分
        """
        self.sample_rate = config["sample_rate"]
        self.window_size = int(config["window_size_ms"] * self.sample_rate / 1000)
        self.window_stride = int(config["window_stride_ms"] * self.sample_rate / 1000)
        self.n_mfcc = config["n_mfcc"]
        self.n_fft = config["n_fft"]
        self.n_mels = config["n_mels"]
        self.use_delta = config["use_delta"]
        self.use_delta2 = config["use_delta2"]
    
    def extract_features(self, audio, normalize=True):
        """
        从音频信号中提取MFCC特征
        
        Args:
            audio: 音频信号，numpy数组
            normalize: 是否对特征进行归一化
            
        Returns:
            features: MFCC特征，shape=(num_frames, num_features)
        """
        # 确保音频是浮点型
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # 如果音频振幅不在[-1, 1]范围内，进行归一化
        if normalize and np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
            
        # 确保音频长度足够，至少能提取5帧特征
        min_samples = self.window_size + 4 * self.window_stride
        if len(audio) < min_samples:
            # 填充音频到最小长度
            audio = np.pad(audio, (0, min_samples - len(audio)), 'constant')
        
        # 提取MFCC特征
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.window_stride,
            win_length=self.window_size,
            n_mels=self.n_mels
        )
        
        # 转置特征以便每行代表一帧
        mfccs = mfccs.T  # shape: (num_frames, n_mfcc)
        
        features = [mfccs]
        
        # 添加一阶差分（delta）特征
        if self.use_delta:
            delta = librosa.feature.delta(mfccs.T, width=3).T
            features.append(delta)
        
        # 添加二阶差分（delta2）特征
        if self.use_delta2:
            delta2 = librosa.feature.delta(mfccs.T, order=2, width=3).T
            features.append(delta2)
        
        # 合并所有特征
        features = np.concatenate(features, axis=1)
        
        return features
    
    def extract_features_batch(self, audio_list, normalize=True):
        """
        批量提取特征
        
        Args:
            audio_list: 音频信号列表
            normalize: 是否对特征进行归一化
            
        Returns:
            features_list: MFCC特征列表
        """
        features_list = []
        for audio in audio_list:
            features = self.extract_features(audio, normalize=normalize)
            features_list.append(features)
        return features_list
    
    def extract_features_sliding_window(self, audio, window_size_ms=1000, stride_ms=500):
        """
        使用滑动窗口提取特征，用于连续音频流处理
        
        Args:
            audio: 音频信号
            window_size_ms: 窗口大小（毫秒）
            stride_ms: 窗口步长（毫秒）
            
        Returns:
            features_windows: 窗口化的MFCC特征列表
        """
        window_samples = int(window_size_ms * self.sample_rate / 1000)
        stride_samples = int(stride_ms * self.sample_rate / 1000)
        
        windows = []
        for start in range(0, len(audio) - window_samples + 1, stride_samples):
            window = audio[start:start + window_samples]
            windows.append(window)
        
        features_windows = self.extract_features_batch(windows)
        return features_windows
    
    def convert_to_tensor(self, features):
        """
        将NumPy特征转换为PyTorch张量
        
        Args:
            features: NumPy特征数组
            
        Returns:
            tensor: PyTorch张量
        """
        if isinstance(features, list):
            return [torch.FloatTensor(f) for f in features]
        else:
            return torch.FloatTensor(features) 
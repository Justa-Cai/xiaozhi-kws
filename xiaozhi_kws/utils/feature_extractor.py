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
                - preemphasis_coeff: 预加重系数
        """
        self.sample_rate = config["sample_rate"]
        self.window_size = int(config["window_size_ms"] * self.sample_rate / 1000)
        self.window_stride = int(config["window_stride_ms"] * self.sample_rate / 1000)
        self.n_fft = config["n_fft"]
        self.n_mels = config.get("n_mels", 80)
        self.n_mfcc = config.get("n_mfcc", 40)
        self.use_delta = config.get("use_delta", True)
        self.use_delta2 = config.get("use_delta2", False)
        self.preemphasis_coeff = config.get("preemphasis_coeff", 0.97)
    
    def extract_features(self, audio, normalize=True):
        """
        从音频信号中提取MFCC特征 (可能包含Delta)

        Args:
            audio: 音频信号，numpy数组
            normalize: 是否对特征进行归一化 (幅度归一化)

        Returns:
            features: MFCC特征，shape=(num_frames, num_features)
        """
        # 确保音频是浮点型
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # 幅度归一化
        if normalize and np.max(np.abs(audio)) > 1.0:
             max_abs = np.max(np.abs(audio))
             if max_abs > 1e-6:
                 audio = audio / max_abs
             else:
                 audio = np.zeros_like(audio)

        # 1. Pre-emphasis
        if self.preemphasis_coeff > 0.0:
            audio = np.append(audio[0], audio[1:] - self.preemphasis_coeff * audio[:-1])

        # 2. Compute MFCC using librosa
        # Use n_mels for the Mel filterbank base, then extract n_mfcc coefficients
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.window_stride,
            win_length=self.window_size,
            window='hann',
            center=True,            # Use padding similar to C++ fix
            n_mels=self.n_mels,     # Base Mel filters
            n_mfcc=self.n_mfcc,     # Number of MFCC coefficients to keep
            # power=2.0 is default for mfcc internal melspectrogram
        )

        # Transpose to have shape (num_frames, n_mfcc)
        mfccs = mfccs.T

        # 3. Calculate delta features if needed
        features_list = [mfccs]
        if self.use_delta:
            # Librosa's default width is 9, let's use a smaller width like 3 for consistency
            # with typical KWS setups unless config specifies otherwise. Let's assume width=3.
            delta = librosa.feature.delta(mfccs.T, width=3).T # Calculate delta on transposed mfccs
            features_list.append(delta)
        if self.use_delta2: # Although config says false, keep the logic just in case
             delta2 = librosa.feature.delta(mfccs.T, order=2, width=3).T
             features_list.append(delta2)

        # 4. Concatenate features
        features = np.concatenate(features_list, axis=1)

        # Ensure minimum length (optional, C++ might handle this)
        # min_frames = 5 # Example
        # if features.shape[0] < min_frames:
        #     padding = np.zeros((min_frames - features.shape[0], features.shape[1]))
        #     features = np.vstack((features, padding))

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
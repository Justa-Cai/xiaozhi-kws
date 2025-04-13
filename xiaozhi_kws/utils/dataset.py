import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import random
from .feature_extractor import FeatureExtractor


class KeywordDataset(Dataset):
    """关键词数据集类"""
    
    def __init__(self, config, mode="train", augment=True):
        """
        初始化数据集
        
        Args:
            config: 数据配置字典
            mode: 'train' 或 'val'
            augment: 是否进行数据增强
        """
        self.config = config
        self.mode = mode
        self.augment = augment
        self.sample_rate = config["feature"]["sample_rate"]
        
        # 确定数据路径
        if mode == "train":
            self.data_path = config["data"]["train_path"]
        else:
            self.data_path = config["data"]["val_path"]
        
        # 关键词和标签
        self.keywords = config["data"]["keywords"]
        self.negative_path = config["data"]["negative_path"]
        self.background_path = config["data"]["background_path"]
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor(config["feature"])
        
        # 加载数据
        self.samples = []
        self.labels = []
        self._load_data()
    
    def _load_data(self):
        """加载音频数据和标签"""
        # 加载关键词样本
        for keyword in self.keywords:
            keyword_path = os.path.join(self.data_path, keyword)
            if not os.path.exists(keyword_path):
                print(f"警告: 关键词目录不存在 {keyword_path}")
                continue
                
            audio_files = [
                os.path.join(keyword_path, f) for f in os.listdir(keyword_path)
                if f.endswith((".wav", ".mp3", ".flac"))
            ]
            
            for audio_file in audio_files:
                self.samples.append(audio_file)
                self.labels.append(1)  # 1 表示关键词
        
        # 加载负样本（非关键词）
        if os.path.exists(self.negative_path):
            negative_files = [
                os.path.join(self.negative_path, f) for f in os.listdir(self.negative_path)
                if f.endswith((".wav", ".mp3", ".flac"))
            ]
            
            for audio_file in negative_files:
                self.samples.append(audio_file)
                self.labels.append(0)  # 0 表示非关键词
    
    def __len__(self):
        return len(self.samples)
    
    def _load_audio(self, audio_path):
        """加载音频文件"""
        try:
            audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # 确保音频长度至少为0.5秒
            min_samples = int(0.5 * self.sample_rate)
            if len(audio) < min_samples:
                audio = np.pad(audio, (0, min_samples - len(audio)), 'constant')
                
            return audio
        except Exception as e:
            print(f"警告: 无法加载音频文件 {audio_path}: {e}")
            # 返回一个0.5秒的空白音频
            return np.zeros(int(0.5 * self.sample_rate))
    
    def _augment_audio(self, audio):
        """对音频进行数据增强"""
        if not self.augment:
            return audio
            
        # 1. 随机调整音量
        if random.random() < 0.5:
            gain = random.uniform(0.7, 1.3)
            audio = audio * gain
            # 确保在[-1, 1]范围内
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
        
        # 2. 添加随机噪声
        if random.random() < 0.3:
            noise_level = random.uniform(0.001, 0.02)
            noise = np.random.randn(len(audio)) * noise_level
            audio = audio + noise
            # 确保在[-1, 1]范围内
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
        
        # 3. 添加背景噪音（如果存在背景噪音目录）
        if random.random() < 0.4 and os.path.exists(self.background_path):
            background_files = [
                os.path.join(self.background_path, f) for f in os.listdir(self.background_path)
                if f.endswith((".wav", ".mp3", ".flac"))
            ]
            if background_files:
                bg_file = random.choice(background_files)
                background, _ = librosa.load(bg_file, sr=self.sample_rate, mono=True)
                
                # 如果背景音频比目标音频长，随机选择一段
                if len(background) > len(audio):
                    start = random.randint(0, len(background) - len(audio))
                    background = background[start:start + len(audio)]
                else:
                    # 如果背景音频较短，则循环重复
                    num_repeats = int(np.ceil(len(audio) / len(background)))
                    background = np.tile(background, num_repeats)
                    background = background[:len(audio)]
                
                # 混合背景音
                bg_level = random.uniform(0.05, 0.2)
                audio = audio * (1 - bg_level) + background * bg_level
        
        # 4. 随机时间移动
        if random.random() < 0.3:
            shift = random.randint(-2000, 2000)  # 移动样本数
            if shift > 0:
                audio = np.pad(audio, (0, shift), 'constant')[:len(audio)]
            elif shift < 0:
                audio = np.pad(audio, (-shift, 0), 'constant')[:-shift]
        
        return audio
    
    def __getitem__(self, idx):
        """获取数据集项"""
        audio_path = self.samples[idx]
        label = self.labels[idx]
        
        # 加载音频
        audio = self._load_audio(audio_path)
        
        # 数据增强（仅训练集）
        if self.mode == "train" and self.augment:
            audio = self._augment_audio(audio)
        
        try:
            # 提取特征
            features = self.feature_extractor.extract_features(audio)
        except Exception as e:
            print(f"警告: 特征提取失败 {audio_path}: {e}")
            # 生成一个简单的特征矩阵代替
            n_features = self.feature_extractor.n_mfcc
            if self.feature_extractor.use_delta:
                n_features *= 2
            if self.feature_extractor.use_delta2:
                n_features += self.feature_extractor.n_mfcc
            features = np.zeros((10, n_features))  # 10帧作为默认长度
        
        # 转换为张量
        features_tensor = torch.FloatTensor(features)
        label_tensor = torch.LongTensor([label])
        
        return features_tensor, label_tensor
    
    @staticmethod
    def collate_fn(batch):
        """批处理函数，处理不同长度的序列"""
        features = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        # 找出最长序列的长度
        max_len = max([feat.shape[0] for feat in features])
        
        # 填充较短的序列
        padded_features = []
        for feat in features:
            padded_feat = torch.nn.functional.pad(
                feat, (0, 0, 0, max_len - feat.shape[0])
            )
            padded_features.append(padded_feat)
        
        # 合并成批次
        features_batch = torch.stack(padded_features)
        labels_batch = torch.cat(labels)
        
        return features_batch, labels_batch


def get_data_loaders(config, batch_size=None):
    """获取训练和验证数据加载器"""
    if batch_size is None:
        batch_size = config["train"]["batch_size"]
    
    train_dataset = KeywordDataset(config, mode="train", augment=True)
    val_dataset = KeywordDataset(config, mode="val", augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=KeywordDataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=KeywordDataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader 
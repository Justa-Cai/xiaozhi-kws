import os
import argparse
import shutil
import random
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm


def prepare_dataset(input_dir, output_dir, keywords, train_ratio=0.8, sample_rate=16000):
    """
    准备关键词唤醒训练数据集
    
    Args:
        input_dir: 输入数据目录
        output_dir: 输出数据目录
        keywords: 关键词列表
        train_ratio: 训练集比例
        sample_rate: 采样率
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    negative_dir = os.path.join(output_dir, "negative")
    background_dir = os.path.join(output_dir, "background")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(negative_dir, exist_ok=True)
    os.makedirs(background_dir, exist_ok=True)
    
    # 处理关键词样本
    for keyword in keywords:
        keyword_dir = os.path.join(input_dir, keyword)
        if not os.path.exists(keyword_dir):
            print(f"警告: 关键词目录不存在 {keyword_dir}")
            continue
        
        # 创建输出目录
        os.makedirs(os.path.join(train_dir, keyword), exist_ok=True)
        os.makedirs(os.path.join(val_dir, keyword), exist_ok=True)
        
        # 获取音频文件
        audio_files = [
            f for f in os.listdir(keyword_dir)
            if f.endswith((".wav", ".mp3", ".flac"))
        ]
        
        random.shuffle(audio_files)
        split_idx = int(len(audio_files) * train_ratio)
        
        train_files = audio_files[:split_idx]
        val_files = audio_files[split_idx:]
        
        print(f"处理关键词 '{keyword}': {len(train_files)} 训练样本, {len(val_files)} 验证样本")
        
        # 处理训练集
        for audio_file in tqdm(train_files, desc=f"处理 {keyword} 训练集"):
            src_path = os.path.join(keyword_dir, audio_file)
            dst_path = os.path.join(train_dir, keyword, audio_file)
            
            # 重采样并保存
            audio, _ = librosa.load(src_path, sr=sample_rate, mono=True)
            sf.write(dst_path, audio, sample_rate)
        
        # 处理验证集
        for audio_file in tqdm(val_files, desc=f"处理 {keyword} 验证集"):
            src_path = os.path.join(keyword_dir, audio_file)
            dst_path = os.path.join(val_dir, keyword, audio_file)
            
            # 重采样并保存
            audio, _ = librosa.load(src_path, sr=sample_rate, mono=True)
            sf.write(dst_path, audio, sample_rate)
    
    # 处理负样本
    negative_src_dir = os.path.join(input_dir, "negative")
    if os.path.exists(negative_src_dir):
        audio_files = [
            f for f in os.listdir(negative_src_dir)
            if f.endswith((".wav", ".mp3", ".flac"))
        ]
        
        print(f"处理负样本: {len(audio_files)} 个文件")
        
        for audio_file in tqdm(audio_files, desc="处理负样本"):
            src_path = os.path.join(negative_src_dir, audio_file)
            dst_path = os.path.join(negative_dir, audio_file)
            
            # 重采样并保存
            audio, _ = librosa.load(src_path, sr=sample_rate, mono=True)
            sf.write(dst_path, audio, sample_rate)
    
    # 处理背景噪音
    background_src_dir = os.path.join(input_dir, "background")
    if os.path.exists(background_src_dir):
        audio_files = [
            f for f in os.listdir(background_src_dir)
            if f.endswith((".wav", ".mp3", ".flac"))
        ]
        
        print(f"处理背景噪音: {len(audio_files)} 个文件")
        
        for audio_file in tqdm(audio_files, desc="处理背景噪音"):
            src_path = os.path.join(background_src_dir, audio_file)
            dst_path = os.path.join(background_dir, audio_file)
            
            # 重采样并保存
            audio, _ = librosa.load(src_path, sr=sample_rate, mono=True)
            sf.write(dst_path, audio, sample_rate)


def generate_synthetic_data(output_dir, num_samples=100, sample_rate=16000, duration=1.0):
    """
    生成合成测试数据（仅用于测试系统）
    
    Args:
        output_dir: 输出目录
        num_samples: 样本数量
        sample_rate: 采样率
        duration: 音频时长（秒）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    keyword_dir = os.path.join(output_dir, "小智")
    negative_dir = os.path.join(output_dir, "negative")
    background_dir = os.path.join(output_dir, "background")
    
    os.makedirs(keyword_dir, exist_ok=True)
    os.makedirs(negative_dir, exist_ok=True)
    os.makedirs(background_dir, exist_ok=True)
    
    # 生成关键词样本（简单的正弦波 + 噪声）
    print(f"生成 {num_samples} 个合成的关键词样本...")
    for i in tqdm(range(num_samples)):
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # 关键词: 混合频率的正弦波
        freqs = [300, 450, 600]  # 模拟人声频率
        signal = np.zeros_like(t)
        for freq in freqs:
            signal += np.sin(2 * np.pi * freq * t)
        
        # 添加包络
        envelope = np.sin(np.pi * t / duration)
        signal = signal * envelope
        
        # 添加一些噪声
        noise = np.random.normal(0, 0.01, len(signal))
        signal = signal + noise
        
        # 归一化
        signal = signal / np.max(np.abs(signal))
        
        # 保存
        sf.write(os.path.join(keyword_dir, f"keyword_{i:04d}.wav"), signal, sample_rate)
    
    # 生成负样本（随机噪声）
    print(f"生成 {num_samples} 个合成的负样本...")
    for i in tqdm(range(num_samples)):
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # 负样本: 随机频率的正弦波混合
        freqs = np.random.uniform(100, 800, 5)  # 随机频率
        signal = np.zeros_like(t)
        for freq in freqs:
            signal += np.sin(2 * np.pi * freq * t) * np.random.uniform(0.5, 1.0)
        
        # 添加较多噪声
        noise = np.random.normal(0, 0.1, len(signal))
        signal = signal + noise
        
        # 归一化
        signal = signal / np.max(np.abs(signal))
        
        # 保存
        sf.write(os.path.join(negative_dir, f"negative_{i:04d}.wav"), signal, sample_rate)
    
    # 生成背景噪音
    print("生成 10 个合成的背景噪音样本...")
    for i in tqdm(range(10)):
        t = np.linspace(0, duration * 3, int(sample_rate * duration * 3), endpoint=False)
        
        # 背景噪音: 不同颜色的噪声
        noise_type = random.choice(["white", "pink", "brown"])
        if noise_type == "white":
            noise = np.random.normal(0, 0.1, len(t))
        elif noise_type == "pink":
            # 简化的粉噪声生成
            white_noise = np.random.normal(0, 1, len(t))
            pink_noise = np.zeros_like(white_noise)
            for i in range(1, len(t)):
                pink_noise[i] = 0.9 * pink_noise[i-1] + white_noise[i]
            noise = pink_noise * 0.1
        else:  # brown
            # 简化的棕噪声生成
            white_noise = np.random.normal(0, 1, len(t))
            brown_noise = np.zeros_like(white_noise)
            for i in range(1, len(t)):
                brown_noise[i] = 0.98 * brown_noise[i-1] + white_noise[i] * 0.05
            noise = brown_noise * 0.3
        
        # 归一化
        noise = noise / np.max(np.abs(noise))
        
        # 保存
        sf.write(os.path.join(background_dir, f"background_{i:04d}.wav"), noise, sample_rate)
    
    print(f"合成数据生成完成，保存到 {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="XiaoZhi-KWS 数据准备工具")
    parser.add_argument('--input_dir', type=str, required=False,
                      help='输入数据目录')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='输出数据目录')
    parser.add_argument('--keywords', type=str, nargs='+', default=["小智"],
                      help='关键词列表')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                      help='训练集比例')
    parser.add_argument('--sample_rate', type=int, default=16000,
                      help='采样率')
    parser.add_argument('--synthetic', action='store_true',
                      help='生成合成数据进行测试')
    parser.add_argument('--num_samples', type=int, default=100,
                      help='合成数据样本数量')
    
    args = parser.parse_args()
    
    if args.synthetic:
        print("生成合成测试数据...")
        generate_synthetic_data(
            args.output_dir,
            num_samples=args.num_samples,
            sample_rate=args.sample_rate
        )
    else:
        if not args.input_dir:
            parser.error("需要提供 --input_dir 参数，或使用 --synthetic 生成合成数据")
        
        print(f"准备数据集...")
        print(f"关键词: {args.keywords}")
        prepare_dataset(
            args.input_dir,
            args.output_dir,
            args.keywords,
            train_ratio=args.train_ratio,
            sample_rate=args.sample_rate
        )
    
    print("完成!")


if __name__ == "__main__":
    main() 
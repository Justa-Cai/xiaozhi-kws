import os
import sys
import argparse
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import KeywordDetector


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="XiaoZhi-KWS 关键词唤醒系统演示")
    parser.add_argument('--model_path', type=str, required=True,
                      help='模型检查点路径')
    parser.add_argument('--threshold', type=float, default=None,
                      help='检测阈值，覆盖配置中的值')
    parser.add_argument('--device', type=str, default=None,
                      help='运行设备 (cuda或cpu)')
    parser.add_argument('--audio', type=str, default=None,
                      help='测试音频文件或目录，为空则进行实时检测')
    return parser.parse_args()


def trigger_callback(confidence):
    """关键词触发回调函数"""
    print(f"✨ 关键词已触发！置信度: {confidence:.4f}")
    # 这里可以添加触发后的动作，例如播放声音、执行命令等
    

def main():
    """主函数"""
    args = parse_args()
    
    print("正在加载模型...")
    detector = KeywordDetector.from_checkpoint(args.model_path, device=args.device)
    
    # 如果提供了自定义阈值
    if args.threshold is not None:
        detector.detection_threshold = args.threshold
        print(f"检测阈值已设置为: {args.threshold}")
    
    # 如果提供了测试音频
    if args.audio:
        if os.path.isdir(args.audio):
            # 获取目录中的所有音频文件
            audio_files = []
            for root, _, files in os.walk(args.audio):
                for file in files:
                    if file.endswith((".wav", ".mp3", ".flac")):
                        audio_files.append(os.path.join(root, file))
            
            print(f"找到 {len(audio_files)} 个音频文件")
            if audio_files:
                results = detector.batch_detect(audio_files)
                
                # 打印统计结果
                positives = sum(1 for r in results if r.get("is_keyword", False))
                print(f"\n检测结果: {positives}/{len(results)} 个文件包含关键词")
            else:
                print("未找到音频文件")
                
        elif os.path.isfile(args.audio):
            # 单个音频文件
            results = detector.batch_detect([args.audio])
        else:
            print(f"无效的音频路径: {args.audio}")
            return
    else:
        # 实时检测
        print("开始实时关键词检测...")
        detector.start_detection(callback=trigger_callback)


if __name__ == "__main__":
    main() 
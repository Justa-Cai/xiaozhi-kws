import os
import sys
import argparse
import yaml
import torch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model
from utils import get_data_loaders
from trainer import Trainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="XiaoZhi-KWS 关键词唤醒系统训练")
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                      help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                      help='恢复训练的检查点路径')
    parser.add_argument('--device', type=str, default=None,
                      help='训练设备 (cuda或cpu)')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 打印配置信息
    print("配置信息:")
    print(f"- 模型类型: {config['model']['type']}")
    print(f"- 学习率: {config['train']['learning_rate']}")
    print(f"- 批大小: {config['train']['batch_size']}")
    print(f"- 训练轮数: {config['train']['epochs']}")
    print(f"- 关键词: {config['data']['keywords']}")
    
    # 准备数据加载器
    train_loader, val_loader = get_data_loaders(config)
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    
    # 创建模型
    model = get_model(config)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 创建训练器
    trainer = Trainer(model, config, train_loader, val_loader, device)
    
    # 开始训练
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main() 
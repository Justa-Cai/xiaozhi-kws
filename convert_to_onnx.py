#!/usr/bin/env python3
"""
转换PyTorch模型为ONNX格式
"""
import os
import sys
import argparse
import torch
import yaml

# 添加xiaozhi_kws到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入xiaozhi_kws模块
from xiaozhi_kws.models import get_model


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="转换PyTorch模型为ONNX格式")
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='PyTorch模型检查点路径')
    parser.add_argument('--output', type=str, default=None,
                      help='输出ONNX模型路径，默认为原文件名添加.onnx后缀')
    parser.add_argument('--save-config', action='store_true',
                      help='保存配置到同名YAML文件')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 加载PyTorch模型
    print(f"加载PyTorch模型: {args.checkpoint}")
    device = torch.device("cpu")  # 使用CPU进行转换
    
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        config = checkpoint["config"]
        
        # 创建模型
        model = get_model(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        # 确定输出路径
        if args.output:
            output_path = args.output
        else:
            output_path = os.path.splitext(args.checkpoint)[0] + ".onnx"
        
        # 获取可能的输入尺寸
        input_shape = [1, 100, 80]  # 批大小，最大帧数，特征维度
        if "feature" in config and "n_mels" in config["feature"]:
            input_shape[2] = config["feature"]["n_mels"]
        
        # 创建模拟输入
        dummy_input = torch.randn(*input_shape, requires_grad=True)
        
        # 导出为ONNX模型
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"模型已转换并保存到: {output_path}")
        
        # 保存配置文件
        if args.save_config:
            config_path = os.path.splitext(output_path)[0] + ".yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            print(f"配置已保存到: {config_path}")
        
    except Exception as e:
        print(f"转换过程中出错: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 
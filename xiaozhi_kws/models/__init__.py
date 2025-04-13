from .cnn_gru import CNNGRU

def get_model(config):
    """
    获取模型实例
    
    Args:
        config: 模型配置
        
    Returns:
        model: 模型实例
    """
    model_type = config["model"]["type"]
    
    if model_type == "cnn_gru":
        return CNNGRU(config)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

__all__ = ['CNNGRU', 'get_model'] 
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNGRU(nn.Module):
    """
    CNN-GRU 模型用于关键词识别
    使用CNN提取特征，GRU建模时序信息
    """
    
    def __init__(self, config):
        """
        初始化模型
        
        Args:
            config: 模型配置
        """
        super(CNNGRU, self).__init__()
        
        input_dim = config["model"]["input_dim"]
        hidden_dim = config["model"]["hidden_dim"]
        num_layers = config["model"]["num_layers"]
        dropout = config["model"]["dropout"]
        
        # CNN特征提取
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # GRU序列建模
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 输出层
        gru_output_dim = hidden_dim * 2  # 因为是双向GRU
        self.fc1 = nn.Linear(gru_output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # 二分类: 是/否关键词
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征, shape=(batch_size, time_steps, features)
            
        Returns:
            output: 模型输出
        """
        batch_size, time_steps, features = x.size()
        
        # 调整维度顺序为 (batch_size, features, time_steps)，适合CNN处理
        x = x.transpose(1, 2)
        
        # CNN特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 调整回 (batch_size, time_steps, features) 用于GRU
        x = x.transpose(1, 2)
        
        # GRU序列建模
        x, _ = self.gru(x)  # x shape: (batch_size, time_steps, hidden_dim*2)
        
        # 对时间维度取平均，得到全局表示
        x = torch.mean(x, dim=1)  # x shape: (batch_size, hidden_dim*2)
        
        # 输出层
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(x)
        
        return x
    
    def detect(self, x, threshold=0.5):
        """
        检测函数，用于推理
        
        Args:
            x: 输入特征
            threshold: 检测阈值
            
        Returns:
            is_keyword: 是否为关键词
            confidence: 置信度
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidence = probs[:, 1].item()  # 第二个类别的概率（关键词）
            is_keyword = confidence > threshold
            
        return is_keyword, confidence 
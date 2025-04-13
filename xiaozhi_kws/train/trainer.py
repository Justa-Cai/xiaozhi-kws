import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """模型训练器"""
    
    def __init__(self, model, config, train_loader, val_loader, device=None):
        """
        初始化训练器
        
        Args:
            model: 模型实例
            config: 训练配置
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 训练设备
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model = self.model.to(self.device)
        
        # 训练参数
        self.epochs = config["train"]["epochs"]
        self.learning_rate = config["train"]["learning_rate"]
        self.weight_decay = config["train"]["weight_decay"]
        self.patience = config["train"]["patience"]
        self.checkpoint_dir = config["train"]["checkpoint_dir"]
        
        # 创建检查点目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # TensorBoard日志
        self.writer = SummaryWriter(log_dir=os.path.join(self.checkpoint_dir, 'logs'))
        
        # 训练相关变量
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.early_stop_counter = 0
    
    def save_checkpoint(self, epoch, val_loss, val_acc, is_best=False):
        """
        保存检查点
        
        Args:
            epoch: 当前轮数
            val_loss: 验证损失
            val_acc: 验证准确率
            is_best: 是否是最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': self.config
        }
        
        # 保存最新检查点
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'checkpoint_last.pth'))
        
        # 如果是最佳模型，单独保存
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'model_best.pth'))
            print(f"√ 最佳模型已保存，验证准确率: {val_acc:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        if not os.path.exists(checkpoint_path):
            print(f"检查点不存在: {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['val_loss']
        self.best_val_acc = checkpoint.get('val_acc', 0.0)
        
        print(f"√ 加载检查点成功: {checkpoint_path}")
        print(f"  继续训练于第 {self.start_epoch} 轮")
        return True
    
    def train_epoch(self, epoch):
        """
        训练一个轮次
        
        Args:
            epoch: 当前轮数
            
        Returns:
            train_loss: 训练损失
            train_acc: 训练准确率
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")
        for batch_idx, (features, targets) in enumerate(pbar):
            features = features.to(self.device)
            # 处理targets张量的维度问题
            if targets.dim() > 1:
                targets = targets.to(self.device).squeeze(1)
            else:
                targets = targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        train_loss = total_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        
        return train_loss, train_acc
    
    def validate(self):
        """
        验证模型
        
        Returns:
            val_loss: 验证损失
            val_acc: 验证准确率
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, targets in self.val_loader:
                features = features.to(self.device)
                # 处理targets张量的维度问题
                if targets.dim() > 1:
                    targets = targets.to(self.device).squeeze(1)
                else:
                    targets = targets.to(self.device)
                
                # 前向传播
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                
                # 统计
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = total_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, resume_from=None):
        """
        训练模型
        
        Args:
            resume_from: 恢复训练的检查点路径
        """
        # 尝试恢复训练
        if resume_from:
            self.load_checkpoint(resume_from)
        
        print(f"训练设备: {self.device}")
        print(f"开始训练 {self.epochs} 轮...")
        
        # 训练历史记录
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # 计时
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.epochs):
            # 训练一个轮次
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 记录指标
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f"Epoch {epoch}/{self.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存检查点
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            self.save_checkpoint(epoch, val_loss, val_acc, is_best)
            
            # 早停检查
            if self.early_stop_counter >= self.patience:
                print(f"早停！验证准确率 {self.patience} 轮未提升")
                break
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"训练完成！总用时: {total_time:.2f}秒")
        print(f"最佳验证准确率: {self.best_val_acc:.2f}%")
        
        # 绘制训练曲线
        self._plot_training_curves(history)
        
        return history
    
    def _plot_training_curves(self, history):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 5))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Validation')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, 'training_curves.png'))
        plt.close() 
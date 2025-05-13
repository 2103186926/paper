#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bilstm_trainer2.py - 改进版BiLSTM模型训练工具 (PyTorch版本)

功能：训练高危操作分类模型（增强版）
- 添加L2正则化
- 增强Dropout策略
- 更多可调超参数
- 增强模型评估指标
- 使用GPU加速训练
"""

import os
import sys
import numpy as np
import json
import argparse
import pandas as pd
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, roc_curve, auc, 
                           precision_recall_curve, average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BiLSTMTrainer2-PyTorch")

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")

class BiLSTMModel(nn.Module):
    """PyTorch版本的BiLSTM模型"""
    
    def __init__(self, 
                input_dim: int,
                lstm_units: int = 128,
                num_lstm_layers: int = 2,
                dense_units: List[int] = [128, 64],
                dropout_rate: float = 0.5):
        """
        初始化BiLSTM模型
        
        参数:
            input_dim: 输入特征维度
            lstm_units: LSTM隐藏单元数量
            num_lstm_layers: LSTM层数
            dense_units: 全连接层的单元数列表
            dropout_rate: Dropout比率
        """
        super(BiLSTMModel, self).__init__()
        
        # 将输入重塑为序列
        self.reshape_dim = input_dim
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=1,  # 输入序列的特征维度
            hidden_size=lstm_units,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # 批归一化
        self.bn_lstm = nn.BatchNorm1d(lstm_units * 2)  # 因为是双向LSTM，所以是lstm_units*2
        
        # 构建全连接层
        fc_layers = []
        lstm_output_dim = lstm_units * 2  # 双向LSTM
        
        for units in dense_units:
            fc_layers.append(nn.Linear(lstm_output_dim, units))
            fc_layers.append(nn.BatchNorm1d(units))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout_rate))
            lstm_output_dim = units
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # 输出层
        self.output_layer = nn.Linear(lstm_output_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """前向传播"""
        # 重塑输入为序列形式 [batch_size, seq_len, features]
        x = x.view(-1, self.reshape_dim, 1)
        
        # 通过BiLSTM
        lstm_out, _ = self.lstm(x)
        
        # 只取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        
        # 批归一化
        lstm_out = self.bn_lstm(lstm_out)
        
        # 通过全连接层
        x = self.fc_layers(lstm_out)
        
        # 输出层
        x = self.output_layer(x)
        x = self.sigmoid(x)
        
        return x

class BiLSTMTrainer2:
    """改进版BiLSTM模型训练器 (PyTorch版本)"""
    
    def __init__(self, features_dir: str, labels_file: str):
        """
        初始化训练器
        
        参数:
            features_dir: 特征向量目录路径
            labels_file: 标签文件路径
        """
        self.features_dir = features_dir
        self.labels_file = labels_file
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self._load_data()
    
    def _load_data(self) -> None:
        """加载特征和标签数据"""
        logger.info(f"开始加载数据...")
        logger.info(f"特征目录: {self.features_dir}")
        logger.info(f"标签文件: {self.labels_file}")
        
        # 检查文件和目录是否存在
        if not os.path.exists(self.features_dir):
            logger.error(f"特征目录不存在: {self.features_dir}")
            sys.exit(1)
        
        if not os.path.exists(self.labels_file):
            logger.error(f"标签文件不存在: {self.labels_file}")
            sys.exit(1)
        
        # 加载标签文件
        try:
            labels_df = pd.read_csv(self.labels_file)
            logger.info(f"已加载标签文件: {self.labels_file}")
        except Exception as e:
            logger.error(f"加载标签文件时出错: {str(e)}")
            sys.exit(1)
        
        # 加载特征向量
        features_list = []
        filenames = []
        missing_files = []
        
        for filename in labels_df['filename']:
            feature_path = os.path.join(self.features_dir, f"{filename}_semantic.npy")
            if os.path.exists(feature_path):
                try:
                    feature = np.load(feature_path)
                    features_list.append(feature)
                    filenames.append(filename)
                except Exception as e:
                    logger.error(f"加载特征文件 {feature_path} 时出错: {str(e)}")
            else:
                missing_files.append(feature_path)
                logger.warning(f"找不到特征文件: {feature_path}")
        
        if missing_files:
            logger.warning(f"共有 {len(missing_files)} 个特征文件缺失")
        
        # 过滤标签数据，只保留有对应特征的样本
        filtered_labels = labels_df[labels_df['filename'].isin(filenames)]
        
        # 检查数据是否为空
        if len(features_list) == 0:
            logger.error("错误: 没有加载到任何特征数据")
            sys.exit(1)
        
        # 将特征列表转换为numpy数组
        X = np.array(features_list)
        y = np.array(filtered_labels['is_dangerous'].astype(int))
        
        logger.info(f"加载了 {len(X)} 个样本")
        logger.info(f"特征维度: {X.shape}")
        logger.info(f"正样本数量: {sum(y)}, 负样本数量: {len(y) - sum(y)}")
        
        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"训练集大小: {len(self.X_train)}, 测试集大小: {len(self.X_test)}")
        
        # 转换为PyTorch张量
        self.X_train = torch.FloatTensor(self.X_train)
        self.X_test = torch.FloatTensor(self.X_test)
        self.y_train = torch.FloatTensor(self.y_train.reshape(-1, 1))
        self.y_test = torch.FloatTensor(self.y_test.reshape(-1, 1))
        
        logger.info("数据已转换为PyTorch张量")
    
    def build_model(self, 
                   lstm_units: int = 128,
                   num_lstm_layers: int = 2,
                   dense_units: List[int] = [128, 64],
                   dropout_rate: float = 0.5) -> None:
        """
        构建增强版BiLSTM模型 (PyTorch版本)
        
        参数:
            lstm_units: LSTM单元数量
            num_lstm_layers: LSTM层数
            dense_units: 全连接层的单元数列表
            dropout_rate: Dropout比率
        """
        try:
            input_dim = self.X_train.shape[1]
            
            # 创建模型
            self.model = BiLSTMModel(
                input_dim=input_dim,
                lstm_units=lstm_units,
                num_lstm_layers=num_lstm_layers,
                dense_units=dense_units,
                dropout_rate=dropout_rate
            )
            
            # 将模型移至GPU
            self.model = self.model.to(device)
            
            # 打印模型结构
            logger.info(f"模型结构:\n{self.model}")
            
            # 计算模型参数量
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"模型总参数量: {total_params}")
            logger.info(f"可训练参数量: {trainable_params}")
            
            logger.info("增强版PyTorch模型构建完成")
            
        except Exception as e:
            logger.error(f"构建模型时出错: {str(e)}")
            raise
    
    def plot_training_history(self, history: Dict[str, List[float]], save_dir: str) -> None:
        """绘制训练历史曲线"""
        plt.figure(figsize=(12, 4))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='训练损失')
        plt.plot(history['val_loss'], label='验证损失')
        plt.title('模型损失曲线', fontproperties='SimHei')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(prop={'family': 'SimHei'})
        
        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='训练准确率')
        plt.plot(history['val_acc'], label='验证准确率')
        plt.title('模型准确率曲线', fontproperties='SimHei')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(prop={'family': 'SimHei'})
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'))
        plt.close()
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_dir: str) -> None:
        """绘制混淆矩阵热力图"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵', fontproperties='SimHei')
        plt.ylabel('真实标签', fontproperties='SimHei')
        plt.xlabel('预测标签', fontproperties='SimHei')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_prob: np.ndarray, save_dir: str) -> float:
        """绘制ROC曲线"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正例率', fontproperties='SimHei')
        plt.ylabel('真正例率', fontproperties='SimHei')
        plt.title('接收者操作特征(ROC)曲线', fontproperties='SimHei')
        plt.legend(loc="lower right", prop={'family': 'SimHei'})
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
        plt.close()
        
        return roc_auc
    
    def plot_pr_curve(self, y_true: np.ndarray, y_pred_prob: np.ndarray, save_dir: str) -> float:
        """绘制PR曲线"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
        pr_auc = average_precision_score(y_true, y_pred_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                 label=f'PR曲线 (AP = {pr_auc:.2f})')
        plt.xlabel('召回率', fontproperties='SimHei')
        plt.ylabel('精确率', fontproperties='SimHei')
        plt.title('精确率-召回率曲线', fontproperties='SimHei')
        plt.legend(loc="lower left", prop={'family': 'SimHei'})
        plt.savefig(os.path.join(save_dir, 'pr_curve.png'))
        plt.close()
        
        return pr_auc
    
    def train(self, 
             epochs: int = 100,
             batch_size: int = 32,
             learning_rate: float = 0.001,
             weight_decay: float = 0.01,
             model_path: str = 'model.pt',
             plots_dir: str = 'plots') -> Dict[str, Any]:
        """
        训练模型并生成评估指标 (PyTorch版本)
        
        参数:
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            weight_decay: L2正则化系数
            model_path: 模型保存路径
            plots_dir: 图表保存目录
        """
        if self.model is None:
            self.build_model()
        
        # 创建图表保存目录
        os.makedirs(plots_dir, exist_ok=True)
        
        try:
            # 定义损失函数和优化器
            criterion = nn.BCELoss()
            optimizer = optim.Adam(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
            
            # 学习率调度器
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=10, 
                verbose=True
            )
            
            # 创建数据加载器
            train_dataset = TensorDataset(self.X_train, self.y_train)
            val_split = int(0.2 * len(train_dataset))
            train_subset, val_subset = torch.utils.data.random_split(
                train_dataset, 
                [len(train_dataset) - val_split, val_split]
            )
            
            train_loader = DataLoader(
                train_subset, 
                batch_size=batch_size, 
                shuffle=True
            )
            
            val_loader = DataLoader(
                val_subset, 
                batch_size=batch_size, 
                shuffle=False
            )
            
            # 训练历史记录
            history = {
                'train_loss': [],
                'val_loss': [],
                'train_acc': [],
                'val_acc': []
            }
            
            # 最佳模型跟踪
            best_val_loss = float('inf')
            best_model_state = None
            
            # 早停计数器
            early_stop_counter = 0
            early_stop_patience = 5
            
            logger.info("开始训练模型...")
            
            # 训练循环
            for epoch in range(epochs):
                # 训练模式
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for inputs, targets in train_loader:
                    # 将数据移至GPU
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # 梯度清零
                    optimizer.zero_grad()
                    
                    # 前向传播
                    outputs = self.model(inputs)
                    
                    # 计算损失
                    loss = criterion(outputs, targets)
                    
                    # 反向传播
                    loss.backward()
                    
                    # 更新权重
                    optimizer.step()
                    
                    # 统计
                    train_loss += loss.item() * inputs.size(0)
                    predicted = (outputs >= 0.5).float()
                    train_correct += (predicted == targets).sum().item()
                    train_total += targets.size(0)
                
                train_loss = train_loss / train_total
                train_acc = train_correct / train_total
                
                # 验证模式
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        # 将数据移至GPU
                        inputs, targets = inputs.to(device), targets.to(device)
                        
                        # 前向传播
                        outputs = self.model(inputs)
                        
                        # 计算损失
                        loss = criterion(outputs, targets)
                        
                        # 统计
                        val_loss += loss.item() * inputs.size(0)
                        predicted = (outputs >= 0.5).float()
                        val_correct += (predicted == targets).sum().item()
                        val_total += targets.size(0)
                
                val_loss = val_loss / val_total
                val_acc = val_correct / val_total
                
                # 更新学习率
                scheduler.step(val_loss)
                
                # 记录训练历史
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                
                # 打印训练信息
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"Val Acc: {val_acc:.4f}")
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                    torch.save(self.model.state_dict(), model_path)
                    early_stop_counter = 0
                    logger.info(f"已保存新的最佳模型，验证损失: {val_loss:.4f}")
                else:
                    early_stop_counter += 1
                
                # 早停检查
                if early_stop_counter >= early_stop_patience:
                    logger.info(f"早停: 验证损失在 {early_stop_patience} 轮内没有改善")
                    break
            
            logger.info("模型训练完成")
            
            # 加载最佳模型
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
            
            # 在测试集上评估模型
            self.model.eval()
            with torch.no_grad():
                # 将测试数据移至GPU
                X_test_gpu = self.X_test.to(device)
                y_pred_prob = self.model(X_test_gpu).cpu().numpy()
            
            y_pred = (y_pred_prob >= 0.5).astype(int)
            y_true = self.y_test.numpy()
            
            # 计算评估指标
            cm = confusion_matrix(y_true, y_pred)
            metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, zero_division=0)),
                'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
                'confusion_matrix': cm.tolist()
            }
            
            # 生成评估图表
            self.plot_training_history(history, plots_dir)
            self.plot_confusion_matrix(cm, plots_dir)
            metrics['roc_auc'] = float(self.plot_roc_curve(y_true, y_pred_prob, plots_dir))
            metrics['pr_auc'] = float(self.plot_pr_curve(y_true, y_pred_prob, plots_dir))
            
            # 输出评估指标
            logger.info("\n评估指标:")
            logger.info(f"准确率: {metrics['accuracy']:.4f}")
            logger.info(f"精确率: {metrics['precision']:.4f}")
            logger.info(f"召回率: {metrics['recall']:.4f}")
            logger.info(f"F1分数: {metrics['f1_score']:.4f}")
            logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"PR AUC: {metrics['pr_auc']:.4f}")
            
            return {
                'history': {
                    'train_loss': history['train_loss'],
                    'val_loss': history['val_loss'],
                    'train_acc': history['train_acc'],
                    'val_acc': history['val_acc']
                },
                'metrics': metrics
            }
        
        except Exception as e:
            logger.error(f"训练模型时出错: {str(e)}")
            raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练改进版高危操作分类模型 (PyTorch版本)")
    parser.add_argument("features_dir", help="特征向量目录路径")
    parser.add_argument("labels_file", help="标签文件路径")
    parser.add_argument("-m", "--model", default="model.pt", help="模型保存路径")
    parser.add_argument("-o", "--output", default="metrics.json", help="指标文件路径")
    parser.add_argument("-p", "--plots", default="plots", help="图表保存目录")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="学习率")
    parser.add_argument("-w", "--weight_decay", type=float, default=0.01, help="权重衰减系数")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细日志")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        trainer = BiLSTMTrainer2(args.features_dir, args.labels_file)
        trainer.build_model()
        metrics = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            model_path=args.model,
            plots_dir=args.plots
        )
        
        # 保存评估指标
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"模型已保存至: {args.model}")
        logger.info(f"评估指标已保存至: {args.output}")
        logger.info(f"评估图表已保存至: {args.plots}目录")
        
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
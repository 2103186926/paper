#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
feature_integration.py - 特征融合工具

功能：融合不同来源的特征向量（如语义特征和行为特征）
输入：A特征向量和B特征向量
输出：融合后的特征向量（.npy文件）
"""

import numpy as np
import os
import argparse
import logging
from typing import List, Optional, Tuple, Union
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print(f"调试器正在监听端口9501... 等待客户端连接...")
    debugpy.wait_for_client()
    print("调试器已连接!")
except Exception as e:
    # 打印详细错误信息
    logger.error(f"启动debugpy监听失败: {e}", exc_info=True)
    sys.exit(1) # 如果希望失败时退出
    # pass # 或者保持现状，允许脚本继续（没有调试器连接）

class FeatureIntegrator:
    """特征融合器，用于融合不同来源的特征向量"""
    
    def __init__(self):
        """初始化特征融合器"""
        pass
    
    @staticmethod
    def load_feature_vector(file_path: str) -> np.ndarray:
        """
        加载特征向量
        
        参数:
            file_path: 特征向量文件路径(.npy)
            
        返回:
            加载的特征向量
        """
        try:
            features = np.load(file_path)
            logger.info(f"已加载特征向量: {file_path}, 形状: {features.shape}")
            return features
        except Exception as e:
            logger.error(f"加载特征向量失败: {file_path}, 错误: {str(e)}")
            raise
    
    @staticmethod
    def concat_features(feature_vectors: List[np.ndarray]) -> np.ndarray:
        """
        简单拼接多个特征向量
        
        参数:
            feature_vectors: 特征向量列表
            
        返回:
            拼接后的特征向量
        """
        try:
            # 确保所有特征向量是二维的
            normalized_vectors = []
            for vector in feature_vectors:
                if vector.ndim == 1:
                    normalized_vectors.append(vector.reshape(1, -1))
                else:
                    normalized_vectors.append(vector)
            
            # 沿着特征维度(axis=1)拼接
            result = np.concatenate(normalized_vectors, axis=1)
            logger.info(f"已拼接特征向量, 形状: {result.shape}")
            return result
        except Exception as e:
            logger.error(f"拼接特征向量失败: {str(e)}")
            raise
    
    @staticmethod
    def average_features(feature_vectors: List[np.ndarray]) -> np.ndarray:
        """
        计算多个特征向量的平均值
        
        参数:
            feature_vectors: 特征向量列表
            
        返回:
            平均后的特征向量
        """
        try:
            # 确保所有特征向量有相同的维度
            shapes = [vector.shape for vector in feature_vectors]
            if len(set(shapes)) > 1:
                raise ValueError(f"特征向量形状不一致: {shapes}")
            
            # 计算平均值
            result = np.mean(feature_vectors, axis=0)
            logger.info(f"已计算特征向量平均值, 形状: {result.shape}")
            return result
        except Exception as e:
            logger.error(f"计算特征向量平均值失败: {str(e)}")
            raise
    
    @staticmethod
    def weighted_combine(feature_vectors: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """
        加权组合多个特征向量
        
        参数:
            feature_vectors: 特征向量列表
            weights: 对应的权重列表
            
        返回:
            加权组合后的特征向量
        """
        try:
            # 检查权重数量与特征向量数量是否相等
            if len(feature_vectors) != len(weights):
                raise ValueError(f"特征向量数量({len(feature_vectors)})与权重数量({len(weights)})不匹配")
            
            # 检查权重之和是否为1
            if abs(sum(weights) - 1.0) > 1e-6:
                logger.warning(f"权重之和不为1: {sum(weights)}, 将自动归一化")
                weights = [w / sum(weights) for w in weights]
            
            # 确保所有特征向量维度一致
            shapes = [vector.shape for vector in feature_vectors]
            if len(set(shapes)) > 1:
                raise ValueError(f"特征向量形状不一致: {shapes}")
            
            # 计算加权组合
            result = np.zeros_like(feature_vectors[0])
            for vector, weight in zip(feature_vectors, weights):
                result += weight * vector
            
            logger.info(f"已计算特征向量加权组合, 形状: {result.shape}, 权重: {weights}")
            return result
        except Exception as e:
            logger.error(f"计算特征向量加权组合失败: {str(e)}")
            raise
    
    @staticmethod
    def normalize_feature(feature_vector: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        归一化特征向量
        
        参数:
            feature_vector: 要归一化的特征向量
            method: 归一化方法，'minmax'或'zscore'
            
        返回:
            归一化后的特征向量
        """
        try:
            if method == 'minmax':
                # Min-Max归一化到[0,1]区间
                min_val = np.min(feature_vector, axis=0)
                max_val = np.max(feature_vector, axis=0)
                denominator = max_val - min_val
                # 避免除以零
                denominator[denominator == 0] = 1
                normalized = (feature_vector - min_val) / denominator
            elif method == 'zscore':
                # Z-score标准化（均值为0，标准差为1）
                mean = np.mean(feature_vector, axis=0)
                std = np.std(feature_vector, axis=0)
                # 避免除以零
                std[std == 0] = 1
                normalized = (feature_vector - mean) / std
            else:
                raise ValueError(f"不支持的归一化方法: {method}")
            
            logger.info(f"已使用{method}方法归一化特征向量")
            return normalized
        except Exception as e:
            logger.error(f"归一化特征向量失败: {str(e)}")
            raise
    
    def integrate_features(self, 
                          feature_vector_a: np.ndarray, 
                          feature_vector_b: np.ndarray,
                          method: str = 'concat',
                          weights: List[float] = None) -> np.ndarray:
        """
        集成两个特征向量
        
        参数:
            feature_vector_a: 第一个特征向量
            feature_vector_b: 第二个特征向量
            method: 集成方法，'concat'、'average'或'weighted'
            weights: 如果method为'weighted'，则需提供权重列表
            
        返回:
            集成后的特征向量
        """
        try:
            if method == 'concat':
                # 确保是一维向量
                if feature_vector_a.ndim > 1 and feature_vector_a.shape[0] == 1:
                    feature_vector_a = feature_vector_a.flatten()
                if feature_vector_b.ndim > 1 and feature_vector_b.shape[0] == 1:
                    feature_vector_b = feature_vector_b.flatten()
                
                # 如果是一维向量，直接拼接
                if feature_vector_a.ndim == 1 and feature_vector_b.ndim == 1:
                    result = np.concatenate([feature_vector_a, feature_vector_b])
                else:
                    # 否则使用concat_features方法
                    result = self.concat_features([feature_vector_a, feature_vector_b])
            elif method == 'average':
                result = self.average_features([feature_vector_a, feature_vector_b])
            elif method == 'weighted':
                if weights is None:
                    weights = [0.5, 0.5]  # 默认权重
                result = self.weighted_combine([feature_vector_a, feature_vector_b], weights)
            else:
                raise ValueError(f"不支持的集成方法: {method}")
            
            logger.info(f"已使用{method}方法集成特征向量, 结果形状: {result.shape}")
            return result
        except Exception as e:
            logger.error(f"集成特征向量失败: {str(e)}")
            raise
    
    def save_features(self, feature_vector: np.ndarray, output_file: str) -> None:
        """
        保存特征向量到文件
        
        参数:
            feature_vector: 要保存的特征向量
            output_file: 输出文件路径
        """
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"创建输出目录: {output_dir}")
            
            # 检查文件是否已存在
            if os.path.exists(output_file):
                logger.warning(f"文件 {output_file} 已存在，将被覆盖")
            
            # 保存特征向量
            np.save(output_file, feature_vector)
            logger.info(f"特征向量已保存至: {output_file}, 形状: {feature_vector.shape}")
        except Exception as e:
            logger.error(f"保存特征向量失败: {str(e)}")
            raise


def process_file_pair(semantic_feature_file: str, 
                     behavior_feature_file: str, 
                     output_file: Optional[str] = None,
                     method: str = 'concat',
                     weights: List[float] = None) -> None:
    """
    处理一对特征文件并集成它们
    
    参数:
        semantic_feature_file: 语义特征文件路径
        behavior_feature_file: 行为特征文件路径
        output_file: 输出文件路径（可选）
        method: 集成方法
        weights: 权重列表（用于weighted方法）
    """
    integrator = FeatureIntegrator()
    
    # 加载特征向量
    semantic_features = integrator.load_feature_vector(semantic_feature_file)
    behavior_features = integrator.load_feature_vector(behavior_feature_file)
    
    # 集成特征
    integrated_features = integrator.integrate_features(
        semantic_features, behavior_features, method, weights
    )
    
    # 如果未指定输出文件名，生成默认文件名
    if output_file is None:
        semantic_base = Path(semantic_feature_file).stem
        behavior_base = Path(behavior_feature_file).stem
        output_file = f"{semantic_base}_{behavior_base}_integrated.npy"
    
    # 保存集成后的特征
    integrator.save_features(integrated_features, output_file)
    logger.info(f"已完成特征集成: {output_file}")


def process_directory_pair(semantic_dir: str, 
                          behavior_dir: str, 
                          output_dir: Optional[str] = None,
                          method: str = 'concat',
                          weights: List[float] = None,
                          pattern_match: bool = True) -> None:
    """
    处理两个目录中的特征文件并集成它们
    
    参数:
        semantic_dir: 语义特征目录路径
        behavior_dir: 行为特征目录路径
        output_dir: 输出目录路径（可选）
        method: 集成方法
        weights: 权重列表（用于weighted方法）
        pattern_match: 是否要求文件名匹配
    """
    # 导入进度条库
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        logger.info("提示: 安装 tqdm 库可以获得更好的进度展示效果 (pip install tqdm)")
    
    # 确定输出目录
    if output_dir is None:
        output_dir = "integrated_features"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"将输出集成特征向量保存到目录: {output_dir}")
    
    # 获取两个目录中的所有特征文件
    semantic_files = [f for f in os.listdir(semantic_dir) if f.endswith('.npy')]
    behavior_files = [f for f in os.listdir(behavior_dir) if f.endswith('.npy')]
    
    if not semantic_files:
        logger.warning(f"在目录 {semantic_dir} 中没有找到特征文件")
        return
    
    if not behavior_files:
        logger.warning(f"在目录 {behavior_dir} 中没有找到特征文件")
        return
    
    # 创建特征集成器
    integrator = FeatureIntegrator()
    
    # 处理文件对
    file_pairs = []
    
    if pattern_match:
        # 尝试匹配相似的文件名
        semantic_bases = [Path(f).stem for f in semantic_files]
        behavior_bases = [Path(f).stem for f in behavior_files]
        
        for s_file in semantic_files:
            s_base = Path(s_file).stem
            
            # 尝试找到匹配的行为特征文件
            matched = False
            for b_file in behavior_files:
                b_base = Path(b_file).stem
                
                # 检查文件名是否相似（可自定义匹配规则）
                # 这里简单检查两个文件名是否有公共部分
                if (s_base in b_base) or (b_base in s_base) or \
                   (s_base.split('_')[0] == b_base.split('_')[0]):
                    file_pairs.append((
                        os.path.join(semantic_dir, s_file),
                        os.path.join(behavior_dir, b_file),
                        os.path.join(output_dir, f"{s_base}_{b_base}_integrated.npy")
                    ))
                    matched = True
                    break
            
            if not matched:
                logger.warning(f"未找到与 {s_file} 匹配的行为特征文件")
    else:
        # 交叉匹配所有文件（笛卡尔积）
        for s_file in semantic_files:
            for b_file in behavior_files:
                s_base = Path(s_file).stem
                b_base = Path(b_file).stem
                file_pairs.append((
                    os.path.join(semantic_dir, s_file),
                    os.path.join(behavior_dir, b_file),
                    os.path.join(output_dir, f"{s_base}_{b_base}_integrated.npy")
                ))
    
    logger.info(f"找到 {len(file_pairs)} 对特征文件需要集成")
    
    # 使用进度条包装文件对列表
    if has_tqdm:
        file_pairs_iter = tqdm(file_pairs, desc="集成特征文件", unit="对")
    else:
        file_pairs_iter = file_pairs
        # 简单的进度显示
        logger.info("开始集成特征文件:")
    
    # 成功和失败的计数器
    success_count = 0
    failure_count = 0
    
    # 处理每对特征文件
    for i, (s_file, b_file, out_file) in enumerate(file_pairs_iter, 1):
        # 如果没有tqdm, 显示简单进度
        if not has_tqdm:
            progress_percent = (i / len(file_pairs)) * 100
            logger.info(f"[{i}/{len(file_pairs)}] {progress_percent:.1f}% - 处理: {Path(s_file).name} + {Path(b_file).name}")
        
        try:
            # 加载特征向量
            semantic_features = integrator.load_feature_vector(s_file)
            behavior_features = integrator.load_feature_vector(b_file)
            
            # 集成特征
            integrated_features = integrator.integrate_features(
                semantic_features, behavior_features, method, weights
            )
            
            # 保存集成后的特征
            integrator.save_features(integrated_features, out_file)
            
            success_count += 1
            if not has_tqdm:
                logger.info(f"✓ 成功集成: {Path(out_file).name}")
        except Exception as e:
            failure_count += 1
            if has_tqdm:
                tqdm.write(f"集成特征 {s_file} + {b_file} 时出错: {str(e)}")
            else:
                logger.error(f"✗ 集成特征 {s_file} + {b_file} 失败: {str(e)}")
            # 继续处理下一对文件，不中断整个流程
    
    # 处理完成后的摘要
    logger.info(f"\n处理完成摘要:")
    logger.info(f"- 总文件对数: {len(file_pairs)}")
    logger.info(f"- 成功集成: {success_count}")
    logger.info(f"- 集成失败: {failure_count}")
    logger.info(f"集成特征向量已保存到目录: {output_dir}")


def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description="融合不同来源的特征向量")
    
    # 主要参数组（文件或目录）
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--files", nargs=2, metavar=('SEMANTIC_FILE', 'BEHAVIOR_FILE'),
                      help="要集成的两个特征文件路径")
    group.add_argument("--dirs", nargs=2, metavar=('SEMANTIC_DIR', 'BEHAVIOR_DIR'),
                      help="包含特征文件的两个目录路径")
    
    # 输出路径
    parser.add_argument("-o", "--output", 
                      help="输出文件或目录路径（可选）")
    
    # 集成方法
    parser.add_argument("-m", "--method", choices=['concat', 'average', 'weighted'],
                      default='concat', help="特征集成方法 (默认: concat)")
    
    # 权重（用于weighted方法）
    parser.add_argument("-w", "--weights", nargs=2, type=float, metavar=('W1', 'W2'),
                      default=[0.5, 0.5], help="加权方法的权重 (默认: 0.5 0.5)")
    
    # 目录模式下的文件匹配选项
    parser.add_argument("--no-match", action="store_true",
                      help="不尝试匹配文件名，而是集成所有可能的组合")
    
    # 添加进度条相关选项
    parser.add_argument("--no-progress", action="store_true", 
                      help="禁用进度条显示")
    
    args = parser.parse_args()
    
    # 如果指定禁用进度条，设置环境变量
    if args.no_progress:
        os.environ["TQDM_DISABLE"] = "1"
    
    if args.files:
        # 处理单对文件
        semantic_file, behavior_file = args.files
        process_file_pair(
            semantic_file, behavior_file, args.output, 
            args.method, args.weights
        )
    else:
        # 处理目录
        semantic_dir, behavior_dir = args.dirs
        process_directory_pair(
            semantic_dir, behavior_dir, args.output,
            args.method, args.weights, not args.no_match
        )


if __name__ == "__main__":
    main()

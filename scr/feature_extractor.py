#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
feature_extractor.py - 时空网格数据恶意操作特征提取工具

功能：从用户日志文件中提取各种恶意行为特征并生成特征向量
输入：用户日志文件(.log)
输出：128维特征向量(.npy文件)
"""

import json
import os
import sys
import numpy as np
import argparse
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from gensim.models import Word2Vec
import re
from pathlib import Path
from scipy import stats
import math


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print(f"调试器正在监听端口9501... 等待客户端连接...")
#     debugpy.wait_for_client()
#     print("调试器已连接!")
# except Exception as e:
#     # 打印详细错误信息
#     logger.error(f"启动debugpy监听失败: {e}", exc_info=True)
#     sys.exit(1) # 如果希望失败时退出
#     # pass # 或者保持现状，允许脚本继续（没有调试器连接）

class FeatureExtractor:
    """时空网格数据恶意行为特征提取器"""
    
    def __init__(self, log_file: str):
        """
        初始化特征提取器
        
        参数:
            log_file: 用户日志文件路径
        """
        self.log_file = log_file
        self.api_sequences = []  # 存储API调用序列
        self.sessions = {}  # 按会话ID组织的API调用
        
        # 读取日志文件
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                self.log_data = [json.loads(line) for line in f if line.strip()]
            logger.info(f"已读取日志文件: {log_file}, 包含 {len(self.log_data)} 条记录")
        except Exception as e:
            logger.error(f"读取日志文件时出错: {str(e)}")
            sys.exit(1)
        
        # 解析日志数据为DataFrame以便于分析
        self._parse_log_data()
        
        # 提取API调用序列
        self._extract_api_sequences()
    
    def _parse_log_data(self) -> None:
        """将日志数据解析为DataFrame以便于分析"""
        try:
            # 创建基本的DataFrame
            self.df = pd.DataFrame(self.log_data)
            
            # 转换时间戳为datetime对象
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            
            # 提取params字段中的子字段
            if 'params' in self.df.columns:
                # 提取常见的params子字段
                params_df = pd.json_normalize(self.df['params'].apply(lambda x: x if isinstance(x, dict) else {}))
                
                # 如果params_df不为空，将其合并到主DataFrame
                if not params_df.empty:
                    # 重命名列以避免冲突
                    params_columns = {col: f"params_{col}" for col in params_df.columns}
                    params_df = params_df.rename(columns=params_columns)
                    
                    # 将索引重置以确保合并正确
                    self.df = self.df.reset_index(drop=True)
                    params_df = params_df.reset_index(drop=True)
                    
                    # 合并数据框
                    self.df = pd.concat([self.df, params_df], axis=1)
            
            logger.info(f"已将日志数据解析为DataFrame，包含 {len(self.df)} 行和 {len(self.df.columns)} 列")
        except Exception as e:
            logger.error(f"解析日志数据时出错: {str(e)}")
            self.df = pd.DataFrame(self.log_data)  # 如果出错，使用原始数据
        
        if not self.log_data:
            logger.warning(f"日志文件为空或格式不正确: {self.log_file}")
            self.df = pd.DataFrame()
            return
    
    def _extract_api_sequences(self) -> None:
        """提取API调用序列，并按会话ID和序列号排序"""
        # 按会话ID组织API调用
        for entry in self.log_data:
            session_id = entry.get('session_id')
            if not session_id:
                continue
                
            if session_id not in self.sessions:
                self.sessions[session_id] = []
                
            self.sessions[session_id].append(entry)
        
        # 对每个会话按序列号排序
        for session_id, calls in self.sessions.items():
            sorted_calls = sorted(calls, key=lambda x: x.get('sequence_num', 0))
            
            # 提取API名称序列
            api_sequence = [call.get('api', '') for call in sorted_calls if call.get('api')]
            if api_sequence:
                self.api_sequences.append(api_sequence)
        
        logger.info(f"已提取 {len(self.sessions)} 个会话的API调用序列")
    
    def extract_features(self) -> np.ndarray:
        """
        提取全面的特征向量
        
        返回:
            128维特征向量
        """
        feature_vectors = []
        
        # 1. API序列特征 (32维)
        api_sequence_features = self._extract_api_sequence_features()
        feature_vectors.append(api_sequence_features)
        
        # 2. 时间行为特征 (16维)
        temporal_features = self._extract_temporal_features()
        feature_vectors.append(temporal_features)
        
        # 3. 空间行为特征 (16维)
        spatial_features = self._extract_spatial_features()
        feature_vectors.append(spatial_features)
        
        # 4. 数据量特征 (16维)
        data_volume_features = self._extract_data_volume_features()
        feature_vectors.append(data_volume_features)
        
        # 5. 会话行为特征 (16维)
        session_features = self._extract_session_features()
        feature_vectors.append(session_features)
        
        # 6. IP与设备特征 (16维)
        ip_device_features = self._extract_ip_device_features()
        feature_vectors.append(ip_device_features)
        
        # 7. 访问模式特征 (16维)
        pattern_features = self._extract_pattern_features()
        feature_vectors.append(pattern_features)
        
        # 合并所有特征向量
        combined_features = np.concatenate(feature_vectors)
        
        # 如果特征维度不是128，进行截断或填充
        if len(combined_features) < 128:
            combined_features = np.pad(combined_features, (0, 128 - len(combined_features)))
        elif len(combined_features) > 128:
            combined_features = combined_features[:128]
        
        return combined_features
    
    def _extract_api_sequence_features(self) -> np.ndarray:
        """
        提取API调用序列特征
        
        返回:
            32维API序列特征向量
        """
        # 如果没有API序列，返回零向量
        if not self.api_sequences:
            logger.warning("没有API序列可用，返回零特征向量")
            return np.zeros(32)
        
        try:
            # 1. API频率特征（8维）
            api_counts = Counter([api for seq in self.api_sequences for api in seq])
            top_apis = api_counts.most_common(8)
            api_freq_features = np.zeros(8)
            
            for i, (api, count) in enumerate(top_apis):
                if i < 8:
                    api_freq_features[i] = count / max(sum(api_counts.values()), 1)
            
            # 2. API序列模式特征（8维）
            api_bigrams = []
            for seq in self.api_sequences:
                for i in range(len(seq) - 1):
                    api_bigrams.append((seq[i], seq[i+1]))
            
            bigram_counts = Counter(api_bigrams)
            top_bigrams = bigram_counts.most_common(8)
            bigram_features = np.zeros(8)
            
            for i, ((api1, api2), count) in enumerate(top_bigrams):
                if i < 8:
                    bigram_features[i] = count / max(sum(bigram_counts.values()), 1)
            
            # 3. API调用多样性特征（8维）
            diversity_features = np.zeros(8)
            
            # 计算唯一API比例
            unique_apis = len(set([api for seq in self.api_sequences for api in seq]))
            total_apis = sum(len(seq) for seq in self.api_sequences)
            diversity_features[0] = unique_apis / max(total_apis, 1)
            
            # 计算序列中的唯一API比例
            seq_diversity = [len(set(seq)) / max(len(seq), 1) for seq in self.api_sequences]
            diversity_features[1] = np.mean(seq_diversity) if seq_diversity else 0
            
            # 计算会话间API差异
            if len(self.api_sequences) > 1:
                session_apis = [set(seq) for seq in self.api_sequences]
                session_differences = []
                for i in range(len(session_apis)):
                    for j in range(i+1, len(session_apis)):
                        jaccard = len(session_apis[i].intersection(session_apis[j])) / max(len(session_apis[i].union(session_apis[j])), 1)
                        session_differences.append(jaccard)
                diversity_features[2] = np.mean(session_differences) if session_differences else 0
            
            # API类型比例
            query_apis = sum(1 for seq in self.api_sequences for api in seq if 'query' in api.lower())
            export_apis = sum(1 for seq in self.api_sequences for api in seq if 'export' in api.lower())
            visualization_apis = sum(1 for seq in self.api_sequences for api in seq if 'visualization' in api.lower())
            
            diversity_features[3] = query_apis / max(total_apis, 1)
            diversity_features[4] = export_apis / max(total_apis, 1)
            diversity_features[5] = visualization_apis / max(total_apis, 1)
            
            # 异常API比例（非常见API的比例）
            common_apis = {'query_spatial_grid_data', 'query_temporal_grid_data', 'export_grid_data', 
                          'visualize_grid_data', 'query_historical_grid_data', 'query_future_grid_data'}
            uncommon_apis = sum(1 for seq in self.api_sequences for api in seq if api not in common_apis)
            diversity_features[6] = uncommon_apis / max(total_apis, 1)
            
            # 最大连续相同API调用长度与总序列长度的比例
            max_repeat_ratio = []
            for seq in self.api_sequences:
                if not seq:
                    continue
                max_repeat = 1
                current_repeat = 1
                for i in range(1, len(seq)):
                    if seq[i] == seq[i-1]:
                        current_repeat += 1
                    else:
                        max_repeat = max(max_repeat, current_repeat)
                        current_repeat = 1
                max_repeat = max(max_repeat, current_repeat)
                max_repeat_ratio.append(max_repeat / len(seq))
            
            diversity_features[7] = np.mean(max_repeat_ratio) if max_repeat_ratio else 0
            
            # 4. Word2Vec特征（8维）
            w2v_features = np.zeros(8)
            try:
                # 尝试训练Word2Vec模型
                if len(self.api_sequences) > 2:  # 确保有足够的序列进行训练
                    model = Word2Vec(sentences=self.api_sequences, vector_size=8, window=3, 
                                    min_count=1, workers=4, sg=1, epochs=100)
                    
                    # 计算所有API调用的平均向量作为特征
                    api_vectors = []
                    unique_apis_set = set(api for seq in self.api_sequences for api in seq)
                    
                    for api in unique_apis_set:
                        if api in model.wv:
                            api_vectors.append(model.wv[api])
                    
                    if api_vectors:
                        w2v_features = np.mean(api_vectors, axis=0)
            except Exception as e:
                logger.warning(f"Word2Vec特征提取失败: {str(e)}")
            
            # 合并所有API序列特征
            api_features = np.concatenate([api_freq_features, bigram_features, diversity_features, w2v_features])
            
            return api_features
        except Exception as e:
            logger.error(f"提取API序列特征出错: {str(e)}")
            return np.zeros(32)
    
    def _extract_temporal_features(self) -> np.ndarray:
        """
        提取时间行为特征
        
        返回:
            16维时间行为特征向量
        """
        try:
            features = np.zeros(16)
            
            # 如果DataFrame为空或没有时间戳列，返回零向量
            if self.df.empty or 'timestamp' not in self.df.columns:
                return features
            
            # 1. API调用频率特征
            # 计算时间差（秒）
            self.df['time_diff'] = self.df['timestamp'].diff().dt.total_seconds()
            
            # 平均调用间隔（秒）
            avg_interval = self.df['time_diff'].dropna().mean()
            features[0] = min(avg_interval / 3600, 1)  # 标准化，最大1小时
            
            # 最小调用间隔（秒）
            min_interval = self.df['time_diff'].dropna().min()
            features[1] = min(min_interval / 60, 1)  # 标准化，最大1分钟
            
            # 调用间隔标准差
            interval_std = self.df['time_diff'].dropna().std()
            features[2] = min(interval_std / 3600, 1)  # 标准化，最大1小时
            
            # 调用间隔的变异系数（标准差/平均值）
            if avg_interval > 0:
                features[3] = min(interval_std / avg_interval, 5) / 5  # 标准化，最大值5
            
            # 2. 短时间内大量请求的特征
            # 计算每分钟的请求数
            if not self.df.empty and 'timestamp' in self.df.columns:
                self.df['minute'] = self.df['timestamp'].dt.floor('min')
                requests_per_minute = self.df.groupby('minute').size()
                
                # 最大每分钟请求数
                max_rpm = requests_per_minute.max()
                features[4] = min(max_rpm / 60, 1)  # 标准化，最大60请求/分钟
                
                # 平均每分钟请求数
                avg_rpm = requests_per_minute.mean()
                features[5] = min(avg_rpm / 30, 1)  # 标准化，最大30请求/分钟
                
                # 请求量的峰谷比
                if avg_rpm > 0:
                    features[6] = min(max_rpm / avg_rpm, 10) / 10  # 标准化，最大值10
            
            # 3. 访问时间模式特征
            if not self.df.empty and 'timestamp' in self.df.columns:
                # 工作时间内的请求比例（9:00-18:00）
                work_hours = ((self.df['timestamp'].dt.hour >= 9) & 
                             (self.df['timestamp'].dt.hour < 18))
                features[7] = work_hours.mean()
                
                # 深夜请求比例（23:00-6:00）
                night_hours = ((self.df['timestamp'].dt.hour >= 23) | 
                              (self.df['timestamp'].dt.hour < 6))
                features[8] = night_hours.mean()
                
                # 周末请求比例
                weekend = (self.df['timestamp'].dt.dayofweek >= 5)
                features[9] = weekend.mean()
            
            # 4. 时间范围特征
            if 'params_temporal_range' in self.df.columns:
                # 过滤掉非列表类型的时间范围
                valid_ranges = self.df['params_temporal_range'].dropna().apply(
                    lambda x: x if isinstance(x, list) and len(x) >= 2 else None).dropna()
                
                if not valid_ranges.empty:
                    # 提取开始和结束时间
                    start_times = []
                    end_times = []
                    future_data_requests = 0
                    past_data_requests = 0
                    large_time_spans = 0
                    
                    current_time = datetime.now()
                    for time_range in valid_ranges:
                        if isinstance(time_range, list) and len(time_range) >= 2:
                            try:
                                start_time = pd.to_datetime(time_range[0])
                                end_time = pd.to_datetime(time_range[1])
                                
                                start_times.append(start_time)
                                end_times.append(end_time)
                                
                                # 检查是否请求未来数据
                                if end_time > current_time:
                                    future_data_requests += 1
                                
                                # 检查是否请求历史数据（超过5年）
                                if start_time < current_time - timedelta(days=5*365):
                                    past_data_requests += 1
                                
                                # 检查时间跨度是否很大（超过1年）
                                if (end_time - start_time).days > 365:
                                    large_time_spans += 1
                            except:
                                pass
                    
                    # 未来数据请求比例
                    features[10] = future_data_requests / len(valid_ranges)
                    
                    # 远古历史数据请求比例
                    features[11] = past_data_requests / len(valid_ranges)
                    
                    # 大时间跨度请求比例
                    features[12] = large_time_spans / len(valid_ranges)
            
            # 5. 临时特征（可能与时间相关的异常模式）
            # 最长连续短间隔请求序列长度 (突发性请求)
            if 'time_diff' in self.df.columns:
                burst_threshold = 5  # 5秒内的请求被视为突发
                bursts = (self.df['time_diff'] <= burst_threshold).astype(int)
                
                # 计算最长连续突发序列
                max_burst = 0
                current_burst = 0
                for b in bursts:
                    if b == 1:
                        current_burst += 1
                    else:
                        max_burst = max(max_burst, current_burst)
                        current_burst = 0
                max_burst = max(max_burst, current_burst)
                
                # 标准化
                features[13] = min(max_burst / 20, 1)  # 最多20次连续突发
                
                # 突发请求比例
                features[14] = bursts.mean()
            
            # 访问的周期性 (entropy of hour distribution)
            if not self.df.empty and 'timestamp' in self.df.columns:
                hour_counts = self.df['timestamp'].dt.hour.value_counts(normalize=True)
                entropy = stats.entropy(hour_counts)
                # 标准化熵值（最大熵为ln(24)≈3.18）
                features[15] = min(entropy / 3.2, 1)
            
            return features
        except Exception as e:
            logger.error(f"提取时间特征时出错: {str(e)}")
            return np.zeros(16)
    
    def _extract_spatial_features(self) -> np.ndarray:
        """
        提取空间行为特征
        
        返回:
            16维空间行为特征向量
        """
        try:
            features = np.zeros(16)
            
            # 如果DataFrame为空，返回零向量
            if self.df.empty:
                return features
            
            # 1. 空间范围特征
            if 'params_spatial_range' in self.df.columns:
                # 过滤掉非列表类型的空间范围
                valid_ranges = self.df['params_spatial_range'].dropna().apply(
                    lambda x: x if isinstance(x, list) and len(x) >= 4 else None).dropna()
                
                if not valid_ranges.empty:
                    # 提取空间范围面积
                    areas = []
                    global_range_requests = 0
                    tiny_range_requests = 0
                    point_like_range_requests = 0
                    
                    for spatial_range in valid_ranges:
                        if isinstance(spatial_range, list) and len(spatial_range) >= 4:
                            # 通常格式为 [min_lon, min_lat, max_lon, max_lat]
                            min_lon, min_lat, max_lon, max_lat = spatial_range[:4]
                            
                            # 计算面积（简单近似，实际应考虑地球曲率）
                            area = (max_lon - min_lon) * (max_lat - min_lat)
                            areas.append(area)
                            
                            # 检查是否为全球范围请求
                            if area > 10000:  # 粗略估计全球面积
                                global_range_requests += 1
                            
                            # 检查是否为极小范围请求
                            if area < 0.01:  # 小于0.01平方度的区域（粗略估计为1km x 1km）
                                tiny_range_requests += 1
                            
                            # 检查是否为点状请求（几乎没有面积）
                            if area < 0.0001:  # 极小区域
                                point_like_range_requests += 1
                    
                    if areas:
                        # 平均请求面积（标准化）
                        avg_area = np.mean(areas)
                        features[0] = min(avg_area / 100, 1)  # 最大100平方度
                        
                        # 最大请求面积（标准化）
                        max_area = np.max(areas)
                        features[1] = min(max_area / 10000, 1)  # 最大10000平方度
                        
                        # 最小请求面积（标准化，反向）
                        min_area = np.min(areas)
                        features[2] = 1 - min(min_area * 1000, 1)  # 接近0时接近1
                        
                        # 面积变异系数（标准差/平均值）
                        area_std = np.std(areas)
                        if avg_area > 0:
                            features[3] = min(area_std / avg_area, 3) / 3
                        
                        # 全球范围请求比例
                        features[4] = global_range_requests / len(valid_ranges)
                        
                        # 极小范围请求比例
                        features[5] = tiny_range_requests / len(valid_ranges)
                        
                        # 点状请求比例
                        features[6] = point_like_range_requests / len(valid_ranges)
            
            # 2. 空间分辨率特征
            if 'params_spatial_resolution' in self.df.columns:
                valid_resolutions = self.df['params_spatial_resolution'].dropna()
                
                if not valid_resolutions.empty:
                    # 分辨率统计
                    high_res_requests = (valid_resolutions < 0.01).sum()  # 高分辨率请求
                    medium_res_requests = ((valid_resolutions >= 0.01) & 
                                          (valid_resolutions < 0.1)).sum()  # 中等分辨率
                    low_res_requests = (valid_resolutions >= 0.1).sum()  # 低分辨率
                    
                    total_requests = len(valid_resolutions)
                    
                    # 各分辨率级别比例
                    features[7] = high_res_requests / total_requests
                    features[8] = medium_res_requests / total_requests
                    features[9] = low_res_requests / total_requests
                    
                    # 平均分辨率（反向标准化，越小值越大）
                    avg_resolution = valid_resolutions.mean()
                    features[10] = 1 - min(avg_resolution / 0.5, 1)
                    
                    # 分辨率变异系数
                    res_std = valid_resolutions.std()
                    if avg_resolution > 0:
                        features[11] = min(res_std / avg_resolution, 3) / 3
            
            # 3. 空间范围变化特征
            if 'params_spatial_range' in self.df.columns:
                # 计算连续请求之间的空间变化
                valid_ranges = self.df['params_spatial_range'].dropna().apply(
                    lambda x: x if isinstance(x, list) and len(x) >= 4 else None)
                
                if len(valid_ranges) > 1:
                    # 提取中心点序列
                    centers = []
                    for spatial_range in valid_ranges:
                        if isinstance(spatial_range, list) and len(spatial_range) >= 4:
                            min_lon, min_lat, max_lon, max_lat = spatial_range[:4]
                            center_lon = (min_lon + max_lon) / 2
                            center_lat = (min_lat + max_lat) / 2
                            centers.append((center_lon, center_lat))
                    
                    if len(centers) > 1:
                        # 计算中心点之间的移动距离（简单欧氏距离）
                        distances = []
                        for i in range(1, len(centers)):
                            lon1, lat1 = centers[i-1]
                            lon2, lat2 = centers[i]
                            
                            # 简单的欧氏距离，理想情况下应使用大圆距离
                            dist = ((lon2 - lon1)**2 + (lat2 - lat1)**2)**0.5
                            distances.append(dist)
                        
                        if distances:
                            # 平均移动距离
                            avg_distance = np.mean(distances)
                            features[12] = min(avg_distance / 10, 1)  # 最大10度
                            
                            # 最大移动距离
                            max_distance = np.max(distances)
                            features[13] = min(max_distance / 50, 1)  # 最大50度
                            
                            # 连续请求中心点重复比例
                            repeat_centers = sum(d < 0.001 for d in distances) / len(distances)
                            features[14] = repeat_centers
                            
                            # 空间跳跃率（大幅度移动的比例）
                            large_jumps = sum(d > 5 for d in distances) / len(distances)
                            features[15] = large_jumps
            
            return features
        except Exception as e:
            logger.error(f"提取空间特征时出错: {str(e)}")
            return np.zeros(16)
    
    def _extract_data_volume_features(self) -> np.ndarray:
        """
        提取数据量和计算时间特征
        
        返回:
            16维数据量特征向量
        """
        try:
            features = np.zeros(16)
            
            # 如果DataFrame为空，返回零向量
            if self.df.empty:
                return features
            
            # 1. 响应大小特征
            if 'response_size' in self.df.columns:
                valid_sizes = self.df['response_size'].dropna()
                
                if not valid_sizes.empty:
                    # 平均响应大小（MB）
                    avg_size = valid_sizes.mean() / (1024 * 1024)
                    features[0] = min(avg_size / 500, 1)  # 最大500MB
                    
                    # 最大响应大小（MB）
                    max_size = valid_sizes.max() / (1024 * 1024)
                    features[1] = min(max_size / 1000, 1)  # 最大1000MB
                    
                    # 总响应大小（GB）
                    total_size = valid_sizes.sum() / (1024 * 1024 * 1024)
                    features[2] = min(total_size / 50, 1)  # 最大50GB
                    
                    # 大响应比例（>100MB）
                    large_responses = (valid_sizes > 100 * 1024 * 1024).mean()
                    features[3] = large_responses
                    
                    # 小响应比例（<1MB）
                    small_responses = (valid_sizes < 1 * 1024 * 1024).mean()
                    features[4] = small_responses
                    
                    # 响应大小变异系数
                    size_std = valid_sizes.std()
                    if avg_size > 0:
                        features[5] = min(size_std / valid_sizes.mean(), 3) / 3
            
            # 2. 计算时间特征
            if 'compute_time' in self.df.columns:
                valid_times = self.df['compute_time'].dropna()
                
                if not valid_times.empty:
                    # 平均计算时间（秒）
                    avg_time = valid_times.mean()
                    features[6] = min(avg_time / 60, 1)  # 最大60秒
                    
                    # 最大计算时间（秒）
                    max_time = valid_times.max()
                    features[7] = min(max_time / 120, 1)  # 最大120秒
                    
                    # 总计算时间（分钟）
                    total_time = valid_times.sum() / 60
                    features[8] = min(total_time / 120, 1)  # 最大120分钟
                    
                    # 长时间计算比例（>30秒）
                    long_compute = (valid_times > 30).mean()
                    features[9] = long_compute
                    
                    # 短时间计算比例（<5秒）
                    short_compute = (valid_times < 5).mean()
                    features[10] = short_compute
                    
                    # 计算时间变异系数
                    time_std = valid_times.std()
                    if avg_time > 0:
                        features[11] = min(time_std / avg_time, 3) / 3
            
            # 3. 数据吞吐量特征（响应大小/计算时间）
            if 'response_size' in self.df.columns and 'compute_time' in self.df.columns:
                self.df['throughput'] = self.df['response_size'] / self.df['compute_time'].clip(lower=0.1)
                valid_throughput = self.df['throughput'].dropna()
                
                if not valid_throughput.empty:
                    # 平均吞吐量（MB/s）
                    avg_throughput = valid_throughput.mean() / (1024 * 1024)
                    features[12] = min(avg_throughput / 50, 1)  # 最大50MB/s
                    
                    # 最大吞吐量（MB/s）
                    max_throughput = valid_throughput.max() / (1024 * 1024)
                    features[13] = min(max_throughput / 100, 1)  # 最大100MB/s
                    
                    # 高吞吐量请求比例（>10MB/s）
                    high_throughput = (valid_throughput > 10 * 1024 * 1024).mean()
                    features[14] = high_throughput
                    
                    # 吞吐量变异系数
                    throughput_std = valid_throughput.std()
                    if avg_throughput > 0:
                        features[15] = min(throughput_std / valid_throughput.mean(), 3) / 3
            
            return features
        except Exception as e:
            logger.error(f"提取数据量特征时出错: {str(e)}")
            return np.zeros(16)
    
    def _extract_session_features(self) -> np.ndarray:
        """
        提取会话行为特征
        
        返回:
            16维会话行为特征向量
        """
        try:
            features = np.zeros(16)
            
            # 如果没有会话数据，返回零向量
            if not self.sessions:
                return features
            
            # 1. 会话基本统计特征
            # 会话数量（标准化）
            num_sessions = len(self.sessions)
            features[0] = min(num_sessions / 10, 1)  # 最大10个会话
            
            # 会话长度统计
            session_lengths = [len(calls) for _, calls in self.sessions.items()]
            
            if session_lengths:
                # 平均会话长度
                avg_length = np.mean(session_lengths)
                features[1] = min(avg_length / 100, 1)  # 最大100个调用
                
                # 最长会话长度
                max_length = np.max(session_lengths)
                features[2] = min(max_length / 200, 1)  # 最大200个调用
                
                # 会话长度变异系数
                length_std = np.std(session_lengths)
                if avg_length > 0:
                    features[3] = min(length_std / avg_length, 3) / 3
                
                # 单调用会话比例
                single_call_sessions = sum(1 for l in session_lengths if l == 1)
                features[4] = single_call_sessions / num_sessions
                
                # 长会话比例（>50个调用）
                long_sessions = sum(1 for l in session_lengths if l > 50)
                features[5] = long_sessions / num_sessions
            
            # 2. 会话时间特征
            session_durations = []
            for session_id, calls in self.sessions.items():
                if len(calls) > 1:
                    # 尝试提取每个会话的时间戳并计算持续时间
                    timestamps = []
                    for call in calls:
                        if 'timestamp' in call and call['timestamp']:
                            try:
                                timestamp = pd.to_datetime(call['timestamp'])
                                timestamps.append(timestamp)
                            except:
                                pass
                    
                    if len(timestamps) > 1:
                        duration = (max(timestamps) - min(timestamps)).total_seconds()
                        session_durations.append(duration)
            
            if session_durations:
                # 平均会话持续时间（分钟）
                avg_duration = np.mean(session_durations) / 60
                features[6] = min(avg_duration / 120, 1)  # 最大120分钟
                
                # 最长会话持续时间（分钟）
                max_duration = np.max(session_durations) / 60
                features[7] = min(max_duration / 240, 1)  # 最大240分钟
                
                # 会话持续时间变异系数
                duration_std = np.std(session_durations)
                if avg_duration > 0:
                    features[8] = min(duration_std / (avg_duration * 60), 3) / 3
                
                # 短会话比例（<5分钟）
                short_sessions = sum(1 for d in session_durations if d < 300)
                features[9] = short_sessions / len(session_durations)
            
            # 长会话比例（>30分钟）
            long_duration_sessions = sum(1 for d in session_durations if d > 1800)
            features[10] = long_duration_sessions / len(session_durations)
            
            # 3. 会话API调用频率特征
            session_frequencies = []
            for session_id, calls in self.sessions.items():
                timestamps = []
                for call in calls:
                    if 'timestamp' in call and call['timestamp']:
                        try:
                            timestamp = pd.to_datetime(call['timestamp'])
                            timestamps.append(timestamp)
                        except:
                            pass
                
                if len(timestamps) > 1:
                    # 计算会话持续时间（秒）
                    duration = (max(timestamps) - min(timestamps)).total_seconds()
                    if duration > 0:
                        # 计算每秒调用频率
                        frequency = len(timestamps) / duration
                        session_frequencies.append(frequency)
            
            if session_frequencies:
                # 平均每秒调用频率
                avg_frequency = np.mean(session_frequencies)
                features[11] = min(avg_frequency * 10, 1)  # 最大0.1调用/秒
                
                # 最高每秒调用频率
                max_frequency = np.max(session_frequencies)
                features[12] = min(max_frequency * 5, 1)  # 最大0.2调用/秒
                
                # 调用频率变异系数
                frequency_std = np.std(session_frequencies)
                if avg_frequency > 0:
                    features[13] = min(frequency_std / avg_frequency, 3) / 3
                
                # 高频率会话比例（>0.05调用/秒）
                high_freq_sessions = sum(1 for f in session_frequencies if f > 0.05)
                features[14] = high_freq_sessions / len(session_frequencies)
            
            # 4. 并发会话特征
            if not self.df.empty and 'timestamp' in self.df.columns and 'session_id' in self.df.columns:
                # 将时间戳按分钟聚合
                self.df['minute'] = self.df['timestamp'].dt.floor('min')
                
                # 计算每分钟活跃的会话数
                concurrent_sessions = self.df.groupby('minute')['session_id'].nunique()
                
                # 平均并发会话数
                avg_concurrent = concurrent_sessions.mean()
                features[15] = min(avg_concurrent / 5, 1)  # 最大5个并发会话
            
            return features
        except Exception as e:
            logger.error(f"提取会话特征时出错: {str(e)}")
            return np.zeros(16)
    
    def _extract_ip_device_features(self) -> np.ndarray:
        """
        提取IP和设备相关特征
        
        返回:
            16维IP和设备特征向量
        """
        try:
            features = np.zeros(16)
            
            # 如果DataFrame为空，返回零向量
            if self.df.empty:
                return features
            
            # 1. IP地址特征
            if 'ip' in self.df.columns:
                # 计算唯一IP数量
                unique_ips = self.df['ip'].nunique()
                total_requests = len(self.df)
                
                # 唯一IP数量（标准化）
                features[0] = min(unique_ips / 20, 1)  # 最大20个IP
                
                # 每个请求的平均唯一IP数量（IP变化频率）
                ip_change_ratio = unique_ips / max(total_requests, 1)
                features[1] = ip_change_ratio
                
                # 计算IP更换频率
                self.df['ip_changed'] = self.df['ip'].shift() != self.df['ip']
                ip_change_rate = self.df['ip_changed'].mean()
                features[2] = ip_change_rate
                
                # 计算每个IP的请求数统计
                ip_counts = self.df['ip'].value_counts()
                
                # 最频繁IP的请求比例
                if not ip_counts.empty:
                    features[3] = ip_counts.iloc[0] / total_requests
                
                # IP分布熵（越高表示IP分布越均匀）
                ip_probs = ip_counts / total_requests
                entropy = stats.entropy(ip_probs)
                # 标准化熵（最大熵取决于唯一IP数，ln(unique_ips)）
                if unique_ips > 1:
                    max_entropy = np.log(unique_ips)
                    features[4] = min(entropy / max_entropy, 1)
            
            # 2. 设备ID特征
            if 'device_id' in self.df.columns:
                # 计算唯一设备数量
                unique_devices = self.df['device_id'].nunique()
                
                # 唯一设备数量（标准化）
                features[5] = min(unique_devices / 20, 1)  # 最大20个设备
                
                # 每个请求的平均唯一设备数量（设备变化频率）
                device_change_ratio = unique_devices / max(total_requests, 1)
                features[6] = device_change_ratio
                
                # 计算设备更换频率
                self.df['device_changed'] = self.df['device_id'].shift() != self.df['device_id']
                device_change_rate = self.df['device_changed'].mean()
                features[7] = device_change_rate
                
                # 计算每个设备的请求数统计
                device_counts = self.df['device_id'].value_counts()
                
                # 最频繁设备的请求比例
                if not device_counts.empty:
                    features[8] = device_counts.iloc[0] / total_requests
                
                # 设备分布熵
                device_probs = device_counts / total_requests
                entropy = stats.entropy(device_probs)
                # 标准化熵
                if unique_devices > 1:
                    max_entropy = np.log(unique_devices)
                    features[9] = min(entropy / max_entropy, 1)
            
            # 3. IP与设备对应关系特征
            if 'ip' in self.df.columns and 'device_id' in self.df.columns:
                # 计算每个IP使用的设备数
                ip_device_counts = self.df.groupby('ip')['device_id'].nunique()
                
                # 平均每个IP使用的设备数
                avg_devices_per_ip = ip_device_counts.mean()
                features[10] = min(avg_devices_per_ip / 5, 1)  # 最大5个设备/IP
                
                # 最大每个IP使用的设备数
                max_devices_per_ip = ip_device_counts.max()
                features[11] = min(max_devices_per_ip / 10, 1)  # 最大10个设备/IP
                
                # 计算每个设备使用的IP数
                device_ip_counts = self.df.groupby('device_id')['ip'].nunique()
                
                # 平均每个设备使用的IP数
                avg_ips_per_device = device_ip_counts.mean()
                features[12] = min(avg_ips_per_device / 5, 1)  # 最大5个IP/设备
                
                # 最大每个设备使用的IP数
                max_ips_per_device = device_ip_counts.max()
                features[13] = min(max_ips_per_device / 10, 1)  # 最大10个IP/设备
                
                # 异常IP-设备对比例（一个IP多个设备或一个设备多个IP）
                anomalous_ip_devices = ((ip_device_counts > 1) | (device_ip_counts > 1)).mean()
                features[14] = anomalous_ip_devices
                
                # IP-设备映射熵（IP和设备的一对一对应程度）
                ip_device_pairs = self.df.groupby(['ip', 'device_id']).size()
                ip_device_probs = ip_device_pairs / ip_device_pairs.sum()
                ip_device_entropy = stats.entropy(ip_device_probs)
                
                # 标准化熵
                max_entropy = np.log(len(ip_device_pairs))
                features[15] = min(ip_device_entropy / max_entropy if max_entropy > 0 else 0, 1)
            
            return features
        except Exception as e:
            logger.error(f"提取IP和设备特征时出错: {str(e)}")
            return np.zeros(16)
    
    def _extract_pattern_features(self) -> np.ndarray:
        """
        提取访问模式特征
        
        返回:
            16维访问模式特征向量
        """
        try:
            features = np.zeros(16)
            
            # 如果DataFrame为空，返回零向量
            if self.df.empty:
                return features
            
            # 1. 请求成功率特征
            if 'status' in self.df.columns:
                # 计算成功请求比例
                success_rate = (self.df['status'] == 'success').mean()
                features[0] = success_rate
                
                # 失败请求比例
                failure_rate = (self.df['status'] == 'error').mean()
                features[1] = failure_rate
                
                # 连续失败请求
                if 'status' in self.df.columns:
                    failure_mask = (self.df['status'] == 'error').astype(int)
                    
                    # 计算最长连续失败序列
                    max_consecutive_failures = 0
                    current_failures = 0
                    for failure in failure_mask:
                        if failure == 1:
                            current_failures += 1
                        else:
                            max_consecutive_failures = max(max_consecutive_failures, current_failures)
                            current_failures = 0
                    max_consecutive_failures = max(max_consecutive_failures, current_failures)
                    
                    # 标准化
                    features[2] = min(max_consecutive_failures / 10, 1)  # 最多10次连续失败
            
            # 2. API类型分布特征
            if 'api' in self.df.columns:
                api_counts = self.df['api'].value_counts(normalize=True)
                
                # 查询类API比例
                query_apis = sum(api_counts.get(api, 0) for api in api_counts.index 
                                if 'query' in api.lower())
                features[3] = query_apis
                
                # 数据修改类API比例
                modify_apis = sum(api_counts.get(api, 0) for api in api_counts.index 
                                 if any(op in api.lower() for op in ['update', 'modify', 'create', 'delete']))
                features[4] = modify_apis
                
                # 未来数据查询API比例
                future_apis = sum(api_counts.get(api, 0) for api in api_counts.index 
                                 if 'future' in api.lower())
                features[5] = future_apis
                
                # API分布熵
                api_entropy = stats.entropy(api_counts)
                # 标准化熵
                max_entropy = np.log(len(api_counts))
                features[6] = min(api_entropy / max_entropy if max_entropy > 0 else 0, 1)
            
            # 3. 数据类别分布特征
            if 'params_data_category' in self.df.columns:
                # 数据类别计数
                category_counts = self.df['params_data_category'].value_counts(normalize=True)
                
                # 类别分布熵
                if not category_counts.empty:
                    category_entropy = stats.entropy(category_counts)
                    # 标准化熵
                    max_entropy = np.log(len(category_counts))
                    features[7] = min(category_entropy / max_entropy if max_entropy > 0 else 0, 1)
                
                # 敏感数据类别访问比例
                sensitive_categories = ['temperature', 'precipitation', 'wind', 'radiation']
                sensitive_access = sum(category_counts.get(cat, 0) for cat in category_counts.index 
                                     if any(s_cat in str(cat).lower() for s_cat in sensitive_categories))
                features[8] = sensitive_access
            
            # 4. 综合行为模式特征
            # 计算请求中的周期性
            # 对每个用户ID计算请求的周期性
            if 'uid' in self.df.columns and 'timestamp' in self.df.columns:
                # 计算每个用户的请求间隔
                user_intervals = {}
                for uid in self.df['uid'].unique():
                    user_df = self.df[self.df['uid'] == uid].sort_values('timestamp')
                    if len(user_df) > 1:
                        intervals = user_df['timestamp'].diff().dropna().dt.total_seconds()
                        user_intervals[uid] = intervals
                
                # 计算间隔的变异系数（标准化标准差）
                interval_cvs = []
                for uid, intervals in user_intervals.items():
                    if len(intervals) > 1:
                        cv = intervals.std() / intervals.mean() if intervals.mean() > 0 else 0
                        interval_cvs.append(cv)
                
                # 平均周期性得分（变异系数的倒数）
                if interval_cvs:
                    avg_periodicity = 1 / (1 + np.mean(interval_cvs))
                    features[9] = avg_periodicity
            
            # 5. 爬虫行为特征
            # 计算每秒请求频率
            if 'timestamp' in self.df.columns:
                total_duration = (self.df['timestamp'].max() - self.df['timestamp'].min()).total_seconds()
                if total_duration > 0:
                    requests_per_second = len(self.df) / total_duration
                    features[10] = min(requests_per_second / 1, 1)  # 最大1请求/秒
            
            # 请求路径覆盖率（数据区域的系统性覆盖）
            if 'params_spatial_range' in self.df.columns:
                valid_ranges = self.df['params_spatial_range'].dropna().apply(
                    lambda x: x if isinstance(x, list) and len(x) >= 4 else None).dropna()
                
                if len(valid_ranges) > 1:
                    # 创建所有请求覆盖的边界框
                    min_lons = []
                    min_lats = []
                    max_lons = []
                    max_lats = []
                    
                    for spatial_range in valid_ranges:
                        if isinstance(spatial_range, list) and len(spatial_range) >= 4:
                            min_lon, min_lat, max_lon, max_lat = spatial_range[:4]
                            min_lons.append(min_lon)
                            min_lats.append(min_lat)
                            max_lons.append(max_lon)
                            max_lats.append(max_lat)
                    
                    if min_lons and min_lats and max_lons and max_lats:
                        # 计算总边界框面积
                        global_min_lon = min(min_lons)
                        global_min_lat = min(min_lats)
                        global_max_lon = max(max_lons)
                        global_max_lat = max(max_lats)
                        
                        total_area = (global_max_lon - global_min_lon) * (global_max_lat - global_min_lat)
                        
                        # 计算各个请求的面积
                        request_areas = []
                        for i in range(len(min_lons)):
                            area = (max_lons[i] - min_lons[i]) * (max_lats[i] - min_lats[i])
                            request_areas.append(area)
                        
                        # 计算请求面积与总面积的比率（覆盖率）
                        if total_area > 0:
                            area_coverage = sum(request_areas) / (total_area * len(request_areas))
                            features[11] = min(area_coverage, 1)
            
            # 请求参数多样性（低多样性可能表示自动化爬取）
            if 'params' in self.df.columns:
                # 计算唯一参数集的数量
                unique_params = set()
                for params in self.df['params']:
                    if isinstance(params, dict):
                        # 将值转换为字符串，避免列表不可哈希的问题
                        param_tuple = tuple(sorted([(key, str(value)) for key, value in params.items()]))
                        unique_params.add(param_tuple)
                
                # 参数多样性得分（唯一参数集/总请求数）
                params_diversity = len(unique_params) / max(len(self.df), 1)
                features[12] = params_diversity
            
            # 6. 未来特征预留
            # 为了保持特征向量长度一致，保留三个位置为未来使用
            features[13] = 0
            features[14] = 0
            features[15] = 0
            
            return features
        except Exception as e:
            logger.error(f"提取访问模式特征时出错: {str(e)}")
            return np.zeros(16)
    
    def extract_api_call_patterns(self) -> Dict[str, Any]:
        """
        提取API调用模式的统计信息
        
        返回:
            包含API调用模式统计信息的字典
        """
        result = {
            "total_api_calls": sum(len(seq) for seq in self.api_sequences),
            "unique_apis": set(),
            "api_frequencies": Counter(),
            "common_api_sequences": Counter(),
            "avg_api_calls_per_session": 0,
            "max_api_calls_per_session": 0,
            "min_api_calls_per_session": float('inf') if self.api_sequences else 0
        }
        
        # 统计API频率
        for seq in self.api_sequences:
            result["unique_apis"].update(seq)
            result["api_frequencies"].update(seq)
            
            # 统计API序列
            for i in range(len(seq) - 1):
                pair = f"{seq[i]} -> {seq[i+1]}"
                result["common_api_sequences"][pair] += 1
            
            # 更新每个会话的API调用数
            calls_count = len(seq)
            result["max_api_calls_per_session"] = max(result["max_api_calls_per_session"], calls_count)
            result["min_api_calls_per_session"] = min(result["min_api_calls_per_session"], calls_count)
        
        # 计算平均API调用数
        if self.api_sequences:
            result["avg_api_calls_per_session"] = result["total_api_calls"] / len(self.api_sequences)
        
        # 将集合转换为列表
        result["unique_apis"] = list(result["unique_apis"])
        
        # 获取前10个最常见的API
        result["top_10_apis"] = dict(result["api_frequencies"].most_common(10))
        
        # 获取前10个最常见的API序列
        result["top_10_sequences"] = dict(result["common_api_sequences"].most_common(10))
        
        return result
    
    def save_features(self, output_file: str) -> None:
        """
        保存特征向量到文件
        
        参数:
            output_file: 输出文件路径
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"创建输出目录: {output_dir}")
        
        # 检查文件是否已存在
        if os.path.exists(output_file):
            logger.warning(f"文件 {output_file} 已存在，将被覆盖")
        
        # 提取特征并保存
        features = self.extract_features()
        np.save(output_file, features)
        logger.info(f"特征向量已保存至: {output_file}")
        
        # 保存API调用模式统计信息
        stats_file = f"{os.path.splitext(output_file)[0]}_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.extract_api_call_patterns(), f, ensure_ascii=False, indent=2)
        logger.info(f"API调用模式统计信息已保存至: {stats_file}")


def process_file(log_file: str, output_file: Optional[str] = None) -> None:
    """
    处理日志文件并提取特征
    
    参数:
        log_file: 日志文件路径
        output_file: 输出文件路径（可选）
    """
    # 确定输出文件名
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(log_file))[0]
        output_file = f"{base_name}_features.npy"
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}")
    
    # 提取特征
    extractor = FeatureExtractor(log_file)
    
    # 保存特征
    try:
        extractor.save_features(output_file)
    except Exception as error:
        logger.error(f"保存特征向量失败: {str(error)}")
        sys.exit(1)


def process_directory(input_dir: str, output_dir: Optional[str] = None) -> None:
    """
    处理目录中的所有日志文件并提取特征
    
    参数:
        input_dir: 输入目录路径，包含日志文件
        output_dir: 输出目录路径（可选）
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
        output_dir = os.path.join(os.path.dirname(input_dir), 
                                  f"{os.path.basename(input_dir)}_features")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"将输出特征向量保存到目录: {output_dir}")
    
    # 遍历输入目录中的所有log文件
    log_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.log'):
                log_files.append(os.path.join(root, file))
    
    if not log_files:
        logger.warning(f"在目录 {input_dir} 中没有找到log日志文件")
        return
    
    logger.info(f"找到 {len(log_files)} 个日志文件需要处理")
    
    # 使用进度条包装文件列表
    if has_tqdm:
        log_files_iter = tqdm(log_files, desc="处理文件", unit="个")
    else:
        log_files_iter = log_files
        # 简单的进度显示
        logger.info("开始处理文件:")
    
    # 成功和失败的计数器
    success_count = 0
    failure_count = 0
    
    # 处理每个日志文件
    for i, log_file in enumerate(log_files_iter, 1):
        # 计算相对路径以保持目录结构
        rel_path = os.path.relpath(log_file, input_dir)
        output_file = os.path.join(output_dir, 
                                   f"{os.path.splitext(rel_path)[0]}_features.npy")
        
        # 确保输出文件的目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 如果没有tqdm, 显示简单进度
        if not has_tqdm:
            progress_percent = (i / len(log_files)) * 100
            logger.info(f"[{i}/{len(log_files)}] {progress_percent:.1f}% - 处理: {os.path.basename(log_file)}")
        
        try:
            process_file(log_file, output_file)
            success_count += 1
            if not has_tqdm:
                logger.info(f"✓ 成功处理: {os.path.basename(log_file)}")
        except Exception as e:
            failure_count += 1
            if has_tqdm:
                tqdm.write(f"处理文件 {log_file} 时出错: {str(e)}")
            else:
                logger.error(f"✗ 处理文件 {log_file} 失败: {str(e)}")
            # 继续处理下一个文件，不中断整个流程
    
    # 处理完成后的摘要
    logger.info(f"\n处理完成摘要:")
    logger.info(f"- 总文件数: {len(log_files)}")
    logger.info(f"- 成功处理: {success_count}")
    logger.info(f"- 处理失败: {failure_count}")
    logger.info(f"特征向量已保存到目录: {output_dir}")


def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description="从用户日志提取恶意行为特征向量")
    
    # 创建互斥组，确保用户只能指定文件或目录，不能同时指定
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", help="用户日志文件路径")
    group.add_argument("-d", "--directory", help="包含多个日志文件的目录路径")
    
    parser.add_argument("-o", "--output", help="输出文件或目录路径（可选）")
    
    # 添加进度条相关选项
    parser.add_argument("--no-progress", action="store_true", 
                      help="禁用进度条显示")
    
    args = parser.parse_args()
    
    # 如果指定禁用进度条，设置环境变量
    if args.no_progress:
        os.environ["TQDM_DISABLE"] = "1"
    
    if args.file:
        # 处理单个文件
        process_file(args.file, args.output)
    else:
        # 处理整个目录
        process_directory(args.directory, args.output)


if __name__ == "__main__":
    main()

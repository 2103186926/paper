#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labels_extractor.py - 标签文件生成工具

功能：批量生成labels.csv标签文件内容
用法：python labels_extractor.py <前缀> <标签值> <数量> [输出文件名]
例如：python labels_extractor.py risk_mixed 1 10 mixed_labels.txt
"""

import os
import sys
import argparse


def generate_labels(prefix: str, label: int, count: int, output_file: str = None) -> None:
    """
    生成标签文件内容
    
    参数:
        prefix: 文件名前缀
        label: 标签值（0或1）
        count: 生成的条目数量
        output_file: 输出文件名（可选，默认为<prefix>_labels.txt）
    """
    # 验证参数
    if count <= 0:
        print("错误: 数量必须大于0")
        return
    
    if label not in [0, 1]:
        print(f"警告: 标签值 {label} 不是标准的二分类值(0或1)，但仍将继续")
    
    # 确定输出文件名
    if output_file is None:
        output_file = f"{prefix}_labels.txt"
    
    # 生成标签内容
    labels = []
    for i in range(1, count + 1):
        # if count < 100:
        #     # 如果数量小于100，使用两位数格式（01, 02, ...）
        #     entry = f"{prefix}_{i:02d},{label}"
        # else:
        #     # 如果数量大于等于100，使用三位数格式（001, 002, ...）
        #     entry = f"{prefix}_{i:03d},{label}"

        # 始终使用两位数格式（01, 02, ...），无论数量大小
        entry = f"{prefix}_{i:02d},{label}"
        labels.append(entry)
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels))
    
    print(f"已生成 {count} 个标签条目，保存至: {output_file}")
    print(f"前5个条目示例:")
    for entry in labels[:5]:
        print(f"  {entry}")
    if count > 5:
        print("  ...")
        print(f"  {labels[-1]}")


def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description="批量生成labels.csv标签文件内容")
    parser.add_argument("prefix", help="文件名前缀，例如: risk_mixed")
    parser.add_argument("label", type=int, choices=[0, 1], help="标签值: 0(安全)或1(危险)")
    parser.add_argument("count", type=int, help="生成的条目数量")
    parser.add_argument("output", nargs="?", default=None, help="输出文件名(可选)")
    
    args = parser.parse_args()
    
    # 调用生成函数
    generate_labels(args.prefix, args.label, args.count, args.output)


if __name__ == "__main__":
    main()
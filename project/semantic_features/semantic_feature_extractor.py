#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
semantic_feature_extractor.py - Python代码语义特征提取工具

功能：从Python源代码直接提取语义特征向量，无需中间文件
输入：Python源代码文件
输出：128维语义特征向量（.npy文件）
"""

import ast
import json
import os
import sys
import re
import numpy as np
import argparse
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import hashlib
from pathlib import Path

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

# 定义敏感模块和函数
SENSITIVE_MODULES = {
    "os": ["system", "popen", "spawn", "exec", "execl", "execlp", "execle", 
           "execv", "execvp", "execvpe", "remove", "unlink", "rmdir"],
    "subprocess": ["run", "call", "check_call", "check_output", "Popen"],
    "pickle": ["load", "loads"],
    "yaml": ["load", "unsafe_load"],
    "marshal": ["load", "loads"],
    "shelve": ["open"],
    "socket": ["connect", "bind", "accept"],
    "urllib": ["urlopen", "Request"],
    "requests": ["get", "post", "put", "delete", "head", "patch"],
    "importlib": ["import_module", "__import__", "reload", "exec_module"]
}

# 定义内置敏感函数
BUILTIN_SENSITIVE = set([
    "eval", "exec", "compile", "__import__", "globals", "locals", 
    "getattr", "setattr", "delattr", "open"
])

# 将SENSITIVE_MODULES中的所有敏感函数添加到BUILTIN_SENSITIVE中
for module, functions in SENSITIVE_MODULES.items():
    for function in functions:
        BUILTIN_SENSITIVE.add(f"{module}.{function}")

# 定义代码混淆模式
CODE_PATTERNS = {
    "base64_decode": r"base64\.(b64decode|decodestring)",
    "hex_decode": r"(bytes|bytearray)\.fromhex|binascii\.unhexlify",
    "rot13": r"\.translate\(.*rot13.*\)|codecs\.encode\(.*rot13",
    "char_code": r"chr\(\d+\)|ord\(",
    "string_concat": r"\+\s*['\"]\w+['\"]",
    "string_join": r"['\"]\s*\.\s*join\(",
    "string_format": r"format\(|%[sd]|f['\"]",
    "string_replace": r"\.replace\(",
    "eval_exec": r"eval\(|exec\(",
    "compile_code": r"compile\(",
    "dynamic_import": r"__import__\(|importlib\.import_module",
    "os_system": r"os\.system\(|os\.popen\(",
    "subprocess_call": r"subprocess\.(call|run|Popen|check_output)",
    "file_open": r"open\(|with\s+open\(",
    "file_read": r"\.read\(\)|\.readlines\(\)",
    "file_write": r"\.write\(|\.writelines\(",
    "network_connect": r"\.connect\(|\.bind\(|\.accept\(|\.listen\(",
    "obfuscated_control": r"getattr\(.*,.*\)\(|globals\(\)\[.*\]",
    "lambda_obfuscation": r"lambda\s+.*:.*\(.*\)",
    "sleep_pattern": r"time\.sleep\(|asyncio\.sleep\(",
    "random_pattern": r"random\.|secrets\.",
    "environment_check": r"platform\.|sys\.platform|os\.name"
}

# 添加高危测试文件相关的常量
HIGH_RISK_PATTERNS = {
    "shell_execution": [
        r"os\.system\(['\"].*['\"]",
        r"subprocess\.call\(['\"].*['\"]",
        r"subprocess\.Popen\(['\"].*['\"]",
        r"commands\.getoutput\(['\"].*['\"]"
    ],
    "code_execution": [
        r"eval\(['\"].*['\"]",
        r"exec\(['\"].*['\"]",
        r"execfile\(['\"].*['\"]",
        r"compile\(['\"].*['\"],['\"].*['\"],['\"].*['\"]"
    ],
    "injection_attacks": [
        r".*\.execute\(['\"].*%.*['\"]",
        r".*\.query\(['\"].*%.*['\"]",
        r".*\.raw\(['\"].*%.*['\"]",
        r"cursor\.execute\(['\"].*\+.*['\"]"
    ],
    "file_operations": [
        r"open\(['\"].*['\"],['\"]w['\"]",
        r"open\(['\"].*['\"],['\"]r['\"]",
        r"open\(['\"]\/.*['\"],['\"].*['\"]",
        r"__import__\(['\"]os['\"].*\.path\.exists"
    ],
    "network_operations": [
        r"urllib\.request\.urlopen\(['\"]http",
        r"requests\.get\(['\"]http",
        r"requests\.post\(['\"]http",
        r"socket\.connect\(\(['\"].*['\"],\d+\)\)"
    ],
    "obfuscation_techniques": [
        r"base64\..*decode",
        r"zlib\.decompress",
        r"\.decode\(['\"].*['\"]",
        r"binascii\.unhexlify",
        r"codecs\.decode\(['\"].*['\"], *['\"]rot13['\"]"
    ]
}

# 添加一个自定义的JSON编码器，用于处理NumPy类型
class NumpyEncoder(json.JSONEncoder):
    """用于处理NumPy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class ASTtoDict(ast.NodeVisitor):
    """
    AST节点访问器类，将AST节点转换为字典结构
    
    继承自ast.NodeVisitor，重写visit方法以递归遍历AST树
    并将每个节点转换为包含类型、位置和子节点的字典
    """
    
    def __init__(self):
        self.node_id = 0  # 为每个节点分配唯一ID
        self.parent_stack = []  # 用于跟踪父节点
    
    def visit(self, node: ast.AST) -> Dict[str, Any]:
        """
        访问AST节点并将其转换为字典
        
        参数:
            node: AST节点对象
            
        返回:
            包含节点信息的字典
        """
        # 创建基本节点字典，包含类型和位置信息
        node_dict = {
            "id": self.node_id,
            "type": node.__class__.__name__,  # 节点类型名称
            "lineno": getattr(node, 'lineno', None),  # 行号（如果有）
            "col_offset": getattr(node, 'col_offset', None),  # 列偏移（如果有）
            "end_lineno": getattr(node, 'end_lineno', None),  # 结束行号（Python 3.8+）
            "end_col_offset": getattr(node, 'end_col_offset', None),  # 结束列偏移（Python 3.8+）
            "parent_id": self.parent_stack[-1]["id"] if self.parent_stack else None,  # 父节点ID
            "children": []  # 子节点列表
        }
        
        # 增加节点ID计数
        self.node_id += 1
        
        # 移除None值的位置信息，减小JSON大小
        node_dict = {k: v for k, v in node_dict.items() if v is not None}
        
        # 添加特定节点类型的属性
        if isinstance(node, ast.Name):
            node_dict["name"] = node.id
        elif isinstance(node, ast.Constant):
            node_dict["value"] = repr(node.value)
            node_dict["kind"] = getattr(node, 'kind', None)
        elif isinstance(node, ast.Str):  # 兼容Python 3.7及以下
            node_dict["value"] = repr(node.s)
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
            node_dict["name"] = node.name
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                node_dict["func_name"] = node.func.id
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    node_dict["func_name"] = f"{node.func.value.id}.{node.func.attr}"
        
        # 将当前节点压入父节点栈
        self.parent_stack.append(node_dict)
        
        # 递归访问子节点
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                # 处理子节点列表（如body, orelse等）
                for item in value:
                    if isinstance(item, ast.AST):
                        child_dict = self.visit(item)
                        node_dict["children"].append(child_dict)
            elif isinstance(value, ast.AST):
                # 处理单个子节点
                child_dict = self.visit(value)
                node_dict["children"].append(child_dict)
        
        # 弹出父节点栈
        self.parent_stack.pop()
        
        return node_dict

def python_to_ast_json(code: str, filename: str = "<unknown>") -> Dict[str, Any]:
    """
    将Python代码转换为JSON格式的AST
    
    参数:
        code: Python源代码
        filename: 文件名
        
    返回:
        包含AST的JSON字典
    """
    try:
        # 解析代码为AST
        tree = ast.parse(code, filename=filename)
        
        # 获取Python版本信息
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # 转换AST为字典
        converter = ASTtoDict()
        ast_dict = converter.visit(tree)
        
        # 构建完整的JSON结构
        result = {
            "metadata": {
                "source_file": os.path.basename(filename),
                "ast_version": python_version,
                "generated_by": "semantic_feature_extractor.py"
            },
            "ast": ast_dict
        }
        
        return result
    
    except SyntaxError as e:
        # 处理语法错误
        error_result = {
            "metadata": {
                "source_file": os.path.basename(filename),
                "error": f"Syntax error: {str(e)}"
            },
            "ast": None
        }
        return error_result


class SemanticFeatureExtractor:
    """语义特征提取器"""
    
    def __init__(self, source_file: str):
        """
        初始化语义特征提取器
        
        参数:
            source_file: Python源代码文件路径（即高危测试文件路径）
        """
        self.source_file = source_file
        self.source_code = ""
        self.ast_json = None
        
        # 读取源代码
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                self.source_code = f.read()
            print(f"已读取源代码文件: {source_file}")
        except Exception as e:
            print(f"读取源代码文件时出错: {str(e)}")
            sys.exit(1)
        
        # 解析AST
        self.ast_json = python_to_ast_json(self.source_code, source_file)
        if self.ast_json["ast"] is None:
            print(f"解析源代码时出错: {self.ast_json['metadata'].get('error', '未知错误')}")
            sys.exit(1)
    
    def extract_features(self) -> np.ndarray:
        """
        提取语义特征向量
        
        返回:
            128维语义特征向量
        """
        # 1. 提取敏感函数频次特征（32维）
        sensitive_features = self._extract_sensitive_function_features()
        
        # 2. 提取N-gram序列特征（64维）
        ngram_features = self._extract_ngram_features()
        
        # 3. 提取混淆模式特征（32维）
        pattern_features = self._extract_pattern_features()
        
        # 合并特征
        features = np.concatenate([sensitive_features, ngram_features, pattern_features])
        
        return features
    
    def _extract_sensitive_function_features(self) -> np.ndarray:
        """
        提取敏感函数频次特征
        
        返回:
            32维敏感函数频次特征向量
        """
        # 初始化特征向量
        features = np.zeros(32)
        
        # 定义敏感函数类别（共8类）
        categories = [
            "code_execution",      # eval, exec, compile
            "dynamic_import",      # __import__, importlib.import_module
            "os_commands",         # os.system, os.popen, subprocess.*
            "file_operations",     # open, read, write
            "network_operations",  # socket, urllib, requests
            "serialization",       # pickle, marshal, yaml
            "reflection",          # getattr, setattr, globals, locals
            "other_sensitive"      # 其他敏感操作
        ]
        
        # 遍历AST查找敏感函数调用
        sensitive_calls = self._find_sensitive_calls(self.ast_json["ast"])
        
        # 统计每个类别的调用次数
        category_counts = {cat: 0 for cat in categories}
        
        for call in sensitive_calls:
            if call in ["eval", "exec", "compile"]:
                category_counts["code_execution"] += 1
            elif call in ["__import__"] or "import_module" in call:
                category_counts["dynamic_import"] += 1
            elif call.startswith("os.") or call.startswith("subprocess."):
                category_counts["os_commands"] += 1
            elif call in ["open"] or any(op in call for op in ["read", "write"]):
                category_counts["file_operations"] += 1
            elif any(net in call for net in ["socket", "urllib", "requests"]):
                category_counts["network_operations"] += 1
            elif any(ser in call for ser in ["pickle", "marshal", "yaml"]):
                category_counts["serialization"] += 1
            elif call in ["getattr", "setattr", "globals", "locals"]:
                category_counts["reflection"] += 1
            else:
                category_counts["other_sensitive"] += 1
        
        # 为每个类别分配4个特征位置
        for i, category in enumerate(categories):
            count = category_counts[category]
            # 使用对数缩放避免大数值 
            scaled_count = np.log1p(count) if count > 0 else 0
            # 将缩放后的值填入特征向量
            features[i * 4:i * 4 + 4] = [
                1.0 if count > 0 else 0.0,  # 是否存在该类别的调用
                scaled_count,               # 调用次数（对数缩放）
                scaled_count / (np.log1p(len(sensitive_calls)) if sensitive_calls else 1),  # 相对频率
                scaled_count ** 2           # 平方项，增强重要性
            ]
        
        return features
    
    def _find_sensitive_calls(self, root_node: Any) -> List[str]:
        """
        按照节点ID升序查找AST中的敏感函数调用
        
        参数:
            root_node: AST根节点
            
        返回:
            敏感函数调用列表
        """
        if root_node is None:
            return []
            
        # 先收集所有节点到一个列表，并按ID排序
        all_nodes = self._collect_all_nodes(root_node)
        
        # 按节点ID排序（如果存在）
        all_nodes.sort(key=lambda node: node.get("id", 0) if isinstance(node, dict) else 0)
        
        # 收集敏感函数调用
        sensitive_calls = []
        for node in all_nodes:
            # 只在Call类型节点检查敏感函数
            if isinstance(node, dict) and node.get("type") == "Call":
                # 直接从节点中获取func_name，如果存在
                func_name = node.get("func_name")
                
                # 如果节点中没有直接提供func_name，则尝试提取
                if not func_name:
                    func_name = self._get_function_name(node)
                
                if func_name and isinstance(func_name, str):
                    # 检查是否是敏感函数（所有敏感函数现在都在BUILTIN_SENSITIVE中）
                    if func_name in BUILTIN_SENSITIVE or any(func_name.endswith(f".{s.split('.')[-1]}") for s in BUILTIN_SENSITIVE if '.' in s):
                        sensitive_calls.append(func_name)
                        print(f"发现敏感函数调用: {func_name}")
        
        print(f"找到 {len(sensitive_calls)} 个敏感函数调用")
        return sensitive_calls
    
    def _collect_all_nodes(self, node: Any, nodes: List[Dict] = None) -> List[Dict]:
        """
        收集AST中的所有节点
        
        参数:
            node: 当前节点
            nodes: 已收集的节点列表
            
        返回:
            包含所有节点的列表
        """
        if nodes is None:
            nodes = []
            
        if node is None:
            return nodes
        
        # 如果是AST根节点，先尝试获取ast字段
        if isinstance(node, dict) and "ast" in node:
            return self._collect_all_nodes(node["ast"], nodes)
            
        # 添加当前节点到节点列表（如果是字典类型）
        if isinstance(node, dict):
            nodes.append(node)
            
            # 递归处理所有可能的子结构
            # 1. 首先处理children列表
            if "children" in node and isinstance(node["children"], list):
                for child in node["children"]:
                    self._collect_all_nodes(child, nodes)
            
            # 2. 处理其他可能包含节点的键值对
            for key, value in node.items():
                if key != "children":  # 已经处理过children
                    self._collect_all_nodes(value, nodes)
                    
        # 处理列表类型
        elif isinstance(node, list):
            for item in node:
                self._collect_all_nodes(item, nodes)
                
        return nodes
    
    def _get_function_name(self, call_node: Dict[str, Any]) -> Optional[str]:
        """
        从Call节点中提取函数名
        
        参数:
            call_node: 函数调用节点
            
        返回:
            函数名字符串或None
        """
        # 首先检查是否有预处理的func_name
        if "func_name" in call_node:
            return call_node["func_name"]
            
        # 策略1: 检查节点的children中的第一个子节点（通常是函数名）
        children = call_node.get("children", [])
        if children and len(children) > 0:
            first_child = children[0]
            if isinstance(first_child, dict):
                # 如果是Name节点，直接获取id
                if first_child.get("type") == "Name" and "id" in first_child:
                    return first_child["id"]
                # 如果是Attribute节点，组合module.attr
                elif first_child.get("type") == "Attribute":
                    attr = first_child.get("attr")
                    # 尝试获取模块名
                    if attr and "value" in first_child and isinstance(first_child["value"], dict):
                        value = first_child["value"]
                        if value.get("type") == "Name" and "id" in value:
                            return f"{value['id']}.{attr}"
                    
                    # 尝试从子节点获取模块名
                    module_name = None
                    for attr_child in first_child.get("children", []):
                        if isinstance(attr_child, dict) and attr_child.get("type") == "Name" and "id" in attr_child:
                            module_name = attr_child["id"]
                        elif "attr" in attr_child:
                            attr = attr_child["attr"]
                    
                    if module_name and attr:
                        return f"{module_name}.{attr}"
                    elif attr:
                        return attr
        
        # 策略2: 从func字段获取
        if "func" in call_node and isinstance(call_node["func"], dict):
            func = call_node["func"]
            if func.get("type") == "Name" and "id" in func:
                return func["id"]
            elif func.get("type") == "Attribute" and "attr" in func:
                attr = func["attr"]
                if "value" in func and isinstance(func["value"], dict):
                    value = func["value"]
                    if value.get("type") == "Name" and "id" in value:
                        return f"{value['id']}.{attr}"
        
        # 策略3: 遍历所有子节点寻找名称
        if children:
            for child in children:
                if not isinstance(child, dict):
                    continue
                
                if child.get("type") == "Name" and "id" in child:
                    return child["id"]
                elif child.get("type") == "Attribute":
                    attr = child.get("attr")
                    if attr:
                        # 从value获取模块名
                        if "value" in child and isinstance(child["value"], dict):
                            value = child["value"]
                            if value.get("type") == "Name" and "id" in value:
                                return f"{value['id']}.{attr}"
                        
                        # 从子节点获取模块名
                        for attr_child in child.get("children", []):
                            if isinstance(attr_child, dict) and attr_child.get("type") == "Name" and "id" in attr_child:
                                return f"{attr_child['id']}.{attr}"
        
        return None
    
    def _extract_ngram_features(self) -> np.ndarray:
        """
        提取AST节点类型的N-gram序列特征
        
        返回:
            64维N-gram序列特征向量
        """
        # 提取AST节点类型序列
        node_types = self._extract_node_type_sequence(self.ast_json["ast"])
        
        # 生成N-gram序列（N=3）
        ngrams = []
        for i in range(len(node_types) - 2):
            ngram = " ".join(node_types[i:i+3])
            ngrams.append(ngram)
        
        # 如果序列太短，添加填充
        if len(ngrams) < 3:
            ngrams.extend(["PAD PAD PAD"] * (3 - len(ngrams)))
        
        # 使用TF-IDF向量化
        vectorizer = TfidfVectorizer(max_features=64, norm='l2')
        
        # 创建一个语料库，包含当前文件的N-gram序列
        corpus = [" ".join(ngrams)]
        
        # 如果语料库为空，返回零向量
        if not corpus[0]:
            return np.zeros(64)
        
        try:
            # 转换为TF-IDF特征
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # 获取特征向量
            features = tfidf_matrix.toarray()[0]
            
            # 如果特征维度不足64，填充零
            if len(features) < 64:
                features = np.pad(features, (0, 64 - len(features)))
            
            return features
        except:
            # 如果向量化失败，返回零向量
            return np.zeros(64)
    
    def _extract_node_type_sequence(self, node: Dict[str, Any]) -> List[str]:
        """
        递归提取AST节点类型序列
        
        参数:
            node: AST节点
            
        返回:
            节点类型序列
        """
        sequence = [node["type"]]
        
        # 递归处理子节点
        for child in node.get("children", []):
            sequence.extend(self._extract_node_type_sequence(child))
        
        return sequence
    
    def _extract_pattern_features(self) -> np.ndarray:
        """
        提取代码混淆模式特征
        
        返回:
            32维混淆模式特征向量
        """
        # 初始化特征向量
        features = np.zeros(32)
        
        # 如果没有源代码，返回零向量
        if not self.source_code:
            print("警告：没有可用的源代码，返回零混淆模式特征向量")
            return features
        
        # 计算代码的熵值作为一个基本特征
        code_entropy = self._calculate_entropy(self.source_code)
        print(f"代码熵值: {code_entropy:.4f}")
        
        # 统计每种模式的匹配次数
        pattern_counts = {}
        for pattern_name, pattern_regex in CODE_PATTERNS.items():
            matches = re.findall(pattern_regex, self.source_code)
            pattern_counts[pattern_name] = len(matches)
            if len(matches) > 0:
                print(f"找到混淆模式 '{pattern_name}': {len(matches)}个匹配")
        
        # 检查高危模式
        high_risk_counts = {}
        for category, patterns in HIGH_RISK_PATTERNS.items():
            category_matches = []
            for pattern in patterns:
                matches = re.findall(pattern, self.source_code)
                category_matches.extend(matches)
            high_risk_counts[category] = len(category_matches)
            if len(category_matches) > 0:
                print(f"高危类别 '{category}': {len(category_matches)}个匹配")
        
        # 将模式分为8类
        pattern_categories = {
            "encoding_decoding": ["base64_decode", "hex_decode", "rot13", "char_code"],
            "string_manipulation": ["string_concat", "string_join", "string_format", "string_replace"],
            "code_execution": ["eval_exec", "compile_code"],
            "dynamic_loading": ["dynamic_import"],
            "system_interaction": ["os_system", "subprocess_call"],
            "file_operations": ["file_open", "file_read", "file_write"],
            "network_operations": ["network_connect"],
            "obfuscation_techniques": ["obfuscated_control", "lambda_obfuscation", "sleep_pattern", 
                                      "random_pattern", "environment_check"]
        }
        
        # 为每个类别计算特征
        for i, (category, patterns) in enumerate(pattern_categories.items()):
            # 计算该类别的总匹配次数
            category_count = sum(pattern_counts.get(pattern, 0) for pattern in patterns)
            
            # 如果有高危测试文件，增强某些类别的特征
            if category in high_risk_counts:
                high_risk_factor = min(1.0, high_risk_counts[category] / 10.0)  # 归一化到0-1之间
                category_count = max(category_count, int(high_risk_factor * 10))
            
            # 计算该类别中匹配的模式数量
            matched_patterns = sum(1 for pattern in patterns if pattern_counts.get(pattern, 0) > 0)
            
            # 计算最大匹配次数
            max_count = max([pattern_counts.get(pattern, 0) for pattern in patterns], default=0)
            
            # 使用代码熵值来增强特定类别的特征
            if category in ["obfuscation_techniques", "encoding_decoding"] and code_entropy > 4.5:
                entropy_factor = min(1.0, (code_entropy - 4.5) / 1.5)  # 归一化到0-1之间
                category_count = max(category_count, int(entropy_factor * 10))
            
            # 填充特征向量
            features[i * 4:i * 4 + 4] = [
                1.0 if category_count > 0 else 0.0,  # 是否存在该类别的模式
                np.log1p(category_count),            # 总匹配次数（对数缩放）
                matched_patterns / len(patterns) if patterns else 0,  # 匹配模式比例
                np.log1p(max_count)                  # 最大匹配次数（对数缩放）
            ]
        
        return features
    
    def _calculate_entropy(self, text: str) -> float:
        """
        计算文本的熵值，作为混淆复杂度的指标
        
        参数:
            text: 输入文本
            
        返回:
            熵值（比特/字符）
        """
        if not text:
            return 0.0
        
        # 计算每个字符的频率
        char_freq = {}
        for char in text:
            if char in char_freq:
                char_freq[char] += 1
            else:
                char_freq[char] = 1
        
        # 计算文本长度
        length = len(text)
        
        # 计算熵值
        entropy = 0.0
        for freq in char_freq.values():
            probability = freq / length
            entropy -= probability * np.log2(probability)
        
        return entropy
    
    def detect_obfuscation(self) -> Dict[str, Any]:
        """
        检测代码是否使用了混淆技术
        
        返回:
            包含混淆检测结果的字典
        """
        if not self.source_code:
            return {"obfuscated": False, "reason": "无可分析代码"}
        
        # 计算熵值
        entropy = float(self._calculate_entropy(self.source_code))  # 确保是Python float
        
        # 计算平均行长度
        lines = [line for line in self.source_code.split('\n') if line.strip()]
        avg_line_length = float(sum(len(line) for line in lines) / len(lines) if lines else 0)  # 确保是Python float
        
        # 检测混淆特征
        obfuscation_indicators = {
            "高熵值": bool(entropy > 5.0),  # 确保是Python bool
            "长行": bool(avg_line_length > 100),  # 确保是Python bool
            "base64编码": bool(re.search(r'base64\.(b64decode|decodestring)', self.source_code)),
            "eval/exec": bool(re.search(r'eval\(|exec\(', self.source_code)),
            "变量名混淆": bool(re.search(r'\b[a-zA-Z]{1,2}\d*\b', self.source_code) and 
                           not re.search(r'def\s+[a-zA-Z]{3,}', self.source_code)),
            "字符串拼接": bool(self.source_code.count('+') > 20 and self.source_code.count("'") > 20)
        }
        
        # 判断是否混淆
        is_obfuscated = bool(sum(1 for val in obfuscation_indicators.values() if val) >= 2)  # 确保是Python bool
        
        reasons = [key for key, val in obfuscation_indicators.items() if val]
        
        result = {
            "obfuscated": is_obfuscated,
            "entropy": entropy,
            "avg_line_length": avg_line_length,
            "indicators": obfuscation_indicators,
            "reasons": reasons
        }
        
        if is_obfuscated:
            print(f"检测到代码混淆，原因: {', '.join(reasons)}")
        
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
            print(f"创建输出目录: {output_dir}")
        
        # 检查文件是否已存在
        if os.path.exists(output_file):
            print(f"文件 {output_file} 已存在，将被覆盖")
        
        # 提取特征并保存
        features = self.extract_features()
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")
        
        # 检查文件是否已存在
        if os.path.exists(output_file):
            print(f"文件 {output_file} 已存在，将被覆盖")
            
        np.save(output_file, features)
        print(f"语义特征向量已保存至: {output_file}")


def process_file(source_file: str, output_file: Optional[str] = None, save_obfuscation: bool = False) -> None:
    """
    处理Python源代码文件并提取语义特征
    
    参数:
        source_file: Python源代码文件路径
        output_file: 输出文件路径（可选）
        save_obfuscation: 是否保存混淆检测结果（默认为False）
    """
    # 确定输出文件名
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(source_file))[0]
        output_file = f"{base_name}_semantic.npy"
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 提取特征
    extractor = SemanticFeatureExtractor(source_file)
    
    # 检测混淆
    obfuscation_result = extractor.detect_obfuscation()
    
    # 仅当save_obfuscation为True时才保存混淆检测结果
    if save_obfuscation:
        # 混淆检测结果文件路径
        obfuscation_file = f"{os.path.splitext(output_file)[0]}_obfuscation.json"
        
        # 尝试保存混淆检测结果
        try:
            # 使用自定义编码器处理NumPy类型
            result_json = json.dumps(obfuscation_result, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            print(f"混淆检测结果: {result_json}")
            
            # 检查混淆检测结果文件是否已存在
            if os.path.exists(obfuscation_file):
                print(f"文件 {obfuscation_file} 已存在，将被覆盖")
                
            with open(obfuscation_file, 'w', encoding='utf-8') as f:
                f.write(result_json)
            print(f"混淆检测结果已保存至: {obfuscation_file}")
            
        except Exception as error:
            # 如果JSON序列化或文件保存出错，提供简单输出
            print(f"混淆检测结果: 是否混淆={obfuscation_result['obfuscated']}, " 
                f"原因={obfuscation_result.get('reasons', [])}")
            print(f"警告: JSON序列化或保存失败 - {str(error)}")
    
    # 保存特征
    try:
        extractor.save_features(output_file)
    except Exception as error:
        print(f"错误: 保存特征向量失败 - {str(error)}")
        sys.exit(1)


def process_directory(input_dir: str, output_dir: Optional[str] = None, save_obfuscation: bool = False) -> None:
    """
    处理目录中的所有Python文件并提取语义特征
    
    参数:
        input_dir: 输入目录路径，包含Python源代码文件
        output_dir: 输出目录路径（可选）
        save_obfuscation: 是否保存混淆检测结果（默认为False）
    """
    # 导入进度条库
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        print("提示: 安装 tqdm 库可以获得更好的进度展示效果 (pip install tqdm)")
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir), 
                                  f"{os.path.basename(input_dir)}_semantic_features")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"将输出特征向量保存到目录: {output_dir}")
    
    # 遍历输入目录中的所有Python文件
    py_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    
    if not py_files:
        print(f"警告: 在目录 {input_dir} 中没有找到Python文件")
        return
    
    print(f"找到 {len(py_files)} 个Python文件需要处理")
    
    # 使用进度条包装文件列表
    if has_tqdm:
        py_files_iter = tqdm(py_files, desc="处理文件", unit="个")
    else:
        py_files_iter = py_files
        # 简单的进度显示
        print("开始处理文件:")
    
    # 成功和失败的计数器
    success_count = 0
    failure_count = 0
    
    # 处理每个Python文件
    for i, py_file in enumerate(py_files_iter, 1):
        # 计算相对路径以保持目录结构
        rel_path = os.path.relpath(py_file, input_dir)
        output_file = os.path.join(output_dir, 
                                   f"{os.path.splitext(rel_path)[0]}_semantic.npy")
        
        # 确保输出文件的目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 如果没有tqdm, 显示简单进度
        if not has_tqdm:
            progress_percent = (i / len(py_files)) * 100
            print(f"[{i}/{len(py_files)}] {progress_percent:.1f}% - 处理: {os.path.basename(py_file)}")
        
        try:
            process_file(py_file, output_file, save_obfuscation)
            success_count += 1
            if not has_tqdm:
                print(f"✓ 成功处理: {os.path.basename(py_file)}")
        except Exception as e:
            failure_count += 1
            if has_tqdm:
                tqdm.write(f"处理文件 {py_file} 时出错: {str(e)}")
            else:
                print(f"✗ 错误: 处理文件 {py_file} 失败: {str(e)}")
            # 继续处理下一个文件，不中断整个流程
    
    # 处理完成后的摘要
    print(f"\n处理完成摘要:")
    print(f"- 总文件数: {len(py_files)}")
    print(f"- 成功处理: {success_count}")
    print(f"- 处理失败: {failure_count}")
    print(f"特征向量已保存到目录: {output_dir}")


def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description="从Python源代码提取语义特征向量")
    
    # 创建互斥组，确保用户只能指定文件或目录，不能同时指定
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", help="Python源代码文件路径")
    group.add_argument("-d", "--directory", help="包含多个Python源代码文件的目录路径")
    
    parser.add_argument("-o", "--output", help="输出文件或目录路径（可选）")
    
    # 添加保存混淆检测结果的选项
    parser.add_argument("--save-obfuscation", action="store_true", 
                      help="保存混淆检测结果到JSON文件")
    
    # 添加进度条相关选项
    parser.add_argument("--no-progress", action="store_true", 
                      help="禁用进度条显示")
    
    args = parser.parse_args()
    
    # 如果指定禁用进度条，设置环境变量
    if args.no_progress:
        os.environ["TQDM_DISABLE"] = "1"
    
    if args.file:
        # 处理单个文件
        process_file(args.file, args.output, args.save_obfuscation)
    else:
        # 处理整个目录
        process_directory(args.directory, args.output, args.save_obfuscation)


if __name__ == "__main__":
    main()
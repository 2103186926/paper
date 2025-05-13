#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成Python恶意代码测试文件
用于生成各种类型的恶意代码测试样本
"""

import os
import sys
import random
import argparse
import string
from typing import List, Dict, Callable
import shutil
import getpass


class MaliciousCodeGenerator:
    """恶意代码生成器，用于批量生成各类型的恶意代码测试文件"""

    def __init__(self):
        """初始化恶意代码生成器"""
        # 定义恶意代码类型映射
        self.code_types = {
            1: "execute",        # 不安全的代码执行
            2: "deserialize",    # 不安全的反序列化
            3: "dynamic_import", # 不安全的动态导入
            4: "file_ops",       # 不安全的文件操作
            5: "network",        # 不安全的网络交互
            6: "system",         # 不安全的系统交互
            7: "user_input",     # 不安全的用户输入
            8: "mixed",          # 混合调用
            9: "safe"            # 安全的Python代码
        }
        
        # 生成随机变量名的字符集
        self.chars = string.ascii_lowercase + string.digits
        
        # 确保您的方法名与这里的引用一致
        # 请检查每个方法名是否正确
        self.generators = {
            1: self.generate_unsafe_code_execution,  # 去掉下划线前缀
            2: self.generate_unsafe_deserialization,
            3: self.generate_unsafe_dynamic_import,
            4: self.generate_unsafe_file_operations,
            5: self.generate_unsafe_network_interactions,
            6: self.generate_unsafe_system_interactions,
            7: self.generate_unsafe_user_input,
            8: self.generate_mixed_code,
            9: self.generate_safe_code  # 添加对应的安全代码生成方法
        }
        
        # 设置默认的混淆级别，避免过度混淆导致生成缓慢
        self.default_obfuscation_level = 1

    def _random_var_name(self, prefix: str = "", length: int = None) -> str:
        """生成随机变量名，确保不以数字开头"""
        if length is None:
            length = random.randint(3, 10)
        
        # 确保首字符是字母
        first_char = random.choice(string.ascii_lowercase)
        
        # 剩余字符可以是字母或数字
        if length > 1:
            suffix = ''.join(random.choice(self.chars) for _ in range(length - 1))
            result = first_char + suffix
        else:
            result = first_char
            
        # 添加前缀（如果有）
        if prefix:
            result = f"{prefix}{result}"
            
        return result

    def _random_string(self, length: int = None) -> str:
        """生成随机字符串"""
        if length is None:
            length = random.randint(5, 20)
        return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

    def _add_comments(self, code: str) -> str:
        """添加随机注释"""
        comments = [
            "# 执行操作，小心使用",
            "# 危险操作，仅用于测试",
            "# 注意：此代码仅用于安全测试环境",
            "# WARNING: This is a malicious code sample for testing",
            "# DANGER: Use with caution",
            "# 安全风险演示",
            "# 用于测试安全检测工具的有效性",
            "# 模拟攻击者行为",
            "# 不要在生产环境中运行"
        ]
        
        lines = code.split('\n')
        commented_lines = []
        
        for line in lines:
            if random.random() < 0.2:  # 20%的概率添加注释
                commented_lines.append(random.choice(comments))
            commented_lines.append(line)
            
        return '\n'.join(commented_lines)

    def _add_obfuscation(self, code: str, level: int = None) -> str:
        """添加代码混淆"""
        if level is None:
            level = self.default_obfuscation_level
        if level == 0:
            return code
            
        # 基本混淆技术
        if level >= 1:
            # 随机添加无用变量
            var_count = random.randint(2, 5)
            obfuscation_vars = []
            for _ in range(var_count):
                var_name = self._random_var_name()
                var_value = f'"{self._random_string()}"'
                obfuscation_vars.append(f"{var_name} = {var_value}")
            
            # 在代码开头添加混淆变量
            code = '\n'.join(obfuscation_vars) + '\n\n' + code
        
        # 中级混淆（添加无用的条件和循环）
        if level >= 2:
            junk_code = []
            junk_code.append(f"for _ in range({random.randint(1, 3)}):")
            junk_code.append(f"    if {random.random()} > {random.random()}:")
            junk_code.append(f"        {self._random_var_name()} = {random.randint(1, 100)}")
            junk_code.append(f"    else:")
            junk_code.append(f"        {self._random_var_name()} = '{self._random_string()}'")
            
            # 在代码中间随机位置插入垃圾代码
            lines = code.split('\n')
            insert_pos = random.randint(0, len(lines))
            lines = lines[:insert_pos] + junk_code + lines[insert_pos:]
            code = '\n'.join(lines)
        
        # 高级混淆（字符串编码，代码拆分等）
        if level >= 3:
            # 随机选择一些字符串进行base64编码
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if '"' in line or "'" in line and random.random() < 0.3:
                    line_parts = line.split('"') if '"' in line else line.split("'")
                    if len(line_parts) >= 3:  # 确保有字符串
                        import_stmt = "import base64"
                        if import_stmt not in code:
                            lines.insert(0, import_stmt)
                        
                        # 获取字符串内容
                        string_content = line_parts[1]
                        # 创建编码字符串
                        encoded = f"base64.b64decode(b'{string_content.encode('utf-8').hex()}').decode('utf-8')"
                        # 替换原字符串
                        lines[i] = line.replace(f'"{string_content}"', encoded).replace(f"'{string_content}'", encoded)
            
            code = '\n'.join(lines)
            
        return code

    def _add_imports(self, code_type: int) -> List[str]:
        """添加必要的导入语句"""
        imports = ["import os", "import sys", "import random"]
        
        if code_type == 1:  # 代码执行
            imports.extend(["import ast", "import codecs", "import subprocess"])
        elif code_type == 2:  # 反序列化
            imports.extend(["import pickle", "import yaml", "import json", "import base64", "import subprocess"])
        elif code_type == 3:  # 动态导入
            imports.extend(["import importlib", "import pkgutil", "import inspect", "import types"])
        elif code_type == 4:  # 文件操作
            imports.extend(["import io", "import tempfile", "import shutil", "import glob", "import pathlib"])
        elif code_type == 5:  # 网络交互
            imports.extend(["import urllib.request", "import requests", "import socket", "import http.client"])
        elif code_type == 6:  # 系统交互
            imports.extend(["import subprocess", "import platform", "import shlex", "import signal", "import pwd"])
        elif code_type == 7:  # 用户输入
            imports.extend(["import getpass", "import readline", "import cmd"])
        elif code_type == 8:  # 混合
            # 随机选择5-8个导入
            all_imports = [
                "import ast", "import codecs", "import pickle", "import yaml", 
                "import json", "import importlib", "import pkgutil", "import io", 
                "import tempfile", "import shutil", "import urllib.request", 
                "import requests", "import socket", "import subprocess", 
                "import platform", "import shlex", "import getpass", "import base64",
                "import glob", "import pathlib", "import inspect", "import types",
                "import http.client", "import signal", "import pwd", "import readline",
                "import cmd"
            ]
            imports.extend(random.sample(all_imports, random.randint(5, 8)))
        
        # 随机打乱顺序，增加多样性
        random.shuffle(imports)
        return imports

    def generate_unsafe_code_execution(self) -> str:
        """生成不安全的代码执行类测试代码"""
        code_templates = [
            # eval模板
            '''
# 不安全的代码执行示例 - 使用eval
{var1} = "{code_snippet}"
{var2} = eval({var1})
print(f"执行结果: {{({var2})}}")

# 额外的恶意代码执行
{extra_var1} = "{extra_code1}"
eval({extra_var1})

# 多层嵌套的恶意执行
{extra_var2} = "eval('{extra_code2}')"
eval({extra_var2})
''',
            # exec模板
            '''
# 不安全的代码执行示例 - 使用exec
{var1} = """{code_snippet}"""
exec({var1})

# 额外的恶意代码执行
{extra_var1} = """
for i in range(3):
    print(f"执行恶意循环: {{i}}")
"""
exec({extra_var1})

# 动态构建并执行恶意代码
{extra_var2} = f"print('动态构建的代码: ' + str({random_num1}))"
exec({extra_var2})
''',
            # compile模板
            '''
# 不安全的代码执行示例 - 使用compile
{var1} = """{code_snippet}"""
{var2} = compile({var1}, '<string>', 'exec')
exec({var2})

# 额外的恶意代码编译和执行
{extra_var1} = """
def {evil_func}():
    print("恶意函数被执行")
    return {random_num2}
    
result = {evil_func}()
print(f"恶意结果: {{result}}")
"""
{extra_var2} = compile({extra_var1}, '<evildoc>', 'exec')
exec({extra_var2})
''',
            # 更复杂的eval示例
            '''
# 动态构建并执行代码
{var1} = "{expr1}"
{var2} = "{expr2}"
{var3} = f"{{{var1}}} + {{{var2}}}"
result = eval({var3})
print(f"计算结果: {{result}}")

# 通过字符串拼接构造恶意表达式
{extra_var1} = "{evil_op1}"
{extra_var2} = "{evil_op2}"
evil_expr = f"{extra_var1} {evil_op_symbol} {extra_var2}"
print(f"恶意表达式: {{evil_expr}}")
evil_result = eval(evil_expr)
print(f"恶意表达式结果: {{evil_result}}")
''',
            # exec与字符串操作结合
            '''
# 从字符串构建代码并执行
code_parts = [
    "def {func_name}():",
    "    print('{message}')",
    "    return {return_val}",
    "{func_name}()"
]
exec("\\n".join(code_parts))

# 额外的多行恶意代码构建
evil_code_parts = [
    "import math",
    "def {evil_func1}(x):",
    "    return math.sin(x) * {random_num3}",
    "",
    "def {evil_func2}(y):",
    "    if y > {random_num4}:",
    "        print('触发恶意条件')",
    "    return y * y",
    "",
    "for i in range({random_num5}):",
    "    print(f'恶意循环次数: {{i}}, 结果: {{{evil_func1}(i) + {evil_func2}(i)}}')"
]
exec("\\n".join(evil_code_parts))
'''
        ]
        
        # 随机选择一个模板
        template = random.choice(code_templates)
        
        # 准备所有可能需要的模板变量
        var1 = self._random_var_name("code_")
        var2 = self._random_var_name("result_")
        var3 = self._random_var_name("expr_")
        extra_var1 = self._random_var_name("evil_")
        extra_var2 = self._random_var_name("malicious_")
        
        # 生成随机数，而不是在模板中直接调用
        random_num1 = random.randint(1, 100)
        random_num2 = random.randint(100, 999)
        random_num3 = random.randint(10, 50)
        random_num4 = random.randint(1, 10)
        random_num5 = random.randint(1, 5)
        
        # 代码片段选项
        code_snippets = [
            "1 + 1",
            "'hello' + ' world'",
            "[x for x in range(5)]",
            "{'a': 1, 'b': 2}",
            "sum([1, 2, 3, 4, 5])",
            "os.getcwd()",
            "sys.version",
            "os.path.exists('.')",
            "os.listdir('.')",
            "random.randint(1, 100)"
        ]
        
        # 恶意代码片段
        evil_code_snippets = [
            "os.system('echo Evil command')",
            "subprocess.call(['echo', 'Malicious subprocess'])",
            "__import__('subprocess').call(['echo', 'Dynamic import subprocess'])",
            "open('evil.txt', 'w').write('Malicious file operation')",
            "getattr(os, 'system')('echo Dynamic attribute access')"
        ]
        
        code_snippet = random.choice(code_snippets)
        extra_code1 = random.choice(evil_code_snippets)
        extra_code2 = random.choice(evil_code_snippets)
        
        # 为更复杂的模板准备变量
        expr1 = random.choice(["1", "2", "3", "4", "5"])
        expr2 = random.choice(["10", "20", "30", "40", "50"])
        evil_op1 = random.choice(["5", "10", "15", "20"])
        evil_op2 = random.choice(["2", "4", "6", "8"])
        evil_op_symbol = random.choice(["+", "-", "*", "**"])
        
        func_name = self._random_var_name("func_")
        evil_func = self._random_var_name("evil_func_")
        evil_func1 = self._random_var_name("evil_calc_")
        evil_func2 = self._random_var_name("evil_proc_")
        message = self._random_string()
        return_val = random.randint(1, 100)
        
        # 填充模板 - 添加所有可能的变量，无论模板是否使用
        code = template.format(
            var1=var1, 
            var2=var2, 
            var3=var3,
            extra_var1=extra_var1,
            extra_var2=extra_var2,
            code_snippet=code_snippet,
            expr1=expr1,
            expr2=expr2,
            func_name=func_name,
            message=message,
            return_val=return_val,
            extra_code1=extra_code1,
            extra_code2=extra_code2,
            evil_func=evil_func,
            evil_func1=evil_func1,
            evil_func2=evil_func2,
            evil_op1=evil_op1,
            evil_op2=evil_op2,
            evil_op_symbol=evil_op_symbol,
            random_num1=random_num1,
            random_num2=random_num2,
            random_num3=random_num3,
            random_num4=random_num4,
            random_num5=random_num5
        )
        
        # 增加导入语句
        imports = self._add_imports(1)
        
        # 组合最终代码
        full_code = '\n'.join(imports) + '\n\n' + code
        
        # 添加混淆和注释
        full_code = self._add_obfuscation(full_code, random.randint(0, 1))
        full_code = self._add_comments(full_code)
        
        return full_code

    def generate_unsafe_deserialization(self) -> str:
        """生成不安全的反序列化类测试代码"""
        code_templates = [
            # pickle反序列化
            '''
# 不安全的反序列化示例 - 使用pickle
{var1} = b"{pickle_data}"
{var2} = pickle.loads({var1})
print(f"反序列化结果: {{({var2})}}")

# 额外的pickle恶意负载
{extra_var1} = pickle.dumps({{"command": "echo Malicious pickle", "args": ["-l", "-a"]}})
{extra_var2} = pickle.loads({extra_var1})
print(f"恶意pickle数据: {{({extra_var2})}}")

# 使用字节串构造恶意pickle
{extra_var3} = b"{pickle_data2}"
try:
    {extra_var4} = pickle.loads({extra_var3})
    print(f"额外恶意pickle结果: {{({extra_var4})}}")
except Exception as e:
    print(f"恶意pickle加载失败: {{e}}")
''',
            # yaml反序列化
            '''
# 不安全的反序列化示例 - 使用yaml
{var1} = """{yaml_data}"""
{var2} = yaml.load({var1}, Loader=yaml.Loader)  # 不安全的加载器
print(f"YAML解析结果: {{({var2})}}")

# 额外的YAML恶意负载
{extra_var1} = """
evil_object: !!python/object/apply:os.system ['echo "Second YAML payload"']
another_evil:
  !!python/object/apply:subprocess.check_output ['echo "Nested YAML payload"']
"""
try:
    {extra_var2} = yaml.load({extra_var1}, Loader=yaml.Loader)
    print(f"额外YAML解析结果: {{({extra_var2})}}")
except Exception as e:
    print(f"恶意YAML解析失败: {{e}}")

# 构造恶意的YAML类
{extra_var3} = """!!python/object/new:type
args: ['EvilClass', (), {{'execute': !!python/name:exec}}]
"""
try:
    {extra_var4} = yaml.load({extra_var3}, Loader=yaml.Loader)
    print(f"危险的YAML类构造: {{type({extra_var4})}}")
except Exception as e:
    print(f"恶意YAML类构造失败: {{e}}")
''',
            # json反序列化
            '''
# JSON反序列化示例
{var1} = """{json_data}"""
{var2} = json.loads({var1})
print(f"JSON解析结果: {{({var2})}}")

# 使用json构造恶意配置
{extra_var1} = """{{
  "type": "malicious",
  "command": "echo 'JSON configured command'",
  "args": ["-a", "-b", "-c"],
  "env": {{"PATH": "/bin:/usr/bin", "USER": "root"}},
  "permissions": "0777",
  "payload": "ZWNobyAiZXZpbCBwYXlsb2FkIgo="
}}"""
{extra_var2} = json.loads({extra_var1})
print(f"恶意JSON配置: {{({extra_var2})}}")

# 处理恶意JSON命令
if {extra_var2}.get("type") == "malicious":
    cmd = {extra_var2}.get("command")
    print(f"从JSON提取的恶意命令: {{cmd}}")
    # 以下行实际执行命令，此处只是示范
    # os.system(cmd)
'''
        ]
        
        # 随机选择一个模板
        template = random.choice(code_templates)
        
        # 准备模板变量
        var1 = self._random_var_name("data_")
        var2 = self._random_var_name("result_")
        extra_var1 = self._random_var_name("evil_data_")
        extra_var2 = self._random_var_name("evil_result_")
        extra_var3 = self._random_var_name("extra_payload_")
        extra_var4 = self._random_var_name("extra_result_")
        
        # 示例数据
        pickle_data = "80 03 63 62 75 69 6c 74 69 6e 73 0a 65 76 61 6c 0a 71 00 58 0c 00 00 00 70 72 69 6e 74 28 27 48 69 27 29 71 01 85 71 02 52 71 03 2e"
        pickle_data2 = "80 03 63 6f 73 0a 73 79 73 74 65 6d 0a 71 00 58 1e 00 00 00 65 63 68 6f 20 22 4d 61 6c 69 63 69 6f 75 73 20 70 69 63 6b 6c 65 20 64 61 74 61 22 71 01 85 71 02 52 71 03 2e"
        yaml_data = "!!python/object/apply:os.system ['echo \"Hello from YAML\"']"
        json_data = '{"name": "test", "value": 42, "command": "echo \\"Hello from JSON\\""}'
        
        # 填充模板
        code = template.format(
            var1=var1, 
            var2=var2,
            extra_var1=extra_var1,
            extra_var2=extra_var2,
            extra_var3=extra_var3,
            extra_var4=extra_var4,
            pickle_data=pickle_data,
            pickle_data2=pickle_data2,
            yaml_data=yaml_data,
            json_data=json_data
        )
        
        # 增加导入语句
        imports = self._add_imports(2)
        
        # 组合最终代码
        full_code = '\n'.join(imports) + '\n\n' + code
        
        # 添加混淆和注释
        full_code = self._add_obfuscation(full_code, random.randint(0, 1))
        full_code = self._add_comments(full_code)
        
        return full_code

    def generate_unsafe_dynamic_import(self) -> str:
        """生成不安全的动态导入类测试代码"""
        code_templates = [
            # __import__
            '''
# 不安全的动态导入示例 - 使用__import__
{var1} = "{module_name}"
{var2} = __import__({var1})
print(f"导入的模块: {{({var2}.__name__)}}")
{var3} = getattr({var2}, "{attr_name}")
print(f"获取的属性: {{({var3})}}")
''',
            # importlib.import_module
            '''
# 不安全的动态导入示例 - 使用importlib.import_module
{var1} = "{module_name}"
{var2} = importlib.import_module({var1})
print(f"导入的模块: {{({var2}.__name__)}}")
''',
            # 动态导入与执行结合
            '''
# 动态导入并执行模块函数
{var1} = "{module_name}"
{var2} = __import__({var1})
{var3} = getattr({var2}, "{func_name}")
result = {var3}({func_args})
print(f"函数执行结果: {{result}}")
'''
        ]
        
        # 随机选择一个模板
        template = random.choice(code_templates)
        
        # 准备模板变量
        var1 = self._random_var_name("module_name_")
        var2 = self._random_var_name("module_")
        var3 = self._random_var_name("attr_")
        
        # 模块和属性选项
        module_names = ["os", "sys", "random", "platform", "math", "datetime"]
        attr_names = ["name", "path", "version", "platform", "getcwd", "listdir"]
        func_names = ["getcwd", "listdir", "random", "choice", "system"]
        func_args = ["'.'", "1, 10", "[1, 2, 3]"]
        
        module_name = random.choice(module_names)
        attr_name = random.choice(attr_names)
        func_name = random.choice(func_names)
        func_arg = random.choice(func_args)
        
        # 填充模板
        code = template.format(
            var1=var1, 
            var2=var2, 
            var3=var3,
            module_name=module_name,
            attr_name=attr_name,
            func_name=func_name,
            func_args=func_arg
        )
        
        # 增加导入语句
        imports = self._add_imports(3)
        
        # 组合最终代码
        full_code = '\n'.join(imports) + '\n\n' + code
        
        # 添加混淆和注释
        full_code = self._add_obfuscation(full_code, random.randint(0, 1))
        full_code = self._add_comments(full_code)
        
        return full_code

    def generate_unsafe_file_operations(self) -> str:
        """生成不安全的文件操作类测试代码"""
        code_templates = [
            # 基本文件读取
            '''
# 不安全的文件操作示例 - 读取文件
{var1} = "{file_path}"
with open({var1}, "r") as {var2}:
    {var3} = {var2}.read()
    print(f"文件内容: {{({var3}[:100])}}")
''',
            # 文件写入
            '''
# 不安全的文件操作示例 - 写入文件
{var1} = "{file_path}"
{var2} = """{file_content}"""
with open({var1}, "w") as {var3}:
    {var3}.write({var2})
print(f"已写入文件: {{({var1})}}")
''',
            # 文件读写组合
            '''
# 文件读写操作组合
{var1} = "{input_file}"
{var2} = "{output_file}"

# 读取文件
with open({var1}, "r") as {var3}:
    {var4} = {var3}.read()
    
# 处理内容
{var5} = {var4}.upper()

# 写入新文件
with open({var2}, "w") as {var6}:
    {var6}.write({var5})
    
print(f"处理完成: {{({var1})}} -> {{({var2})}}")
''',
            # 危险路径遍历
            '''
# 危险的路径遍历操作
{var1} = "../{dangerous_path}"
if os.path.exists({var1}):
    with open({var1}, "r") as {var2}:
        {var3} = {var2}.read()
        print(f"敏感文件内容: {{({var3}[:100])}}")
else:
    print(f"文件不存在: {{({var1})}}")
'''
        ]
        
        # 随机选择一个模板
        template = random.choice(code_templates)
        
        # 准备模板变量
        var1 = self._random_var_name("file_path_")
        var2 = self._random_var_name("file_")
        var3 = self._random_var_name("content_")
        var4 = self._random_var_name("data_")
        var5 = self._random_var_name("processed_")
        var6 = self._random_var_name("output_")
        
        # 文件路径和内容
        file_paths = ["test.txt", "data.json", "config.ini", "log.txt"]
        dangerous_paths = ["etc/passwd", "etc/shadow", "var/log/syslog", "root/.ssh/id_rsa"]
        file_contents = [
            "Hello, this is a test file.",
            "{'name': 'test', 'value': 42}",
            "USER=admin\nPASSWORD=secret\nAPI_KEY=1234567890"
        ]
        
        file_path = random.choice(file_paths)
        dangerous_path = random.choice(dangerous_paths)
        file_content = random.choice(file_contents)
        input_file = random.choice(file_paths)
        output_file = f"output_{random.randint(1, 1000)}.txt"
        
        # 填充模板
        code = template.format(
            var1=var1, 
            var2=var2, 
            var3=var3,
            var4=var4,
            var5=var5,
            var6=var6,
            file_path=file_path,
            file_content=file_content,
            input_file=input_file,
            output_file=output_file,
            dangerous_path=dangerous_path
        )
        
        # 增加导入语句
        imports = self._add_imports(4)
        
        # 组合最终代码
        full_code = '\n'.join(imports) + '\n\n' + code
        
        # 添加混淆和注释
        full_code = self._add_obfuscation(full_code, random.randint(0, 1))
        full_code = self._add_comments(full_code)
        
        return full_code

    def generate_unsafe_network_interactions(self) -> str:
        """生成不安全的网络交互类测试代码"""
        code_templates = [
            # urllib请求 - 添加注释，防止实际执行
            '''
# 不安全的网络交互示例 - urllib
{var1} = "{url}"
# 以下代码在实际环境中会执行网络请求，此处仅作示例
# {var2} = urllib.request.urlopen({var1})
# {var3} = {var2}.read()
print(f"模拟请求URL: {{({var1})}}")
''',
            # requests GET请求 - 添加注释，防止实际执行
            '''
# 不安全的网络交互示例 - requests.get
{var1} = "{url}"
# 以下代码在实际环境中会执行网络请求，此处仅作示例
# {var2} = requests.get({var1})
print(f"模拟GET请求: {{({var1})}}")
''',
            # requests POST请求 - 添加注释，防止实际执行
            '''
# 不安全的网络交互示例 - requests.post
{var1} = "{url}"
{var2} = {{"username": "{username}", "password": "{password}"}}
# 以下代码在实际环境中会执行网络请求，此处仅作示例
# {var3} = requests.post({var1}, data={var2})
print(f"模拟POST请求: {{({var1})}} 数据: {{({var2})}}")
''',
            # socket连接 - 添加注释，防止实际执行
            '''
# 不安全的网络交互示例 - socket
{var1} = "{host}"
{var2} = {port}
# 以下代码在实际环境中会建立socket连接，此处仅作示例
# {var3} = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# {var3}.connect(({var1}, {var2}))
print(f"模拟Socket连接: {{({var1})}}:{{({var2})}}")
'''
        ]
        
        # 随机选择一个模板
        template = random.choice(code_templates)
        
        # 准备模板变量
        var1 = self._random_var_name("url_")
        var2 = self._random_var_name("response_")
        var3 = self._random_var_name("data_")
        var4 = self._random_var_name("received_")
        
        # URL和其他网络参数
        urls = [
            "http://example.com",
            "https://httpbin.org/get",
            "https://postman-echo.com/get",
            "http://info.cern.ch"
        ]
        hosts = ["example.com", "httpbin.org", "google.com"]
        ports = [80, 443, 8080]
        usernames = ["admin", "user", "test"]
        passwords = ["password", "123456", "secret"]
        
        url = random.choice(urls)
        host = random.choice(hosts)
        port = random.choice(ports)
        username = random.choice(usernames)
        password = random.choice(passwords)
        
        # 填充模板
        code = template.format(
            var1=var1, 
            var2=var2, 
            var3=var3,
            var4=var4,
            url=url,
            host=host,
            port=port,
            username=username,
            password=password
        )
        
        # 增加导入语句
        imports = self._add_imports(5)
        
        # 组合最终代码
        full_code = '\n'.join(imports) + '\n\n' + code
        
        # 添加混淆和注释
        full_code = self._add_obfuscation(full_code, random.randint(0, 1))
        full_code = self._add_comments(full_code)
        
        return full_code

    def generate_unsafe_system_interactions(self) -> str:
        """生成不安全的系统交互类测试代码"""
        code_templates = [
            # os.system - 添加注释，防止实际执行
            '''
# 不安全的系统交互示例 - os.system
{var1} = "{command}"
# 以下代码在实际环境中会执行系统命令，此处仅作示例
# {var2} = os.system({var1})
print(f"模拟执行系统命令: {{({var1})}}")
''',
            # os.popen
            '''
# 不安全的系统交互示例 - os.popen
{var1} = "{command}"
{var2} = os.popen({var1})
{var3} = {var2}.read()
print(f"命令输出: {{({var3})}}")
''',
            # subprocess.call
            '''
# 不安全的系统交互示例 - subprocess.call
{var1} = "{command}".split()
{var2} = subprocess.call({var1})
print(f"子进程返回码: {{({var2})}}")
''',
            # subprocess.Popen
            '''
# 不安全的系统交互示例 - subprocess.Popen
{var1} = "{command}"
{var2} = subprocess.Popen({var1}, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
{var3}, {var4} = {var2}.communicate()
print(f"标准输出: {{({var3}.decode())}}")
print(f"标准错误: {{({var4}.decode())}}")
'''
        ]
        
        # 随机选择一个模板
        template = random.choice(code_templates)
        
        # 准备模板变量
        var1 = self._random_var_name("cmd_")
        var2 = self._random_var_name("status_")
        var3 = self._random_var_name("output_")
        var4 = self._random_var_name("error_")
        
        # 命令选项
        commands = [
            "echo Hello World",
            "ls -la",
            "dir",
            "whoami",
            "cat /etc/passwd",
            "type C:\\Windows\\System32\\drivers\\etc\\hosts",
            "ping -c 1 example.com",
            "uname -a",
            "systeminfo"
        ]
        
        command = random.choice(commands)
        
        # 填充模板
        code = template.format(
            var1=var1, 
            var2=var2, 
            var3=var3,
            var4=var4,
            command=command
        )
        
        # 增加导入语句
        imports = self._add_imports(6)
        
        # 组合最终代码
        full_code = '\n'.join(imports) + '\n\n' + code
        
        # 添加混淆和注释
        full_code = self._add_obfuscation(full_code, random.randint(0, 1))
        full_code = self._add_comments(full_code)
        
        return full_code

    def generate_unsafe_user_input(self) -> str:
        """生成不安全的用户输入类测试代码"""
        code_templates = [
            # 基本输入 - 替换为模拟代码
            '''
# 不安全的用户输入示例 - input
# 以下代码在实际运行时会等待用户输入，此处用模拟代码替代
# {var1} = input("请输入您的姓名: ")
{var1} = "模拟用户输入"  # 用固定值替代真实输入
print(f"你好, {{({var1})}}")
''',
            # 输入与eval结合 - 替换为模拟代码
            '''
# 不安全的用户输入示例 - input与eval结合
# 以下代码在实际运行时会等待用户输入，此处用模拟代码替代
# {var1} = input("请输入一个Python表达式: ")
{var1} = "1 + 1"  # 用固定表达式替代真实输入
{var2} = eval({var1})
print(f"表达式结果: {{({var2})}}")
''',
            # 输入与exec结合 - 替换为模拟代码
            '''
# 不安全的用户输入示例 - input与exec结合
# 以下代码在实际运行时会等待用户输入，此处用模拟代码替代
# {var1} = input("请输入Python代码: ")
{var1} = "print('模拟执行用户代码')"  # 用固定代码替代真实输入
exec({var1})
''',
            # 密码输入 - 替换为模拟代码
            '''
# 不安全的用户输入示例 - getpass
# 以下代码在实际运行时会等待用户输入密码，此处用模拟代码替代
# {var1} = getpass.getpass("请输入密码: ")
{var1} = "模拟密码"  # 用固定密码替代真实输入
if {var1} == "secret":
    print("密码正确")
else:
    print("密码错误")
'''
        ]
        
        # 随机选择一个模板
        template = random.choice(code_templates)
        
        # 准备模板变量
        var1 = self._random_var_name("input_")
        var2 = self._random_var_name("result_")
        
        # 填充模板
        code = template.format(
            var1=var1, 
            var2=var2
        )
        
        # 增加导入语句
        imports = self._add_imports(7)
        
        # 组合最终代码
        full_code = '\n'.join(imports) + '\n\n' + code
        
        # 添加混淆和注释
        full_code = self._add_obfuscation(full_code, random.randint(0, 1))
        full_code = self._add_comments(full_code)
        
        return full_code

    def generate_mixed_code(self) -> str:
        """生成混合调用类测试代码（综合以上7种类型）"""
        # 从每种类型的生成器中随机选择2-4个
        generators = [
            self.generate_unsafe_code_execution,
            self.generate_unsafe_deserialization,
            self.generate_unsafe_dynamic_import,
            self.generate_unsafe_file_operations,
            self.generate_unsafe_network_interactions,
            self.generate_unsafe_system_interactions,
            self.generate_unsafe_user_input
        ]
        
        selected_generators = random.sample(generators, random.randint(2, 4))
        code_blocks = []
        
        # 生成每个类型的代码块
        for generator in selected_generators:
            # 提取代码主体（移除导入语句，避免重复）
            code = generator()
            lines = code.split('\n')
            import_end = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('import '):
                    import_end = i
                    break
            
            # 添加主体代码
            code_block = '\n'.join(lines[import_end:])
            code_blocks.append(code_block)
        
        # 组合所有代码块
        combined_code = '\n\n'.join(code_blocks)
        
        # 增加混合类型的导入
        imports = self._add_imports(8)
        
        # 组合最终代码
        full_code = '\n'.join(imports) + '\n\n' + combined_code
        
        # 添加注释和文档字符串
        full_code = self._add_comments(full_code)
        full_code = f'"""\n混合类型的恶意代码测试示例\n包含多种不安全操作类型\n"""\n\n' + full_code
        
        return full_code

    def generate_safe_code(self) -> str:
        """生成安全的Python测试代码，不包含任何恶意操作"""
        code_templates = [
            # 计算器类模板
            '''
class Calculator:
    """一个简单的计算器类"""
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b == 0:
            raise ValueError("除数不能为零!")
        return a / b


def demo_calculator():
    """展示计算器功能的函数"""
    calc = Calculator()
    print("计算器演示:")
    print(f"5 + 3 = {{calc.add(5, 3)}}")
    print(f"5 - 3 = {{calc.subtract(5, 3)}}")
    print(f"5 * 3 = {{calc.multiply(5, 3)}}")
    try:
        print(f"5 / 0 = {{calc.divide(5, 0)}}")
    except ValueError as e:
        print(f"错误: {{e}}")
''',
            # 文本处理模板
            '''
def process_text(text):
    """处理文本并返回一些基本信息"""
    word_count = len(text.split())
    char_count = len(text)
    lines = text.split('\\n')
    line_count = len(lines)
    return {{
        "word_count": word_count,
        "char_count": char_count,
        "line_count": line_count
    }}


def demo_text_processing():
    """展示文本处理功能"""
    sample_text = "这是示例文本。\\n包含多行内容。\\n用于测试文本处理功能。"
    print("文本处理演示:")
    result = process_text(sample_text)
    print(f"文本：{{sample_text}}")
    print(f"单词数量：{{result['word_count']}}")
    print(f"字符数量：{{result['char_count']}}")
    print(f"行数：{{result['line_count']}}")
''',
            # 列表处理模板
            '''
def sort_items(items, reverse=False):
    """对列表进行排序"""
    return sorted(items, reverse=reverse)


def filter_items(items, min_val=0):
    """过滤列表中小于指定值的元素"""
    return [item for item in items if item >= min_val]


def demo_list_processing():
    """展示列表处理功能"""
    numbers = [{random_list}]
    print("列表处理演示:")
    print(f"原始列表: {{numbers}}")
    print(f"排序后: {{sort_items(numbers)}}")
    print(f"降序排列: {{sort_items(numbers, reverse=True)}}")
    min_threshold = {min_val}
    print(f"过滤小于{{min_threshold}}的元素: {{filter_items(numbers, min_threshold)}}")
''',
            # 字典处理模板
            '''
class DataProcessor:
    """数据处理类"""
    
    def __init__(self):
        self.data = {{}}
    
    def add_item(self, key, value):
        """添加数据项"""
        self.data[key] = value
    
    def get_item(self, key, default=None):
        """获取数据项"""
        return self.data.get(key, default)
    
    def remove_item(self, key):
        """删除数据项"""
        if key in self.data:
            del self.data[key]
            return True
        return False
    
    def list_items(self):
        """列出所有数据项"""
        return list(self.data.items())


def demo_data_processing():
    """展示字典处理功能"""
    processor = DataProcessor()
    processor.add_item("name", "{name}")
    processor.add_item("age", {age})
    processor.add_item("city", "{city}")
    
    print("字典处理演示:")
    print(f"所有数据: {{processor.list_items()}}")
    print(f"获取name: {{processor.get_item('name')}}")
    print(f"获取不存在的键: {{processor.get_item('email', '未设置')}}")
    
    removed = processor.remove_item("age")
    print(f"删除age: {{removed}}")
    print(f"删除后的数据: {{processor.list_items()}}")
'''
        ]
        
        # 随机选择一个模板
        template = random.choice(code_templates)
        
        # 准备变量
        demo_func = self._random_var_name("demo_")
        process_func = self._random_var_name("process_text_")
        sort_func = self._random_var_name("sort_items_")
        filter_func = self._random_var_name("filter_items_")
        data_class = "".join(word.capitalize() for word in self._random_var_name("data_").split("_"))
        
        # 生成随机数据
        random_list = ", ".join(str(random.randint(1, 100)) for _ in range(random.randint(5, 10)))
        min_val = random.randint(20, 50)
        names = ["张三", "李四", "王五", "赵六", "钱七", "孙八"]
        cities = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京"]
        name = random.choice(names)
        age = random.randint(18, 65)
        city = random.choice(cities)
        
        # 填充模板
        code = template.format(
            demo_func=demo_func,
            process_func=process_func,
            sort_func=sort_func,
            filter_func=filter_func,
            data_class=data_class,
            random_list=random_list,
            min_val=min_val,
            name=name,
            age=age,
            city=city
        )
        
        # 增加主函数模板
        main_code = f'''
if __name__ == "__main__":
    {demo_func}()
'''
        
        # 增加文档字符串
        docstring = f'''"""
这是一个安全的Python示例文件。
它展示了一些基本的Python编程概念和功能。
不包含任何恶意代码或不安全的操作。
"""

'''
        
        # 添加导入语句（仅添加安全的标准库）
        imports = ["import math", "import random", "import datetime", "import time", "import re"]
        selected_imports = random.sample(imports, random.randint(1, 3))
        
        # 组合最终代码
        full_code = docstring + "\n".join(selected_imports) + "\n\n" + code + "\n" + main_code
        
        # 适当添加一些注释
        commented_code = []
        for line in full_code.split("\n"):
            # 10%的概率添加注释
            if random.random() < 0.1 and line.strip() and not line.strip().startswith("#"):
                comments = [
                    "# 这是安全的代码示例",
                    "# 演示基本功能",
                    "# 处理数据",
                    "# 检查输入",
                    "# 返回结果"
                ]
                commented_code.append(random.choice(comments))
            commented_code.append(line)
        
        full_code = "\n".join(commented_code)
        
        return full_code

    def generate_file(self, code_type: int, file_path: str) -> None:
        """生成单个测试文件"""
        if code_type not in self.code_types:
            raise ValueError(f"不支持的代码类型: {code_type}")
        
        # 调用对应的生成器函数
        code = self.generators[code_type]()
        
        # 添加文件头注释（安全文件使用不同的头注释）
        if code_type == 9:  # 安全文件
            header = f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
安全Python代码示例: {os.path.basename(file_path)}
类型: {self.code_types[code_type]}
生成时间: {self._get_timestamp()}
描述: 此文件仅包含安全的Python代码，用于正常功能演示
\"\"\"

"""
        else:  # 恶意文件
            header = f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
恶意代码测试文件: {os.path.basename(file_path)}
类型: {self.code_types[code_type]}
生成时间: {self._get_timestamp()}
警告: 此文件仅用于安全测试，包含不安全的代码模式
\"\"\"

"""
        
        code_with_header = header + code
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code_with_header)
            
        print(f"已生成文件: {file_path}")

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def batch_generate(self, code_type: int, number: int, output_dir: str) -> None:
        """批量生成测试文件"""
        if code_type not in self.code_types:
            raise ValueError(f"不支持的代码类型: {code_type}")
            
        type_name = self.code_types[code_type]
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"开始生成 {number} 个类型为 '{type_name}' 的测试文件...")
        
        for i in range(1, number + 1):
            # 构建文件名
            if code_type == 9:  # 安全代码使用不同的命名格式
                file_name = f"{type_name}_{i:02d}.py"
            else:  # 恶意代码保持原有的命名格式
                file_name = f"risk_{type_name}_{i:02d}.py"
            
            file_path = os.path.join(output_dir, file_name)
            
            # 生成文件
            self.generate_file(code_type, file_path)
            
            # 更频繁地显示进度
            if i % 5 == 0 or i == number:
                print(f"进度: {i}/{number} ({i/number*100:.1f}%)")
            # 大批量时添加简单进度指示
            else:
                print(".", end="", flush=True)
        
        print(f"\n批量生成完成，共生成 {number} 个文件。")


def main():
    """主函数，处理命令行参数并执行生成操作"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="批量生成Python代码测试文件")
    
    # 添加参数
    parser.add_argument("type", type=int, choices=range(1, 10), 
                      help="代码类型: 1=代码执行, 2=反序列化, 3=动态导入, 4=文件操作, 5=网络交互, 6=系统交互, 7=用户输入, 8=混合调用, 9=安全代码")
    parser.add_argument("number", type=int, help="要生成的文件数量")
    parser.add_argument("output", type=str, help="输出目录路径")
    
    # 解析参数
    args = parser.parse_args()
    
    # 创建生成器实例
    generator = MaliciousCodeGenerator()
    
    # 执行批量生成
    generator.batch_generate(args.type, args.number, args.output)


if __name__ == "__main__":
    main()

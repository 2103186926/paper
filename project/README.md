
## semantic_feature_extractor.py
该项目：整合了Project03项目中 AST 到 JSON 与语义特征提取功能，直接将 Python 源代码文件作为输入，并输出 128 维语义特征向量。不再需要中间生成AST JSON文件，实现了Python代码 → 特征向量的直接转换。

输入：Python测试源码
输出：128 维语义特征向量 + 混淆检测结果的JSON文件

**改进1：支持处理整个文件夹的代码
允许用户通过 -f/--file 处理单个文件或通过 -d/--directory 处理整个目录

**改进2：使用tqdm库添加进度条功能

**改进3：将保存混淆检测结果到JSON文件设置成可选参数，默认是输出128维特征向量

如：python .\semantic_feature_extractor.py -d .\tools\mook_data\execute\  -o .\semantic_features\execute_features



## mook_data_extractor.py  批量生成python测试文件的工具类
第一个参数：1到9分别别对应
    1.不安全的代码执行类:eval(), exec(), compile()等函数调用
    2.不安全的反序列化类：pickle.load(), pickle.loads(), yaml.load()等函数调用
    3.不安全的动态导入类：__import__(), importlib.import_module()等函数调用
    4.不安全的文件操作类：open(), read(), write()等函数调用
    5.不安全的网络交互类：urllib.request.urlopen(), requests.get(), requests.post()等函数调用
    6.不安全的系统交互类：os.system, os.popen, subprocess.*等函数调用
    7.不安全的用户输入类：input()等函数调用
    8.混合调用类：以上7种恶意代码的混合调用
    9: "safe"            # 安全的Python代码
第二个参数：生成的测试文件数量
第三个参数：输出文件夹路径

如：python .\mook_data_extractor.py 8 100 .\mook_data\mix


## labels标签批量生成工具
参数:
        prefix: 文件名前缀
        label: 标签值（0或1）
        count: 生成的条目数量
        output_file: 输出文件名（可选，默认为<prefix>_labels.txt）

python .\labels_extractor.py risk_mixed 1 18 test



## bilstm_trainer.py 训练高危操作分类模型
用法：python bilstm_trainer.py <特征目录> <标签文件> [-m 模型文件] [-o 指标文件] [-e 轮数] [-b 批次大小]
参数：
  <特征目录>           特征向量目录路径
  <标签文件>           标签文件路径（CSV格式，包含filename和is_dangerous列）
  -m, --model          可选，模型保存路径，默认为model.h5
  -o, --output         可选，指标文件路径，默认为metrics.json
  -e, --epochs         可选，训练轮数，默认为50
  -b, --batch_size     可选，批次大小，默认为32
输出：
  训练好的BiLSTM模型（.h5文件）
  评估指标（.json文件）

示例：
  python bilstm_trainer.py features_dir labels.csv
  python bilstm_trainer.py features_dir labels.csv -m my_model.h5 -o my_metrics.json -e 100 -b 64

python .\bilstm_trainer.py ..\feature_fusion\output labels.csv




# bilstm_trainer2.py 
这个改进版本应该能够更好地处理过拟合问题，并提供更丰富的模型评估信息。建议您可以通过调整以下超参数来进一步优化模型：
lstm_units：LSTM单元数量
num_lstm_layers：LSTM层数
dense_units：全连接层的单元数配置
dropout_rate：Dropout比率
l2_lambda：L2正则化系数
learning_rate：学习率（可通过ReduceLROnPlateau自动调整）



# bilstm_trainer3.py 
将机器学习的Kslearn 换成了 Pytorch库，并将数据与模型都放到GPU上训练
python .\bilstm_trainer3.py ..\semantic_features\test\ .\labels.csv -e 50 -b 32 -l 0.001 -w 0.01

# bilstm_trainer4.py 
在3的基础上，做了一些调参优化
1、将代码参数部分加上了注释
2、对于恶意程序检测，召回率比精确率更重要（宁可误报，不可漏报），使用F2分数替代F1分数，更重视召回率


## tools/feature_intergration.py 特征向量集成脚本

输入：A特征向量和B特征向量
输出：融合后的特征向量（.npy文件）

1.多种集成方法：
  concat：简单拼接两个特征向量（默认方法）
  average：计算两个特征向量的平均值
  weighted：加权组合两个特征向量
2.特征归一化：支持Min-Max和Z-score两种归一化方法
3.方式：
  支持处理单对文件
  支持处理整个目录（自动匹配文件或全组合）
  提供进度条显示

TODO
# 单对文件集成
python feature_integration.py --files semantic_features.npy behavior_features.npy -o integrated_features.npy 
# 采用加权集成
python feature_integration.py --files semantic_features.npy behavior_features.npy -m weighted -w 0.7 0.3 
# 目录处理（自动匹配文件名）
python feature_integration.py --dirs semantic_features_dir/ behavior_features_dir/ -o integrated_features/
# 目录处理（不匹配文件名，处理所有组合）
python feature_integration.py --dirs semantic_features_dir/ behavior_features_dir/ --no-match

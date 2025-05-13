import random
import datetime
import json
import time
import os
import argparse  # 导入参数解析模块

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生成模拟的时空网格数据恶意操作日志')
    
    # 添加可选参数
    parser.add_argument('-o', '--output_dir', type=str, default="./logs",
                        help='日志文件输出目录路径 (默认: ./logs)')
    parser.add_argument('-n', '--num_logs', type=int, default=15,
                        help='要生成的日志文件数量 (默认: 15)')
    parser.add_argument('-s', '--start_index', type=int, default=1,
                        help='日志文件起始索引 (默认: 1)')
    parser.add_argument('-p', '--prefix', type=str, default="log",
                        help='日志文件前缀 (默认: log)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='显示详细输出信息')
    
    return parser.parse_args()

# 确保logs目录存在
os.makedirs("./logs", exist_ok=True)

# 定义一些常量和帮助函数
def generate_bbox(region="全中国"):
    """
    根据region参数生成对应的经纬度范围
    
    参数:
        region (str): 地区名称
        
    返回:
        list: 包含[最小经度, 最小纬度, 最大经度, 最大纬度]的列表
    """
    if region == "全中国":
        return [73.5, 18.0, 135.0, 54.0]
    elif region == "中国东部":
        return [105.0, 18.0, 135.0, 54.0]
    elif region == "中国中部":
        return [90.0, 18.0, 105.0, 54.0]
    elif region == "中国西部":
        return [73.5, 18.0, 90.0, 54.0]
    elif region == "北京":
        return [115.7, 39.4, 117.4, 41.6]
    elif region == "上海":
        return [120.85, 30.67, 122.05, 31.87]
    else:
        # 生成一个0.1°×0.1°的随机小方块
        base_lon = random.uniform(73.5, 134.9)
        base_lat = random.uniform(18.0, 53.9)
        return [base_lon, base_lat, base_lon + 0.1, base_lat + 0.1]

def format_log_entry(timestamp, uid, api, params, ip, device_id, response_size, compute_time, session_id, sequence_num, status="success"):
    """
    格式化日志条目为JSON格式
    
    参数:
        timestamp (str): 时间戳
        uid (str): 用户ID
        api (str): API名称
        params (dict): API参数
        ip (str): IP地址
        device_id (str): 设备ID
        response_size (int): 响应大小(字节)
        compute_time (float): 计算时间(秒)
        session_id (int/str): 会话ID
        sequence_num (int): 序列号
        status (str): 请求状态
        
    返回:
        str: JSON格式的日志条目
    """
    log_entry = {
        "timestamp": timestamp, # 时间戳
        "session_id": session_id, # 会话ID
        "sequence_num": sequence_num,  # 操作序号
        "action": "API_CALL", # 操作类型
        "uid": uid, # 用户ID
        "ip": ip, # IP地址
        "device_id": device_id, # 设备ID
        "api": api, # API名称
        "params": params, # API参数
        "response_size": response_size, # 响应大小(字节)
        "compute_time": compute_time, # 计算时间(秒)
        "status": status # 请求状态
    }
    return json.dumps(log_entry, ensure_ascii=False)

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.verbose:
        print(f"将在 {args.output_dir} 目录下生成 {args.num_logs} 个日志文件")
    
    # 生成指定数量的日志文件
    for log_index in range(args.start_index, args.start_index + args.num_logs):
        # 确定日志文件名
        filename = f"{args.output_dir}/{args.prefix}{log_index:02d}.log"
        
        with open(filename, "w", encoding="utf-8") as f:
            # 根据不同的恶意类型生成日志
            if log_index % 15 == 1:
                # 案例1：高密度数据爬取（短时爆频查询）
                uid = f"user_{log_index}_{random.randint(1000, 9999)}"
                base_time = datetime.datetime(2025, 3, 28, 14, 30, 0)
                
                # 1小时内发起100次相同或高度重叠的请求
                for i in range(100):
                    current_time = base_time + datetime.timedelta(minutes=i*0.6)  # 每36秒一次请求
                    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 随机IP和设备ID
                    ip = f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"
                    device_id = f"DEVICE{random.randint(100000, 999999)}"
                    
                    # 构建请求参数
                    params = {
                        "spatial_range": generate_bbox("全中国"),
                        "temporal_range": ["2025-04-01 00:00:00", "2025-04-01 05:00:00"],
                        "spatial_resolution": 0.01,
                        "temporal_resolution": 15,
                        "data_category": "temperature"
                    }
                    
                    # 大数据量、高计算时间
                    response_size = random.randint(450000000, 480000000)  # 近4.8亿数据点
                    compute_time = random.randint(30, 60)  # 30-60秒
                    
                    # 写入日志
                    log_entry = format_log_entry(
                        timestamp, uid, "query_spatial_grid_data", params,
                        ip, device_id, response_size, compute_time, f"session_{log_index}", i
                    )
                    f.write(f"{log_entry}\n")
                
            elif log_index % 15 == 2:
                # 案例2：异常的未来时间数据挖掘
                uid = f"user_{log_index}_{random.randint(1000, 9999)}"
                base_time = datetime.datetime(2025, 3, 28, 10, 0, 0)
                
                # 在同一天内查询未来9个月的数据
                for i in range(250):
                    current_time = base_time + datetime.timedelta(minutes=i*5)  # 每5分钟一次请求
                    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 随机IP和设备ID，但保持相似性
                    ip = f"172.16.{random.randint(1, 10)}.{random.randint(1, 254)}"
                    device_id = f"DEVICE{log_index}_{random.randint(100000, 999999)}"
                    
                    # 构建请求参数 - 查询未来日期
                    future_date = datetime.date(2025, 4, 1) + datetime.timedelta(days=i % 270)  # 未来9个月循环
                    params = {
                        "spatial_range": generate_bbox("全中国"),
                        "temporal_range": [
                            f"{future_date.strftime('%Y-%m-%d')} 00:00:00", 
                            f"{future_date.strftime('%Y-%m-%d')} 05:00:00"
                        ],
                        "spatial_resolution": 0.01,
                        "temporal_resolution": 15,
                        "data_category": "temperature"
                    }
                    
                    # 大数据量、高计算时间
                    response_size = random.randint(450000000, 480000000)  # 近4.8亿数据点
                    compute_time = random.randint(40, 60)  # 40-60秒
                    
                    # 写入日志
                    log_entry = format_log_entry(
                        timestamp, uid, "query_future_grid_data", params,
                        ip, device_id, response_size, compute_time, f"session_{log_index}", i
                    )
                    f.write(f"{log_entry}\n")
                
            elif log_index % 15 == 3:
                # 案例3：扫描式网格细化攻击（试探系统负载极限）
                uid = f"user_{log_index}_{random.randint(1000, 9999)}"
                base_time = datetime.datetime(2025, 4, 1, 8, 0, 0)
                
                # 逐步提高精度的查询
                resolutions = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
                
                for i, resolution in enumerate(resolutions):
                    current_time = base_time + datetime.timedelta(minutes=i*10)  # 每10分钟一次请求
                    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 随机IP和设备ID
                    ip = f"10.0.{random.randint(1, 254)}.{random.randint(1, 254)}"
                    device_id = f"DEVICE{random.randint(100000, 999999)}"
                    
                    # 构建请求参数
                    params = {
                        "spatial_range": generate_bbox("全中国"),
                        "temporal_range": ["2025-04-01 00:00:00", "2025-04-01 05:00:00"],
                        "spatial_resolution": resolution,
                        "temporal_resolution": 15,
                        "data_category": "temperature"
                    }
                    
                    # 计算时间随精度增加而增加
                    compute_factor = min(100, 1/resolution * 10)  # 精度越高，计算时间越长
                    response_size = int(4680000 * (0.1/resolution)**2)  # 数据量随精度平方增长
                    compute_time = int(3 * (0.1/resolution)**2)  # 计算时间随精度平方增长
                    
                    # 当精度太高时超时
                    status = "success" if resolution >= 0.005 else "timeout"
                    
                    # 写入日志
                    log_entry = format_log_entry(
                        timestamp, uid, "query_spatial_grid_data", params,
                        ip, device_id, response_size, compute_time, f"session_{log_index}", i, status
                    )
                    f.write(f"{log_entry}\n")
                    
                    # 对于超时的请求，模拟用户重试行为
                    if status == "timeout":
                        for retry in range(3):
                            retry_time = current_time + datetime.timedelta(seconds=30*(retry+1))
                            retry_timestamp = retry_time.strftime("%Y-%m-%d %H:%M:%S")
                            
                            retry_log = format_log_entry(
                                retry_timestamp, uid, "query_spatial_grid_data", params,
                                ip, device_id, 0, 0, f"session_{log_index}", i, "timeout"
                            )
                            f.write(f"{retry_log}\n")
                
            elif log_index % 15 == 4:
                # 案例4：伪装成"分布式"查询（多设备/多账号共谋）
                # 主账号
                main_uid = f"user_{log_index}_{random.randint(1000, 9999)}"
                # 5个傀儡账号
                puppet_uids = [f"user_{log_index}_{random.randint(1000, 9999)}" for _ in range(5)]
                
                base_time = datetime.datetime(2025, 3, 28, 14, 0, 0)
                timestamp = base_time.strftime("%Y-%m-%d %H:%M:%S")
                
                # 划分区域
                regions = ["中国东部", "中国中部", "中国西部"]
                time_periods = [
                    ["2025-04-01 00:00:00", "2025-04-01 02:00:00"],
                    ["2025-04-01 02:00:00", "2025-04-01 05:00:00"]
                ]
                
                # 主账号查询
                main_ip = "172.16.100.100"
                main_device = "A1B2C3"
                main_params = {
                    "spatial_range": generate_bbox("中国东部"),
                    "temporal_range": time_periods[0],
                    "spatial_resolution": 0.01,
                    "temporal_resolution": 15,
                    "data_category": "temperature"
                }
                
                main_log = format_log_entry(
                    timestamp, main_uid, "query_spatial_grid_data", main_params,
                    main_ip, main_device, 180000000, 35, f"session_{log_index}", 0
                )
                f.write(f"{main_log}\n")
                
                # 傀儡账号在毫秒级时间差内同时查询
                for i, puppet_uid in enumerate(puppet_uids):
                    puppet_time = base_time + datetime.timedelta(milliseconds=random.randint(50, 500))
                    puppet_timestamp = puppet_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    
                    puppet_ip = f"172.16.100.{100+i+1}"  # 相似的IP地址
                    
                    if i == 0:  # 第一个傀儡账号使用类似的设备ID
                        puppet_device = "A1B2C4"  # 仅1位不同
                        region = "中国东部"
                        period = time_periods[1]
                    else:
                        puppet_device = f"DEVICE{random.randint(100000, 999999)}"
                        region = regions[i % len(regions)]
                        period = time_periods[i % len(time_periods)]
                    
                    puppet_params = {
                        "spatial_range": generate_bbox(region),
                        "temporal_range": period,
                        "spatial_resolution": 0.01,
                        "temporal_resolution": 15,
                        "data_category": "temperature"
                    }
                    
                    puppet_log = format_log_entry(
                        puppet_timestamp, puppet_uid, "query_spatial_grid_data", puppet_params,
                        puppet_ip, puppet_device, 180000000, random.randint(30, 40), f"session_{log_index}", i
                    )
                    f.write(f"{puppet_log}\n")
                
            elif log_index % 15 == 5:
                # 案例5：低速隐蔽式数据拼接（长时间维度爬取）
                uid = f"user_{log_index}_{random.randint(1000, 9999)}"
                
                # 30天内，每天查询1小时的数据
                for day in range(30):
                    # 凌晨3点查询，避开高峰期
                    current_time = datetime.datetime(2025, 4, 1, 3, 0, 0) + datetime.timedelta(days=day)
                    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 随机IP和设备ID，但保持一定的相似性
                    ip = f"45.67.{random.randint(1, 10)}.{random.randint(1, 254)}"
                    device_id = f"MOBILE{random.randint(1000, 9999)}"
                    
                    # 构建请求参数 - 每天查询不同的1小时段
                    hour_to_query = day % 24  # 循环查询一天24小时
                    params = {
                        "spatial_range": generate_bbox("全中国"),
                        "temporal_range": [
                            f"2025-04-01 {hour_to_query:02d}:00:00", 
                            f"2025-04-01 {hour_to_query:02d}:59:59"
                        ],
                        "spatial_resolution": 0.01,
                        "temporal_resolution": 15,
                        "data_category": "temperature"
                    }
                    
                    # 每次查询的数据量都小于单日限额
                    response_size = random.randint(35000000, 36000000)  # 约3600万数据点
                    compute_time = random.randint(10, 20)  # 10-20秒
                    
                    # 写入日志
                    log_entry = format_log_entry(
                        timestamp, uid, "query_spatial_grid_data", params,
                        ip, device_id, response_size, compute_time, f"session_{log_index}", day
                    )
                    f.write(f"{log_entry}\n")
                
            elif log_index % 15 == 6:
                # 案例6：假装"科研用途"的学术欺诈
                uid = f"user_{log_index}_{random.randint(1000, 9999)}"
                base_time = datetime.datetime(2025, 3, 25, 10, 0, 0)
                timestamp = base_time.strftime("%Y-%m-%d %H:%M:%S")
                
                # 先进行身份认证
                auth_params = {
                    "identity": "professor",
                    "institution": "某大学气象系",
                    "research_purpose": "中国地区气象模式优化研究",
                    "email": "professor@some-university.edu.cn"
                }
                
                auth_log = format_log_entry(
                    timestamp, uid, "verify_academic_identity", auth_params,
                    "202.38.64.123", "LAPTOP2023", 2048, 1.5, f"session_{log_index}", 0
                )
                f.write(f"{auth_log}\n")
                
                # 等待验证通过
                approval_time = base_time + datetime.timedelta(hours=2)
                approval_timestamp = approval_time.strftime("%Y-%m-%d %H:%M:%S")
                
                approval_log = format_log_entry(
                    approval_timestamp, "SYSTEM", "academic_identity_approved", 
                    {"uid": uid, "status": "approved"}, 
                    "internal", "system", 1024, 0.3, f"session_{log_index}", 1
                )
                f.write(f"{approval_log}\n")
                
                # 下载高精度气象数据
                download_time = base_time + datetime.timedelta(hours=3)
                download_timestamp = download_time.strftime("%Y-%m-%d %H:%M:%S")
                
                download_params = {
                    "spatial_range": generate_bbox("全中国"),
                    "temporal_range": ["2025-04-01 00:00:00", "2025-04-01 23:59:59"],
                    "spatial_resolution": 0.01,
                    "temporal_resolution": 15,
                    "data_category": "temperature",
                    "academic_token": "acad_x7c91j2b3k4l"
                }
                
                download_log = format_log_entry(
                    download_timestamp, uid, "download_academic_data", download_params,
                    "202.38.64.123", "LAPTOP2023", 864000000, 120, f"session_{log_index}", 2
                )
                f.write(f"{download_log}\n")
                
                # 没有进行任何数据分析操作
                # 3天后，数据在暗网出现（不在日志中，但是这是案例的背景）
                
            elif log_index % 15 == 7:
                # 全国拼图扫描（总结中第2种恶意行为模式）
                uid = f"user_{log_index}_{random.randint(1000, 9999)}"
                
                # 连续7天，每天查询不同的地理区域
                regions = [
                    "华南", "华东", "华北", "华中", "西南", "西北", "东北"
                ]
                
                # 为每个区域定义经纬度范围
                region_bbox = {
                    "华南": [105.0, 20.0, 120.0, 26.0],
                    "华东": [115.0, 26.0, 123.0, 35.0],
                    "华北": [110.0, 35.0, 120.0, 42.0],
                    "华中": [108.0, 26.0, 116.0, 34.0],
                    "西南": [97.0, 21.0, 108.0, 33.0],
                    "西北": [73.5, 30.0, 110.0, 42.0],
                    "东北": [118.0, 40.0, 135.0, 54.0]
                }
                
                for day, region in enumerate(regions):
                    # 每天10点查询
                    current_time = datetime.datetime(2025, 4, 1, 10, 0, 0) + datetime.timedelta(days=day)
                    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 随机但相似的IP和设备ID
                    ip = f"61.135.{random.randint(1, 10)}.{random.randint(1, 254)}"
                    device_id = f"WIN10-PC-{random.randint(1000, 9999)}"
                    
                    # 构建请求参数
                    params = {
                        "spatial_range": region_bbox[region],
                        "temporal_range": ["2025-04-01 00:00:00", "2025-04-01 01:00:00"],  # 只取一小时
                        "spatial_resolution": 0.01,
                        "temporal_resolution": 15,
                        "data_category": "temperature"
                    }
                    
                    # 响应大小和计算时间
                    area_size_factor = (
                        (region_bbox[region][2] - region_bbox[region][0]) * 
                        (region_bbox[region][3] - region_bbox[region][1])
                    ) / 100  # 根据区域大小调整数据量
                    
                    response_size = int(100000000 * area_size_factor)  # 根据区域大小调整
                    compute_time = int(15 * area_size_factor)  # 根据区域大小调整
                    
                    # 写入日志
                    log_entry = format_log_entry(
                        timestamp, uid, "query_spatial_grid_data", params,
                        ip, device_id, response_size, compute_time, f"session_{log_index}", day
                    )
                    f.write(f"{log_entry}\n")
                
            elif log_index % 15 == 8:
                # 定向高频采样（总结中第3种恶意行为模式）
                uid = f"user_{log_index}_{random.randint(1000, 9999)}"
                base_time = datetime.datetime(2025, 4, 1, 9, 0, 0)
                
                # 每15分钟查一次，固定spatial_res=0.01°，但time_range每次只移动15分钟
                for i in range(96):  # 24小时 * 4 (15分钟一次) = 96次请求
                    current_time = base_time + datetime.timedelta(minutes=i*15)
                    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 查询的时间范围每次只移动15分钟
                    start_time = datetime.datetime(2025, 4, 1, 0, 0, 0) + datetime.timedelta(minutes=i*15)
                    end_time = start_time + datetime.timedelta(minutes=15)
                    
                    # 随机IP和设备ID，但保持一定的相似性
                    ip = f"58.217.{random.randint(1, 10)}.{random.randint(1, 254)}"
                    device_id = f"MOBILE{random.randint(1000, 9999)}"
                    
                    # 构建请求参数
                    params = {
                        "spatial_range": generate_bbox("全中国"),
                        "temporal_range": [
                            start_time.strftime("%Y-%m-%d %H:%M:%S"),
                            end_time.strftime("%Y-%m-%d %H:%M:%S")
                        ],
                        "spatial_resolution": 0.01,
                        "temporal_resolution": 15,
                        "data_category": "temperature"
                    }
                    
                    # 每次查询的数据量
                    response_size = random.randint(22000000, 23000000)  # 约2200万数据点/15分钟
                    compute_time = random.randint(8, 12)
                    
                    # 写入日志
                    log_entry = format_log_entry(
                        timestamp, uid, "query_spatial_grid_data", params,
                        ip, device_id, response_size, compute_time, f"session_{log_index}", i
                    )
                    f.write(f"{log_entry}\n")
                
            elif log_index % 15 == 9:
                # 夜间批量导出（总结中第6种恶意行为模式）
                uid = f"user_{log_index}_{random.randint(1000, 9999)}"
                base_time = datetime.datetime(2025, 4, 1, 2, 0, 0)  # 凌晨2点开始
                
                # 在凌晨2:00~4:00（服务器低峰期）持续调用export_grid_data
                for i in range(50):
                    current_time = base_time + datetime.timedelta(minutes=i*2.4)  # 平均2.4分钟一次请求
                    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 随机IP，但不是浏览器
                    ip = f"122.114.{random.randint(1, 254)}.{random.randint(1, 254)}"
                    device_id = "curl/7.76.1"  # 使用curl工具，非浏览器
                    
                    # 构建请求参数
                    params = {
                        "spatial_range": generate_bbox("全中国"),
                        "temporal_range": ["2025-04-01 00:00:00", "2025-04-01 23:59:59"],  # 全天数据
                        "spatial_resolution": 0.1,  # 使用较低精度以避免数据量过大
                        "temporal_resolution": 60,  # 1小时精度
                        "data_category": "temperature",
                        "format": "csv",
                        "compression": "gzip"
                    }
                    
                    # 响应大小和计算时间
                    response_size = random.randint(400000000, 450000000)  # 导出大量数据
                    compute_time = random.randint(30, 60)  # 耗时较长
                    
                    # 写入日志，使用export_grid_data API
                    log_entry = format_log_entry(
                        timestamp, uid, "export_grid_data", params,
                        ip, device_id, response_size, compute_time, f"session_{log_index}", i
                    )
                    f.write(f"{log_entry}\n")
                
            elif log_index % 15 == 10:
                # 案例11: 精度波动攻击（频率和精度角度）
                uid = f"user_{log_index}_{random.randint(1000, 9999)}"
                base_time = datetime.datetime(2025, 4, 2, 9, 0, 0)
                
                # 使用变化的精度模式
                precision_patterns = [
                    [0.1, 0.1, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01, 0.005, 0.005],  # 递增精度
                    [0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.005, 0.005, 0.01, 0.01]  # 波动精度
                ]
                
                # 执行30次查询，使用波动的精度
                for i in range(30):
                    pattern_index = i % 2
                    precision_index = (i // 2) % len(precision_patterns[pattern_index])
                    resolution = precision_patterns[pattern_index][precision_index]
                    
                    current_time = base_time + datetime.timedelta(minutes=i*8)  # 每8分钟一次请求
                    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 随机但相似的IP和设备ID
                    ip = f"103.54.{random.randint(1, 10)}.{random.randint(1, 254)}"
                    device_id = f"LAPTOP-{random.choice(['A','B','C','D'])}{random.randint(1000, 9999)}"
                    
                    # 构建请求参数 - 逐渐扩大范围
                    area_expand = min(i/10, 1.0)  # 随着请求次数扩大范围，最大扩大1倍
                    base_bbox = generate_bbox("北京")
                    range_width = base_bbox[2] - base_bbox[0]
                    range_height = base_bbox[3] - base_bbox[1]
                    
                    expanded_bbox = [
                        base_bbox[0] - range_width * area_expand * 0.1,
                        base_bbox[1] - range_height * area_expand * 0.1,
                        base_bbox[2] + range_width * area_expand * 0.1,
                        base_bbox[3] + range_height * area_expand * 0.1
                    ]
                    
                    params = {
                        "spatial_range": expanded_bbox,
                        "temporal_range": ["2025-04-01 00:00:00", "2025-04-01 23:59:59"],
                        "spatial_resolution": resolution,
                        "temporal_resolution": 15,
                        "data_category": "temperature"
                    }
                    
                    # 响应大小和计算时间与精度相关
                    response_size = int(3000000 * (0.01/resolution)**2 * (1 + area_expand))
                    compute_time = int(5 * (0.01/resolution)**2 * (1 + area_expand))
                    
                    # 写入日志
                    log_entry = format_log_entry(
                        timestamp, uid, "query_spatial_grid_data", params,
                        ip, device_id, response_size, compute_time, f"session_{log_index}", i
                    )
                    f.write(f"{log_entry}\n")

            elif log_index % 15 == 11:
                # 案例12: 跨区域对比分析（范围角度）
                uid = f"user_{log_index}_{random.randint(1000, 9999)}"
                base_time = datetime.datetime(2025, 4, 3, 14, 0, 0)
                
                # 定义敏感区域（军事设施周边）的小块区域坐标
                sensitive_regions = [
                    [116.28, 39.98, 116.35, 40.05],  # 北京某敏感区域1
                    [116.36, 39.97, 116.43, 40.04],  # 北京某敏感区域2
                    [116.29, 39.90, 116.36, 39.97],  # 北京某敏感区域3
                    [116.37, 39.89, 116.44, 39.96],  # 北京某敏感区域4
                    [116.45, 39.87, 116.52, 39.94],  # 北京某敏感区域5
                    [116.37, 39.80, 116.44, 39.87],  # 北京某敏感区域6
                ]
                
                # 查询40次，每次查询一个小区域，看起来像是在做区域对比
                for i in range(40):
                    current_time = base_time + datetime.timedelta(minutes=i*7)  # 每7分钟一次请求
                    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 随机但看似相关的IP和设备ID
                    ip = f"220.181.{random.randint(60, 70)}.{random.randint(1, 254)}"
                    device_id = f"ResearchStation-{random.randint(100, 999)}"
                    
                    # 选择区域 - 每6次循环完整覆盖一次所有敏感区域
                    region_index = i % len(sensitive_regions)
                    current_region = sensitive_regions[region_index]
                    
                    # 构建请求参数
                    params = {
                        "spatial_range": current_region,
                        "temporal_range": ["2025-04-01 12:00:00", "2025-04-01 18:00:00"],
                        "spatial_resolution": 0.001,  # 非常高的精度
                        "temporal_resolution": 5,     # 5分钟间隔
                        "data_category": "temperature",
                        "analysis_type": "region_comparison"  # 伪装成区域对比分析
                    }
                    
                    # 响应大小和计算时间
                    # 区域小但精度高，导致数据量适中
                    response_size = random.randint(5000000, 8000000)
                    compute_time = random.randint(8, 15)
                    
                    # 写入日志
                    log_entry = format_log_entry(
                        timestamp, uid, "query_spatial_grid_data", params,
                        ip, device_id, response_size, compute_time, f"session_{log_index}", i
                    )
                    f.write(f"{log_entry}\n")

            elif log_index % 15 == 12:
                # 案例13: 历史全覆盖（时间段角度）
                uid = f"user_{log_index}_{random.randint(1000, 9999)}"
                base_time = datetime.datetime(2025, 4, 5, 1, 0, 0)  # 凌晨1点开始
                
                # 设定目标区域
                target_region = [121.45, 31.20, 121.55, 31.30]  # 上海某区域
                
                # 查询过去5年的历史数据，每个月一次查询
                years = range(2020, 2026)
                months = range(1, 13)
                
                sequence_num = 0
                for year in years:
                    for month in months:
                        # 如果是2025年且月份超过4月，就不再查询
                        if year == 2025 and month > 4:
                            continue
                            
                        current_time = base_time + datetime.timedelta(minutes=sequence_num*30)  # 每30分钟一次请求
                        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                        
                        # 构建该月的日期范围
                        days_in_month = 31  # 默认值
                        if month in [4, 6, 9, 11]:
                            days_in_month = 30
                        elif month == 2:
                            # 简单的闰年检查
                            if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                                days_in_month = 29
                            else:
                                days_in_month = 28
                        
                        start_date = f"{year}-{month:02d}-01 00:00:00"
                        end_date = f"{year}-{month:02d}-{days_in_month} 23:59:59"
                        
                        # 使用一致的IP地址但不同设备ID
                        ip = "180.167.168.42"  # 固定IP
                        device_id = f"HistoricalAnalyzer-{random.randint(1000, 9999)}"
                        
                        # 构建请求参数
                        params = {
                            "spatial_range": target_region,
                            "temporal_range": [start_date, end_date],
                            "spatial_resolution": 0.02,  # 适中的空间精度
                            "temporal_resolution": 60,   # 1小时时间精度
                            "data_category": "temperature",
                            "data_source": "historical_archive"
                        }
                        
                        # 响应大小和计算时间
                        response_size = random.randint(15000000, 25000000)  # 月数据量
                        compute_time = random.randint(10, 20)
                        
                        # 写入日志
                        log_entry = format_log_entry(
                            timestamp, uid, "query_historical_grid_data", params,
                            ip, device_id, response_size, compute_time, f"session_{log_index}", sequence_num
                        )
                        f.write(f"{log_entry}\n")
                        
                        sequence_num += 1

            elif log_index % 15 == 13:
                # 案例14: 差异化精度攻击（精度角度）
                uid = f"user_{log_index}_{random.randint(1000, 9999)}"
                base_time = datetime.datetime(2025, 4, 8, 16, 0, 0)
                
                # 定义敏感区域和非敏感区域
                sensitive_regions = [
                    ["核心区域", [116.30, 39.90, 116.45, 40.05], 0.001],  # 北京核心区域，超高精度
                    ["军事区域", [117.10, 39.10, 117.25, 39.25], 0.002],  # 某军事区域，高精度
                    ["港口区域", [121.45, 31.20, 121.60, 31.35], 0.005]   # 上海港区，中高精度
                ]
                
                non_sensitive_regions = [
                    ["农业区域1", [115.00, 38.50, 116.00, 39.50], 0.1],   # 农业区域，低精度
                    ["山区", [110.00, 35.00, 112.00, 37.00], 0.2],        # 山区，极低精度
                    ["沙漠区域", [100.00, 40.00, 105.00, 42.00], 0.5]     # 沙漠区域，超低精度
                ]
                
                # 合并所有区域到一个列表
                all_regions = sensitive_regions + non_sensitive_regions
                
                # 看似合理的查询模式，但对敏感区域使用高精度，非敏感区域使用低精度
                for i, (region_name, bbox, resolution) in enumerate(all_regions):
                    # 对每个区域进行3次查询，使用不同的时间段
                    for j in range(3):
                        current_time = base_time + datetime.timedelta(hours=i, minutes=j*20)
                        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                        
                        # 查询的时间段 - 对敏感区域使用高时间精度
                        if i < len(sensitive_regions):  # 敏感区域
                            temp_resolution = 5  # 5分钟
                            time_window = 3  # 3小时
                        else:  # 非敏感区域
                            temp_resolution = 60  # 1小时
                            time_window = 12  # 12小时
                        
                        start_hour = (8 * j) % 24
                        end_hour = (start_hour + time_window) % 24
                        
                        # 确保开始时间小于结束时间
                        if end_hour < start_hour:
                            temp_start = f"2025-04-08 {start_hour:02d}:00:00"
                            temp_end = f"2025-04-09 {end_hour:02d}:00:00"
                        else:
                            temp_start = f"2025-04-08 {start_hour:02d}:00:00"
                            temp_end = f"2025-04-08 {end_hour:02d}:00:00"
                        
                        # 使用多个不同的IP和设备，但有规律性
                        ip = f"59.{108+i}.{200+j}.{random.randint(1, 254)}"
                        device_id = f"RegionalScanner-{i+1}-{j+1}"
                        
                        # 构建请求参数
                        params = {
                            "spatial_range": bbox,
                            "temporal_range": [temp_start, temp_end],
                            "spatial_resolution": resolution,
                            "temporal_resolution": temp_resolution,
                            "data_category": "temperature",
                            "region_label": region_name
                        }
                        
                        # 响应大小和计算时间 - 敏感区域数据量大且计算时间长
                        area_size = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
                        base_size = int(1000000 * area_size / resolution)
                        response_size = min(base_size, 500000000)  # 限制最大值
                        compute_time = min(int(base_size / 10000000), 60)  # 限制最大值
                        
                        sequence_num = i * 3 + j
                        
                        # 写入日志
                        log_entry = format_log_entry(
                            timestamp, uid, "query_differential_precision", params,
                            ip, device_id, response_size, compute_time, f"session_{log_index}", sequence_num
                        )
                        f.write(f"{log_entry}\n")

            elif log_index % 15 == 14:
                # 案例15: 节假日隐蔽爬取（频率和时间段角度）
                uid = f"user_{log_index}_{random.randint(1000, 9999)}"
                
                # 选择国庆假期作为攻击时间
                holiday_start = datetime.datetime(2025, 10, 1, 0, 0, 0)
                
                # 定义不同的数据类别和参数组合
                data_categories = ["temperature", "humidity", "wind_speed", "pressure", "precipitation"]
                param_combinations = []
                
                # 创建30个不同的参数组合
                for _ in range(30):
                    # 在全国范围内随机选择一个省级区域
                    base_lon = random.uniform(75.0, 130.0)
                    base_lat = random.uniform(20.0, 50.0)
                    region_width = random.uniform(1.0, 5.0)
                    region_height = random.uniform(1.0, 5.0)
                    bbox = [base_lon, base_lat, base_lon + region_width, base_lat + region_height]
                    
                    # 随机选择一个数据类别
                    category = random.choice(data_categories)
                    
                    # 随机选择时间范围 - 限制在假期内
                    day_offset = random.randint(0, 6)  # 0-6天，覆盖整个国庆假期
                    hour_range = random.randint(2, 8)  # 每次查询2-8小时
                    start_hour = random.randint(0, 23-hour_range)
                    
                    start_date = holiday_start + datetime.timedelta(days=day_offset, hours=start_hour)
                    end_date = start_date + datetime.timedelta(hours=hour_range)
                    
                    # 随机选择分辨率
                    spatial_res = random.choice([0.05, 0.02, 0.01])
                    temporal_res = random.choice([15, 30, 60])
                    
                    param_combinations.append({
                        "bbox": bbox,
                        "category": category,
                        "start_date": start_date,
                        "end_date": end_date,
                        "spatial_res": spatial_res,
                        "temporal_res": temporal_res
                    })
                
                # 在假期内分散请求，每天特别集中在凌晨0-6点
                for i, params in enumerate(param_combinations):
                    # 确定请求时间 - 集中在凌晨，但有一些随机性
                    if i % 5 == 0:  # 20%的请求发生在白天
                        hour = random.randint(9, 17)
                    else:  # 80%的请求发生在凌晨
                        hour = random.randint(0, 6)
                    
                    # 计算当前请求时间
                    day_offset = i // 5  # 每天最多5个请求
                    minute = random.randint(0, 59)
                    current_time = holiday_start + datetime.timedelta(days=day_offset, hours=hour, minutes=minute)
                    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 使用多个不同的IP和设备ID，但来自同一运营商
                    ip = f"121.{random.randint(1, 10)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
                    device_id = f"MobileDevice-{random.choice(['iOS','Android'])}-{random.randint(10000, 99999)}"
                    
                    # 构建请求参数
                    request_params = {
                        "spatial_range": params["bbox"],
                        "temporal_range": [
                            params["start_date"].strftime("%Y-%m-%d %H:%M:%S"),
                            params["end_date"].strftime("%Y-%m-%d %H:%M:%S")
                        ],
                        "spatial_resolution": params["spatial_res"],
                        "temporal_resolution": params["temporal_res"],
                        "data_category": params["category"],
                        "holiday_type": "national_day"  # 添加这个参数伪装成假期旅游数据分析
                    }
                    
                    # 响应大小和计算时间
                    # 计算区域大小
                    area_size = (params["bbox"][2] - params["bbox"][0]) * (params["bbox"][3] - params["bbox"][1])
                    time_hours = (params["end_date"] - params["start_date"]).total_seconds() / 3600
                    
                    # 根据区域大小、时间范围和精度计算数据量
                    data_points = int(area_size * time_hours * (1 / params["spatial_res"]) * (60 / params["temporal_res"]))
                    response_size = data_points * 20  # 假设每个数据点20字节
                    compute_time = int(data_points / 500000) + random.randint(1, 5)  # 计算时间
                    
                    # 限制最大值
                    response_size = min(response_size, 400000000)
                    compute_time = min(compute_time, 50)
                    
                    # 写入日志
                    log_entry = format_log_entry(
                        timestamp, uid, "query_holiday_data", request_params,
                        ip, device_id, response_size, compute_time, f"session_{log_index}", i
                    )
                    f.write(f"{log_entry}\n")

        if args.verbose:
            print(f"已生成日志文件: {filename}")
        else:
            print(f"{filename}")

if __name__ == "__main__":
    main()

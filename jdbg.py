#!/usr/bin/env python3
"""
统计 legacy.json 中 user="bird" 的 jobs 信息：
1. 统计各 status 的个数
2. 生成 start_time24h 的统计柱状图（每2小时一个区间）
3. 找到最晚跑的 job（中国时间，6:00为一天截止）
"""

import json
from collections import Counter
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
rcParams['axes.unicode_minus'] = False

def load_legacy_json(file_path='legacy.json'):
    """加载 legacy.json 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def filter_bird_jobs(data):
    """过滤出 user="bird" 的所有 jobs"""
    return [job for job in data if job.get('user') == 'ke']

def count_statuses(jobs):
    """统计各 status 的个数"""
    statuses = [job.get('status', 'unknown') for job in jobs]
    return Counter(statuses)

def parse_edt_time(time_str):
    """解析中国时间字符串"""
    try:
        return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    except:
        return None

def get_hour_bucket(dt, bucket_size=2):
    """获取时间所属的区间（每 bucket_size 小时一个区间）"""
    if dt is None:
        return None
    hour = dt.hour
    bucket = (hour // bucket_size) * bucket_size
    return bucket

def generate_time_distribution(jobs):
    """生成24小时时间分布统计（每2小时一个区间）"""
    time_dist = [0] * 12  # 12个区间：0-2, 2-4, ..., 22-24
    
    for job in jobs:
        start_time = job.get('start_time', {})
        edt_time_str = start_time.get('edt')
        if edt_time_str:
            dt = parse_edt_time(edt_time_str)
            if dt:
                bucket = get_hour_bucket(dt, bucket_size=2)
                if bucket is not None and 0 <= bucket < 24:
                    time_dist[bucket // 2] += 1
    
    return time_dist

def find_latest_job(jobs):
    """找到最晚跑的 job（中国时间，6:00为一天截止）"""
    latest_job = None
    latest_time = None
    latest_time_from_6am = None
    
    for job in jobs:
        start_time = job.get('start_time', {})
        edt_time_str = start_time.get('edt')
        if edt_time_str:
            dt = parse_edt_time(edt_time_str)
            if dt:
                # 计算从当天 6:00 开始的时间差
                day_start = dt.replace(hour=6, minute=0, second=0, microsecond=0)
                
                # 如果当前时间在 6:00 之前，则属于前一天
                if dt < day_start:
                    day_start = day_start - timedelta(days=1)
                
                time_from_6am = (dt - day_start).total_seconds()
                
                if latest_time_from_6am is None or time_from_6am > latest_time_from_6am:
                    latest_time_from_6am = time_from_6am
                    latest_job = job
                    latest_time = dt
    
    return latest_job, latest_time, latest_time_from_6am

def plot_time_distribution(time_dist, output_file='bird_jobs_time_distribution.png'):
    """生成时间分布柱状图"""
    buckets = [f'{i*2:02d}:00-{(i+1)*2:02d}:00' for i in range(12)]
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(buckets, time_dist, color='steelblue', edgecolor='black', alpha=0.7)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Time Interval (edt Time)', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Jobs', fontsize=12, fontweight='bold')
    plt.title('Distribution of Bird Jobs by Start Time (2-hour intervals)', 
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n柱状图已保存到: {output_file}")
    plt.close()

def main():
    # 加载数据
    print("正在加载 legacy.json...")
    data = load_legacy_json('legacy.json')
    print(f"总共 {len(data)} 个 jobs")
    
    # 过滤 bird 用户的 jobs
    bird_jobs = filter_bird_jobs(data)
    print(f"\n找到 {len(bird_jobs)} 个 bird 用户的 jobs")
    
    if len(bird_jobs) == 0:
        print("没有找到 bird 用户的 jobs，退出。")
        return
    
    # 1. 统计各 status 的个数
    print("\n" + "="*50)
    print("1. Status 统计：")
    print("="*50)
    status_counts = count_statuses(bird_jobs)
    for status, count in sorted(status_counts.items()):
        print(f"  {status:15s}: {count:4d}")
    print(f"  {'Total':15s}: {sum(status_counts.values()):4d}")
    
    # 2. 生成时间分布统计
    print("\n" + "="*50)
    print("2. 24小时时间分布统计（每2小时一个区间）：")
    print("="*50)
    time_dist = generate_time_distribution(bird_jobs)
    for i, count in enumerate(time_dist):
        bucket_label = f'{i*2:02d}:00-{(i+1)*2:02d}:00'
        print(f"  {bucket_label}: {count:4d}")
    
    # 生成柱状图
    plot_time_distribution(time_dist)
    
    # 3. 找到最晚跑的 job
    print("\n" + "="*50)
    print("3. 最晚跑的 job（中国时间，6:00为一天截止）：")
    print("="*50)
    latest_job, latest_time, latest_time_from_6am = find_latest_job(bird_jobs)
    
    if latest_job:
        hours = int(latest_time_from_6am // 3600)
        minutes = int((latest_time_from_6am % 3600) // 60)
        
        print(f"  开始时间 (edt): {latest_job['start_time']['edt']}")
        print(f"  距离当天 6:00: {hours} 小时 {minutes} 分钟")
        print(f"  Status: {latest_job.get('status', 'unknown')}")
        print(f"  TPU: {latest_job.get('tpu', 'unknown')}")
        print(f"  Job Tags: {latest_job.get('job_tags', 'N/A')}")
        print(f"  Windows ID: {latest_job.get('windows_id', 'unknown')}")
    else:
        print("  未找到有效的 job")
    
    print("\n" + "="*50)
    print("分析完成！")
    print("="*50)

if __name__ == '__main__':
    main()


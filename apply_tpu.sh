#!/bin/bash

# 获取脚本参数以替换 'py'
input_value=$1

while true; do
    echo "正在运行 'tpu apply $input_value'..."
    python /home/jzc/zhichengjiang/working/xibo_tpu_manager/tpu.py apply $input_value

    # 检查退出码
    if [ $? -eq 0 ]; then
        echo "'tpu apply $input_value' 成功执行！"
        break
    else
        echo "'tpu apply $input_value' 执行失败，重试中..."
    fi
    
    # 可选：添加延迟以避免过于频繁的重试
    sleep 5
done

# python /home/jzc/zhichengjiang/working/xibo_tpu_manager/tpu.py run $input_value 2 bird
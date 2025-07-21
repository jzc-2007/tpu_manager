bash apply_tpu.sh py &
bash apply_tpu.sh py2 &
bash apply_tpu.sh py3 &
bash apply_tpu.sh py4 &

# 等待所有后台任务完成
wait
echo "所有 TPU 应用命令已完成。"
# 结束脚本
exit 0
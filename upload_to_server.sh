#!/bin/bash
# 上传LW-YOLOv8项目到服务器

SERVER="root@connect.cqa1.seetacloud.com"
PORT="24095"
REMOTE_DIR="/autodl-tmp/ultralytics"

echo "🚀 开始上传LW-YOLOv8项目到服务器..."

# 1. 创建远程目录
echo "📁 创建远程目录..."
ssh -p $PORT $SERVER "mkdir -p $REMOTE_DIR"

# 2. 上传核心代码文件
echo "📤 上传核心代码文件..."
scp -P $PORT train_lw_yolov8.py $SERVER:$REMOTE_DIR/
scp -P $PORT inference_lw_yolov8.py $SERVER:$REMOTE_DIR/
scp -P $PORT server_train_compare.py $SERVER:$REMOTE_DIR/
scp -P $PORT run_all.py $SERVER:$REMOTE_DIR/

# 3. 上传自定义模块
echo "📤 上传自定义模块..."
scp -P $PORT -r ultralytics/ $SERVER:$REMOTE_DIR/

# 4. 上传数据集配置
echo "📤 上传数据集配置..."
scp -P $PORT datasets/dataset.yaml $SERVER:$REMOTE_DIR/datasets/

# 5. 上传模型配置
echo "📤 上传模型配置..."
scp -P $PORT ultralytics/cfg/models/v8/lw-yolov8.yaml $SERVER:$REMOTE_DIR/ultralytics/cfg/models/v8/

# 6. 上传文档
echo "📤 上传文档..."
scp -P $PORT README_LW_YOLOv8.md $SERVER:$REMOTE_DIR/
scp -P $PORT 使用说明.md $SERVER:$REMOTE_DIR/
scp -P $PORT 服务器部署指南.md $SERVER:$REMOTE_DIR/

echo "✅ 上传完成！"
echo ""
echo "🔗 连接到服务器："
echo "ssh -p $PORT $SERVER"
echo ""
echo "🚀 启动训练："
echo "cd $REMOTE_DIR"
echo "tmux new-session -d -s yolo_training"
echo "tmux attach -t yolo_training"
echo "python server_train_compare.py"
echo ""
echo "📊 监控进度："
echo "tmux attach -t yolo_training"
echo "tail -f logs/baseline-yolov8.log"
echo "tail -f logs/lw-yolov8.log"

# 注意：数据集太大，需要单独上传
echo ""
echo "⚠️  注意：数据集需要单独上传（文件较大）："
echo "scp -P $PORT -r datasets/ $SERVER:$REMOTE_DIR/" 
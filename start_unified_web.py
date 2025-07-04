#!/usr/bin/env python3
"""
统一YOLOv8 Web应用启动脚本
集成训练、推理、模型管理功能
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path


def check_files():
    """检查必要文件"""
    required_files = [
        'unified_web_yolo.py',
        'templates/unified_yolo.html',
        'train_model.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 缺少文件: {', '.join(missing_files)}")
        return False
    
    print("✅ 必要文件检查通过")
    return True

def create_directories():
    """创建必要目录"""
    dirs = ['templates', 'static', 'runs', 'runs/train', 'uploads', 'temp']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✅ 目录结构创建完成")

def check_datasets():
    """检查数据集"""
    dataset_paths = [
        'dataset_OnHands/data.yaml',
    ]
    
    found_datasets = []
    for path in dataset_paths:
        if Path(path).exists():
            found_datasets.append(path)
    
    if found_datasets:
        print(f"✅ 找到数据集: {', '.join(found_datasets)}")
    else:
        print("⚠️  未找到数据集，请确保数据集配置正确")
    
    return len(found_datasets) > 0

def main():
    print("🚀 启动统一YOLOv8 Web应用...")
    print("=" * 60)
    
    
    # 检查文件
    if not check_files():
        print("\n请确保所有必要文件存在")
        sys.exit(1)
    
    # 创建目录
    create_directories()
    
    # 检查数据集
    check_datasets()
    
    print("\n📋 功能说明:")
    print("  🏠 概览页面 - 系统状态总览")
    print("  📈 训练页面 - 实时训练监控")
    print("  🔍 推理页面 - 图像检测测试")
    print("  💾 模型管理 - 模型对比下载")
    
    print(f"\n🌐 启动Web服务器...")
    print("访问地址: http://localhost:5000")
    print("按 Ctrl+C 停止服务器")
    print("=" * 60)
    
    # 延迟打开浏览器
    def open_browser():
        time.sleep(3)
        try:
            webbrowser.open('http://localhost:5000')
            print("🌐 浏览器已自动打开")
        except:
            print("⚠️  请手动打开浏览器访问 http://localhost:5000")
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # 启动Web应用
    try:
        # 导入并运行统一Web应用
        from unified_web_yolo import app, socketio
        print("📱 Web应用启动成功!")
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n\n👋 服务器已停止")
    except ImportError as e:
        print(f"\n❌ 导入错误: {e}")
        print("请确保 unified_web_yolo.py 文件存在且无语法错误")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print("请检查:")
        print("1. 端口5000是否被占用")
        print("2. 文件权限是否正确")
        print("3. Python环境是否配置正确")

if __name__ == '__main__':
    main() 
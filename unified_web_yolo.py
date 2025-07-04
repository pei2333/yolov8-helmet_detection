from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_socketio import SocketIO, emit
import threading
import subprocess
import os
import sys
import json
import time
import yaml
from pathlib import Path
import queue
import signal
import psutil
from datetime import datetime
import re
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import torch
from ultralytics import YOLO
import shutil
import tempfile
import zipfile

# Flask应用配置
app = Flask(__name__)
app.config['SECRET_KEY'] = 'unified_yolo_web_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB最大上传
socketio = SocketIO(app, cors_allowed_origins="*", max_size=50*1024*1024)

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 全局变量
training_process = None
current_training_info = {
    'status': 'idle',
    'model': None,
    'epochs': 0,
    'current_epoch': 0,
    'metrics': {},
    'start_time': None,
    'log_buffer': [],
    'process_id': None
}

# 推理状态
inference_models = {}
current_inference_model = None

class TrainingMonitor:
    """训练监控器"""
    def __init__(self):
        self.process = None
        self.log_queue = queue.Queue()
        
    def start_training(self, model_type, epochs=100, data_path='dataset_OnHands/data.yaml', batch_size=8):
        """启动训练进程"""
        global current_training_info
        
        if self.process and self.process.poll() is None:
            return False, "训练已在进行中"
        
        # 构建训练命令
        cmd = [
            sys.executable, 'train_model.py',  # 使用当前Python解释器
            model_type,
            '--epochs', str(epochs),
            '--data', data_path,
            '--workers', '0',
            '--batch', str(batch_size)
        ]
        
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        
        try:
            # 启动训练进程
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                cwd=os.getcwd(),
                env=env,
                encoding='utf-8',
                errors='ignore'
            )
            
            # 更新训练信息
            current_training_info.update({
                'status': 'training',
                'model': model_type,
                'epochs': epochs,
                'current_epoch': 0,
                'start_time': datetime.now().isoformat(),
                'process_id': self.process.pid,
                'log_buffer': [],
                'batch_size': batch_size,
                'data_path': data_path
            })
            
            # 启动日志监控线程
            log_thread = threading.Thread(target=self._monitor_logs)
            log_thread.daemon = True
            log_thread.start()
            
            return True, "训练启动成功"
            
        except Exception as e:
            return False, f"启动训练失败: {str(e)}"
    
    def stop_training(self):
        """停止训练进程"""
        global current_training_info
        
        if self.process and self.process.poll() is None:
            try:
                # 尝试优雅关闭
                parent = psutil.Process(self.process.pid)
                children = parent.children(recursive=True)
                for child in children:
                    child.terminate()
                parent.terminate()
                
                # 等待进程结束
                gone, alive = psutil.wait_procs([parent] + children, timeout=5)
                for p in alive:
                    p.kill()
                
                current_training_info['status'] = 'stopped'
                socketio.emit('training_stopped', {'message': '训练已停止'})
                return True, "训练已停止"
                
            except Exception as e:
                return False, f"停止训练失败: {str(e)}"
        else:
            return False, "没有正在运行的训练进程"
    
    def _monitor_logs(self):
        """监控训练日志输出"""
        global current_training_info
        
        if not self.process:
            return
        
        try:
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    line = line.strip()
                    current_training_info['log_buffer'].append(line)
                    
                    # 保持日志缓冲区大小
                    if len(current_training_info['log_buffer']) > 1000:
                        current_training_info['log_buffer'] = current_training_info['log_buffer'][-500:]
                    
                    # 解析训练指标
                    self._parse_metrics(line)
                    
                    # 实时发送日志
                    socketio.emit('training_log', {'log': line})
            
            # 进程结束
            self.process.wait()
            if current_training_info['status'] == 'training':
                current_training_info['status'] = 'completed'
                socketio.emit('training_completed', {'message': '训练完成'})
                
        except Exception as e:
            current_training_info['status'] = 'error'
            socketio.emit('training_error', {'error': str(e)})
    
    def _parse_metrics(self, log_line):
        """解析训练指标"""
        global current_training_info
        
        # 解析epoch信息
        epoch_match = re.search(r'Epoch\s+(\d+)/(\d+)', log_line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            total_epochs = int(epoch_match.group(2))
            current_training_info['current_epoch'] = current_epoch
            current_training_info['epochs'] = total_epochs
            
            progress = (current_epoch / total_epochs) * 100
            socketio.emit('training_progress', {
                'epoch': current_epoch,
                'total_epochs': total_epochs,
                'progress': progress
            })
        
        # 解析mAP指标
        map_match = re.search(r'mAP50:\s*([\d.]+)', log_line)
        if map_match:
            map50 = float(map_match.group(1))
            current_training_info['metrics']['mAP50'] = map50
            socketio.emit('metrics_update', {'mAP50': map50})
        
        # 解析loss信息
        loss_match = re.search(r'loss:\s*([\d.]+)', log_line)
        if loss_match:
            loss = float(loss_match.group(1))
            current_training_info['metrics']['loss'] = loss
            socketio.emit('metrics_update', {'loss': loss})

class InferenceEngine:
    """推理引擎"""
    def __init__(self):
        self.models = {}
        self.current_model = None
        
    def load_model(self, model_path, model_name=None):
        """加载模型"""
        try:
            if not Path(model_path).exists():
                return False, f"模型文件不存在: {model_path}"
            
            model = YOLO(model_path)
            if model_name is None:
                model_name = Path(model_path).stem
            
            self.models[model_name] = {
                'model': model,
                'path': model_path,
                'loaded_time': datetime.now().isoformat()
            }
            
            return True, f"模型 {model_name} 加载成功"
            
        except Exception as e:
            return False, f"加载模型失败: {str(e)}"
    
    def set_current_model(self, model_name):
        """设置当前使用的模型"""
        if model_name in self.models:
            self.current_model = model_name
            return True, f"切换到模型: {model_name}"
        else:
            return False, f"模型 {model_name} 未加载"
    
    def predict_image(self, image_data, conf_threshold=0.5):
        """图像推理"""
        if not self.current_model or self.current_model not in self.models:
            return False, "没有选择有效的模型", None
        
        try:
            model = self.models[self.current_model]['model']
            
            # 处理base64图像数据
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                # 解码base64图像
                header, data = image_data.split(',', 1)
                image_bytes = base64.b64decode(data)
                image = Image.open(BytesIO(image_bytes))
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                # 直接使用图像路径
                image = cv2.imread(image_data)
            
            if image is None:
                return False, "无法读取图像", None
            
            # 推理
            results = model(image, conf=conf_threshold)
            
            # 处理结果
            result_data = self._process_results(results[0], image)
            
            return True, "推理成功", result_data
            
        except Exception as e:
            return False, f"推理失败: {str(e)}", None
    
    def _process_results(self, result, original_image):
        """处理推理结果"""
        try:
            # 绘制检测框
            annotated_image = result.plot()
            
            # 转换为base64
            _, buffer = cv2.imencode('.jpg', annotated_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 提取检测信息
            detections = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                class_names = {0: "head", 1: "helmet"}
                
                for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                    x1, y1, x2, y2 = box
                    detections.append({
                        'id': i,
                        'class': class_names.get(int(cls), f"class_{int(cls)}"),
                        'confidence': float(conf),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'area': float((x2-x1) * (y2-y1))
                    })
            
            return {
                'image': f"data:image/jpeg;base64,{image_base64}",
                'detections': detections,
                'summary': {
                    'total_detections': len(detections),
                    'head_count': len([d for d in detections if d['class'] == 'head']),
                    'helmet_count': len([d for d in detections if d['class'] == 'helmet']),
                    'safety_rate': len([d for d in detections if d['class'] == 'helmet']) / max(1, len(detections)) * 100
                }
            }
            
        except Exception as e:
            raise Exception(f"处理结果失败: {str(e)}")
    
    def get_model_info(self, model_name):
        """获取模型信息"""
        if model_name not in self.models:
            return None
        
        model_info = self.models[model_name]
        model = model_info['model']
        
        try:
            # 获取模型统计信息
            model_stats = {
                'name': model_name,
                'path': model_info['path'],
                'loaded_time': model_info['loaded_time'],
                'task': getattr(model, 'task', 'detect'),
                'names': getattr(model.model, 'names', {}),
                'device': str(next(model.model.parameters()).device) if hasattr(model, 'model') else 'unknown'
            }
            
            # 计算模型大小
            model_path = Path(model_info['path'])
            if model_path.exists():
                model_stats['file_size'] = model_path.stat().st_size
                model_stats['file_size_mb'] = round(model_stats['file_size'] / 1024 / 1024, 2)
            
            return model_stats
            
        except Exception as e:
            return {'name': model_name, 'error': str(e)}

# 创建实例
monitor = TrainingMonitor()
inference_engine = InferenceEngine()

# ============= 路由定义 =============

@app.route('/')
def index():
    """主页面"""
    return render_template('unified_yolo.html')

@app.route('/training')
def training_page():
    """训练页面"""
    return render_template('unified_yolo.html', page='training')

@app.route('/inference')
def inference_page():
    """推理页面"""
    return render_template('unified_yolo.html', page='inference')

@app.route('/models')
def models_page():
    """模型管理页面"""
    return render_template('unified_yolo.html', page='models')

# ============= 训练API =============

@app.route('/api/models')
def get_models():
    """获取可用模型列表"""
    models = [
        {'id': 'baseline', 'name': 'YOLOv8s Baseline', 'description': '基线模型', 'category': 'baseline'},
        {'id': 'csp-ctfn', 'name': 'CSP-CTFN Only', 'description': 'CSP-CTFN模块优化', 'category': 'component'},
        {'id': 'psc-head', 'name': 'PSC-Head Only', 'description': 'PSC检测头优化', 'category': 'component'},
        {'id': 'siou', 'name': 'SIoU Only', 'description': 'SIoU损失函数优化', 'category': 'component'},
        {'id': 'lw-yolov8', 'name': 'LW-YOLOv8 Full', 'description': '完整轻量化模型', 'category': 'lightweight'},
        {'id': 'improved-csp-ctfn', 'name': 'Improved CSP-CTFN', 'description': '改进的CSP-CTFN模块', 'category': 'improved'},
        {'id': 'plus', 'name': 'YOLOv8-PLUS', 'description': '集成C3k2+SPPF+C2PSA的增强版本', 'category': 'enhanced'}
    ]
    return jsonify(models)

@app.route('/api/datasets')
def get_datasets():
    """获取可用数据集"""
    datasets = []
    dataset_paths = [
        'dataset_OnHands/data.yaml',
        'datasets_mini/dataset_mini.yaml'
    ]
    
    for path in dataset_paths:
        if Path(path).exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                datasets.append({
                    'path': path,
                    'name': Path(path).stem,
                    'train': data.get('train', ''),
                    'val': data.get('val', ''),
                    'nc': data.get('nc', 0),
                    'names': data.get('names', [])
                })
            except:
                datasets.append({
                    'path': path,
                    'name': Path(path).stem,
                    'error': '无法读取配置文件'
                })
    
    return jsonify(datasets)

@app.route('/api/start_training', methods=['POST'])
def start_training():
    """启动训练"""
    data = request.json
    model_type = data.get('model', 'baseline')
    epochs = data.get('epochs', 100)
    data_path = data.get('dataset', 'dataset_OnHands/data.yaml')
    batch_size = data.get('batch_size', 8)
    
    success, message = monitor.start_training(model_type, epochs, data_path, batch_size)
    return jsonify({'success': success, 'message': message})

@app.route('/api/stop_training', methods=['POST'])
def stop_training():
    """停止训练"""
    success, message = monitor.stop_training()
    return jsonify({'success': success, 'message': message})

@app.route('/api/training_status')
def get_training_status():
    """获取训练状态"""
    return jsonify(current_training_info)

@app.route('/api/runs')
def get_runs():
    """获取训练结果列表"""
    runs_dir = Path('runs/train')
    runs = []
    
    if runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                weights_file = run_dir / 'weights' / 'best.pt'
                results_file = run_dir / 'results.png'
                
                # 尝试读取训练配置
                config_info = {}
                try:
                    args_file = run_dir / 'args.yaml'
                    if args_file.exists():
                        with open(args_file, 'r') as f:
                            config_info = yaml.safe_load(f)
                except:
                    pass
                
                run_info = {
                    'name': run_dir.name,
                    'path': str(run_dir),
                    'has_weights': weights_file.exists(),
                    'has_results': results_file.exists(),
                    'created': run_dir.stat().st_mtime,
                    'weights_path': str(weights_file) if weights_file.exists() else None,
                    'results_path': str(results_file) if results_file.exists() else None,
                    'config': config_info
                }
                runs.append(run_info)
    
    # 按创建时间排序
    runs.sort(key=lambda x: x['created'], reverse=True)
    return jsonify(runs)

# ============= 推理API =============

@app.route('/api/inference/models')
def get_inference_models():
    """获取已加载的推理模型"""
    models_info = []
    for name, info in inference_engine.models.items():
        model_info = inference_engine.get_model_info(name)
        if model_info:
            model_info['is_current'] = (name == inference_engine.current_model)
            models_info.append(model_info)
    
    return jsonify(models_info)

@app.route('/api/inference/load_model', methods=['POST'])
def load_inference_model():
    """加载推理模型"""
    data = request.json
    model_path = data.get('model_path')
    model_name = data.get('model_name')
    
    if not model_path:
        return jsonify({'success': False, 'message': '请提供模型路径'})
    
    success, message = inference_engine.load_model(model_path, model_name)
    return jsonify({'success': success, 'message': message})

@app.route('/api/inference/set_model', methods=['POST'])
def set_inference_model():
    """设置当前推理模型"""
    data = request.json
    model_name = data.get('model_name')
    
    success, message = inference_engine.set_current_model(model_name)
    return jsonify({'success': success, 'message': message})

@app.route('/api/inference/predict', methods=['POST'])
def predict_image():
    """图像推理"""
    try:
        if 'image' in request.files:
            # 文件上传
            file = request.files['image']
            if file.filename == '':
                return jsonify({'success': False, 'message': '没有选择文件'})
            
            # 保存临时文件
            temp_path = os.path.join(tempfile.gettempdir(), f"temp_inference_{int(time.time())}_{file.filename}")
            file.save(temp_path)
            
            conf_threshold = float(request.form.get('conf_threshold', 0.5))
            
            success, message, result_data = inference_engine.predict_image(temp_path, conf_threshold)
            
            # 清理临时文件
            try:
                os.remove(temp_path)
            except:
                pass
            
        else:
            # JSON数据
            data = request.json
            image_data = data.get('image_data')
            conf_threshold = data.get('conf_threshold', 0.5)
            
            success, message, result_data = inference_engine.predict_image(image_data, conf_threshold)
        
        if success:
            return jsonify({'success': True, 'message': message, 'result': result_data})
        else:
            return jsonify({'success': False, 'message': message})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'推理失败: {str(e)}'})

# ============= 模型管理API =============

@app.route('/api/models/compare', methods=['POST'])
def compare_models():
    """模型对比"""
    data = request.json
    run_names = data.get('runs', [])
    
    comparison_data = []
    for run_name in run_names:
        run_dir = Path('runs/train') / run_name
        if run_dir.exists():
            # 读取训练结果
            results = {}
            try:
                # 尝试读取结果文件
                results_csv = run_dir / 'results.csv'
                if results_csv.exists():
                    import pandas as pd
                    df = pd.read_csv(results_csv)
                    if not df.empty:
                        latest = df.iloc[-1]
                        results = {
                            'mAP50': float(latest.get('metrics/mAP50(B)', 0)),
                            'mAP50-95': float(latest.get('metrics/mAP50-95(B)', 0)),
                            'precision': float(latest.get('metrics/precision(B)', 0)),
                            'recall': float(latest.get('metrics/recall(B)', 0)),
                            'final_epoch': int(latest.get('epoch', 0))
                        }
            except:
                pass
            
            # 获取模型大小
            weights_file = run_dir / 'weights' / 'best.pt'
            model_size = 0
            if weights_file.exists():
                model_size = weights_file.stat().st_size / 1024 / 1024  # MB
            
            comparison_data.append({
                'name': run_name,
                'metrics': results,
                'model_size_mb': round(model_size, 2),
                'weights_path': str(weights_file) if weights_file.exists() else None
            })
    
    return jsonify(comparison_data)

@app.route('/api/models/download/<run_name>')
def download_model(run_name):
    """下载模型权重"""
    run_dir = Path('runs/train') / run_name
    weights_file = run_dir / 'weights' / 'best.pt'
    
    if weights_file.exists():
        return send_file(weights_file, as_attachment=True, download_name=f'{run_name}_best.pt')
    else:
        return jsonify({'error': '模型文件不存在'}), 404

@app.route('/api/models/export', methods=['POST'])
def export_models():
    """导出多个模型为ZIP包"""
    data = request.json
    run_names = data.get('runs', [])
    
    if not run_names:
        return jsonify({'error': '没有选择模型'}), 400
    
    # 创建临时ZIP文件
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    
    try:
        with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
            for run_name in run_names:
                run_dir = Path('runs/train') / run_name
                weights_file = run_dir / 'weights' / 'best.pt'
                
                if weights_file.exists():
                    zipf.write(weights_file, f'{run_name}_best.pt')
                
                # 添加结果图片
                results_file = run_dir / 'results.png'
                if results_file.exists():
                    zipf.write(results_file, f'{run_name}_results.png')
        
        return send_file(temp_zip.name, as_attachment=True, download_name='yolo_models.zip')
        
    except Exception as e:
        return jsonify({'error': f'导出失败: {str(e)}'}), 500
    finally:
        # 清理临时文件将在响应发送后进行
        pass

# ============= 文件服务 =============

@app.route('/results/<path:filename>')
def serve_results(filename):
    """提供结果文件服务"""
    return send_from_directory('runs/train', filename)

# ============= WebSocket处理 =============

@socketio.on('connect')
def handle_connect():
    """WebSocket连接处理"""
    emit('connected', {'message': '连接成功'})
    # 发送当前训练状态
    emit('training_status', current_training_info)

@socketio.on('disconnect')
def handle_disconnect():
    """WebSocket断开处理"""
    print('客户端断开连接')

@socketio.on('request_logs')
def handle_request_logs():
    """请求历史日志"""
    logs = current_training_info.get('log_buffer', [])
    emit('historical_logs', {'logs': logs})

@socketio.on('join_room')
def handle_join_room(data):
    """加入房间（用于分组通信）"""
    room = data.get('room', 'default')
    # join_room(room)
    emit('joined_room', {'room': room})

# ============= 工具函数 =============

def ensure_directories():
    """确保必要的目录存在"""
    dirs = ['templates', 'static', 'runs', 'runs/train', 'uploads', 'temp']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_available_models():
    """自动加载可用的模型"""
    runs_dir = Path('runs/train')
    if runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                weights_file = run_dir / 'weights' / 'best.pt'
                if weights_file.exists():
                    success, message = inference_engine.load_model(str(weights_file), run_dir.name)
                    if success:
                        print(f"✅ 自动加载模型: {run_dir.name}")

# ============= 应用启动 =============

if __name__ == '__main__':
    # 确保目录结构
    ensure_directories()
    
    # 自动加载可用模型
    load_available_models()
    
    print("🚀 启动统一YOLOv8 Web应用...")
    print("功能包括:")
    print("  📈 实时训练监控")
    print("  🔍 模型推理测试")
    print("  📊 模型性能对比")
    print("  💾 模型管理下载")
    print("访问地址: http://localhost:5000")
    print("=" * 50)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True) 
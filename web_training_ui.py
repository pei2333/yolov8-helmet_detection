from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import subprocess
import os
import json
import time
import yaml
from pathlib import Path
import queue
import signal
import psutil
from datetime import datetime
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = 'yolo_training_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# 全局变量
training_process = None
training_queue = queue.Queue()
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

class TrainingMonitor:
    def __init__(self):
        self.process = None
        self.log_queue = queue.Queue()
        
    def start_training(self, model_type, epochs=100, data_path='datasets_mini/dataset_mini.yaml'):
        """启动训练进程"""
        global current_training_info
        
        if self.process and self.process.poll() is None:
            return False, "训练已在进行中"
        
        # 构建训练命令
        cmd = [
            'python', 'train_model.py',
            model_type,
            '--epochs', str(epochs),
            '--data', data_path,
            '--workers', '0',
            '--batch', '8'
        ]
        
        try:
            # 启动训练进程
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                cwd=os.getcwd()
            )
            
            # 更新训练信息
            current_training_info.update({
                'status': 'training',
                'model': model_type,
                'epochs': epochs,
                'current_epoch': 0,
                'start_time': datetime.now().isoformat(),
                'process_id': self.process.pid,
                'log_buffer': []
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

# 创建训练监控器实例
monitor = TrainingMonitor()

@app.route('/')
def index():
    """主页面"""
    return render_template('training_ui.html')

@app.route('/api/models')
def get_models():
    """获取可用模型列表"""
    models = [
        {'id': 'baseline', 'name': 'YOLOv8s Baseline', 'description': '基线模型'},
        {'id': 'csp-ctfn', 'name': 'CSP-CTFN Only', 'description': 'CSP-CTFN模块优化'},
        {'id': 'psc-head', 'name': 'PSC-Head Only', 'description': 'PSC检测头优化'},
        {'id': 'siou', 'name': 'SIoU Only', 'description': 'SIoU损失函数优化'},
        {'id': 'lw-yolov8', 'name': 'LW-YOLOv8 Full', 'description': '完整轻量化模型'},
        {'id': 'improved-csp-ctfn', 'name': 'Improved CSP-CTFN', 'description': '改进的CSP-CTFN模块'}
    ]
    return jsonify(models)

@app.route('/api/start_training', methods=['POST'])
def start_training():
    """启动训练"""
    data = request.json
    model_type = data.get('model', 'baseline')
    epochs = data.get('epochs', 100)
    
    success, message = monitor.start_training(model_type, epochs)
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
                
                run_info = {
                    'name': run_dir.name,
                    'path': str(run_dir),
                    'has_weights': weights_file.exists(),
                    'has_results': results_file.exists(),
                    'created': run_dir.stat().st_mtime
                }
                runs.append(run_info)
    
    # 按创建时间排序
    runs.sort(key=lambda x: x['created'], reverse=True)
    return jsonify(runs)

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

if __name__ == '__main__':
    # 确保模板目录存在
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("启动YOLOv8训练Web UI...")
    print("访问地址: http://localhost:5000")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True) 
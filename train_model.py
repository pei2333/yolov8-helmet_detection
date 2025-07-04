import os
import sys
import argparse

# 设置环境变量和编码
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform.startswith('win'):
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'Chinese')
    except:
        pass

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    import torch
    import numpy as np
    from ultralytics import YOLO
    print(f"SUCCESS: Import dependencies - PyTorch {torch.__version__}, NumPy {np.__version__}", flush=True)
except ImportError as e:
    print(f"ERROR: Import failed - {e}", flush=True)
    print("Please ensure all dependencies are installed", flush=True)
    sys.exit(1)

def train_model(model_name, epochs=100, data_path='dataset_OnHands\data.yaml'):
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    data_yaml = data_path
    model_configs = {
        'baseline': 'yolov8s.pt',
        'csp-ctfn': 'ultralytics/cfg/models/v8/csp-ctfn-only.yaml',
        'psc-head': 'ultralytics/cfg/models/v8/psc-head-only.yaml',
        'siou': 'ultralytics/cfg/models/v8/siou-only.yaml',
        'lw-yolov8': 'ultralytics/cfg/models/v8/lw-yolov8-full.yaml',
        'improved-csp-ctfn': 'ultralytics/cfg/models/v8/improved-csp-ctfn.yaml',
        'plus': 'ultralytics/cfg/models/v8/lw-yolov8-plus.yaml',
    }
    
    if model_name not in model_configs:
        print(f"ERROR: Unknown model - {model_name}", flush=True)
        print(f"Available models: {list(model_configs.keys())}", flush=True)
        return
    
    # 训练参数（全量数据 + 增强数据增强）
    train_args = {
        'data': data_yaml,
        'batch': 8, 
        'imgsz': 640,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'workers': 8,          # 多进程数据加载
        'cache': 'ram',        # 缓存到内存加速训练
        'project': 'runs/train',
        'patience': 30,
        'save': True,
        'plots': True,
        'verbose': True,
        'amp': True,
        'exist_ok': True,
        'epochs': epochs,
        'fraction': 1.0,       # 使用100%的训练数据
        
        # 增强的数据增强参数
        'hsv_h': 0.025,        # 色调变化
        'hsv_s': 0.8,          # 饱和度变化  
        'hsv_v': 0.6,          # 亮度变化
        'degrees': 20.0,       # 旋转角度
        'translate': 0.15,     # 平移范围
        'scale': 0.8,          # 缩放范围
        'shear': 8.0,          # 剪切变换
        'perspective': 0.0005, # 透视变换
        'fliplr': 0.5,         # 水平翻转
        'flipud': 0.0,         # 不使用垂直翻转
        'mosaic': 1.0,         # 马赛克增强
        'mixup': 0.15,         # 图像混合
        'copy_paste': 0.3,     # 复制粘贴
        'erasing': 0.4,        # 随机擦除
        'auto_augment': 'randaugment',  # 自动增强
        'augment': True,       # 启用所有增强
    }
    
    print(f"Starting training: {model_name}", flush=True)
    print(f"Config: {model_configs[model_name]}", flush=True)
    print("=" * 60, flush=True)
    
    try:
        if not os.path.exists(data_yaml):
            print(f"ERROR: Dataset file not found - {data_yaml}", flush=True)
            return
            
        print(f"Using dataset: {data_yaml}", flush=True)
        torch.cuda.empty_cache()
        model = YOLO(model_configs[model_name])
        
        results = model.train(**train_args, name=f'{model_name}-{epochs}ep')
        print(f"SUCCESS: {model_name} training completed!", flush=True)
        
        if hasattr(results, 'results_dict'):
            print(f"Final mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}", flush=True)
            print(f"Final mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}", flush=True)
            
    except Exception as e:
        print(f"ERROR: {model_name} training failed - {str(e)}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练LW-YOLOv8模型')
    parser.add_argument('model', help='要训练的模型名称')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--data', type=str, default='dataset_OnHands\data.yaml', help='数据集YAML文件路径')
    parser.add_argument('--batch', type=int, default=32, help='批次大小')
    parser.add_argument('--workers', type=int, default=8, help='数据加载器进程数')
    
    args = parser.parse_args()
    
    print("Available models:", flush=True)
    print("- baseline: YOLOv8s baseline", flush=True)
    print("- csp-ctfn: CSP-CTFN module", flush=True)
    print("- psc-head: PSC-Head module", flush=True)
    print("- siou: SIoU Loss module", flush=True)
    print("- lw-yolov8: Complete LW-YOLOv8", flush=True)
    print("- improved-csp-ctfn: Improved CSP-CTFN", flush=True)
    print("- plus: Enhanced PLUS with C3k2+SPPF+C2PSA", flush=True)
    print(flush=True)
    
    train_model(args.model, args.epochs, args.data)
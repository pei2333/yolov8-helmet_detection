#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def show_menu():
    print("\n" + "="*70)
    print("🚧 轻量化安全帽检测系统")
    print("="*70)
    print("1. 训练基线YOLOv8模型")
    print("2. 训练轻量化改进模型")
    print("3. 🔥 全量数据训练 (完整数据集)")
    print("4. 模型性能对比评估")
    print("5. 实时检测 (摄像头/视频)")
    print("6. 测试模块导入")
    print("7. 模型架构分析")
    print("8. 完整微调测试 (集成所有模块)")
    print("9. 模型推理基准测试")
    print("0. 退出")
    print("="*70)

def select_dataset():
    print("\n📊 选择训练数据集:")
    print("1. 子集 (500张图像) - 快速验证")
    print("2. 中等规模 (1500张图像) - 平衡训练")
    print("3. 完整数据集 (5000张图像) - 完整训练")
    
    while True:
        choice = input("请选择数据集规模 (1-3): ").strip()
        if choice == "1":
            return "subset", 500
        elif choice == "2":
            return "medium", 1500
        elif choice == "3":
            return "full", 5000
        else:
            print("❌ 无效选择，请重新输入")

def train_baseline():
    print("\n🔄 训练基线YOLOv8模型")
    dataset_type, dataset_size = select_dataset()
    
    try:
        from models.baseline_trainer import BaselineTrainer
        trainer = BaselineTrainer(
            dataset_type=dataset_type,
            dataset_size=dataset_size
        )
        trainer.train()
    except Exception as e:
        print(f"❌ 基线模型训练失败: {e}")
        import traceback
        traceback.print_exc()

def train_lightweight():
    print("\n🔄 训练轻量化改进模型")
    dataset_type, dataset_size = select_dataset()
    
    try:
        from models.lightweight_trainer import LightweightTrainer
        trainer = LightweightTrainer(
            dataset_type=dataset_type,
            dataset_size=dataset_size
        )
        trainer.train()
    except Exception as e:
        print(f"❌ 轻量化模型训练失败: {e}")
        import traceback
        traceback.print_exc()

def test_modules():
    print("\n🧪 测试模块导入...")
    
    try:
        print("测试基础依赖...")
        import torch
        import torchvision
        import ultralytics
        import cv2
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ Torchvision: {torchvision.__version__}")
        print(f"✅ Ultralytics: {ultralytics.__version__}")
        print(f"✅ OpenCV: {cv2.__version__}")
        
        print("\n测试自定义模块...")
        from modules.fasternet import FasterNetBlock, C2f_Fast
        from modules.fsdi import FSDI
        from modules.attention import A2_Attention, PAM_Attention
        from modules.losses import FocalerCIOULoss, EnhancedFocalLoss
        print("✅ 所有自定义模块导入成功")
        
        print("\n测试模块功能...")
        import torch
        
        # 测试FasterNet
        faster_block = FasterNetBlock(64, 64)
        test_input = torch.randn(1, 64, 32, 32)
        output = faster_block(test_input)
        print(f"✅ FasterNet: {test_input.shape} -> {output.shape}")
        
        # 测试FSDI
        fsdi = FSDI([256, 512, 1024], 256)
        features = [
            torch.randn(1, 256, 32, 32),
            torch.randn(1, 512, 16, 16),
            torch.randn(1, 1024, 8, 8)
        ]
        fsdi_out = fsdi(features)
        print(f"✅ FSDI: 多尺度融合成功，输出数量: {len(fsdi_out)}")
        
        # 测试注意力
        attention = A2_Attention(256)
        att_input = torch.randn(1, 256, 32, 32)
        att_output = attention(att_input)
        print(f"✅ A2_Attention: {att_input.shape} -> {att_output.shape}")
        
        print("\n🎉 所有模块测试通过！")
        
    except Exception as e:
        print(f"❌ 模块测试失败: {e}")
        import traceback
        traceback.print_exc()

def analyze_models():
    print("\n📊 模型架构分析...")
    
    try:
        import torch
        from modules.fasternet import FasterNetBlock, C2f_Fast
        from modules.fsdi import FSDI
        from modules.attention import A2_Attention
        
        print("=== 模型参数量分析 ===")
        
        # FasterNet Block
        faster_block = FasterNetBlock(256, 256)
        faster_params = sum(p.numel() for p in faster_block.parameters())
        print(f"FasterNet Block (256->256): {faster_params:,} 参数")
        
        # C2f_Fast
        c2f_fast = C2f_Fast(256, 256, n=3)
        c2f_params = sum(p.numel() for p in c2f_fast.parameters())
        print(f"C2f_Fast (n=3): {c2f_params:,} 参数")
        
        # FSDI
        fsdi = FSDI([256, 512, 1024], 256)
        fsdi_params = sum(p.numel() for p in fsdi.parameters())
        print(f"FSDI 融合模块: {fsdi_params:,} 参数")
        
        # A2 Attention
        attention = A2_Attention(256)
        att_params = sum(p.numel() for p in attention.parameters())
        print(f"A2 注意力: {att_params:,} 参数")
        
        print(f"\n总轻量化模块参数: {faster_params + c2f_params + fsdi_params + att_params:,}")
        
        print("\n=== 推理性能测试 ===")
        import time
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"测试设备: {device}")
        
        test_input = torch.randn(1, 256, 64, 64).to(device)
        faster_block = faster_block.to(device)
        
        # 预热
        for _ in range(10):
            _ = faster_block(test_input)
        
        # 测试推理时间
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = faster_block(test_input)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # ms
        print(f"FasterNet平均推理时间: {avg_time:.2f} ms")
        
    except Exception as e:
        print(f"❌ 模型分析失败: {e}")
        import traceback
        traceback.print_exc()

def train_full_dataset():
    """全量数据训练 - 使用完整数据集进行高质量训练"""
    print("\n🔥 全量数据训练")
    print("=" * 50)
    print("📋 配置信息:")
    print("  - 数据集: 完整数据集 (5000+ 张图像)")
    print("  - 模型: 轻量化改进模型")
    print("  - 训练轮数: 200 epochs")
    print("  - 批次大小: 16 (GPU) / 4 (CPU)")
    print("  - 验证频率: 每10轮验证一次")
    print("  - 早停: 耐心度50轮")
    print("=" * 50)
    
    confirm = input("⚠️  全量训练需要较长时间，确认开始？(y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ 用户取消训练")
        return
    
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = 16 if torch.cuda.is_available() else 4
        
        print(f"\n🚀 开始全量数据训练...")
        print(f"   设备: {device}")
        print(f"   批次大小: {batch_size}")
        
        from models.lightweight_trainer import LightweightTrainer
        trainer = LightweightTrainer(
            dataset_type="full",
            dataset_size=5000,
            epochs=200,
            batch_size=batch_size,
            patience=50,
            val_interval=10
        )
        
        print("📊 开始训练，这可能需要数小时...")
        results = trainer.train()
        
        print(f"\n🎉 全量训练完成！")
        print(f"   最佳mAP50: {results.get('best_map50', 'N/A')}")
        print(f"   最佳mAP75: {results.get('best_map75', 'N/A')}")
        print(f"   模型保存路径: {results.get('save_dir', 'N/A')}")
        
    except Exception as e:
        print(f"❌ 全量训练失败: {e}")
        import traceback
        traceback.print_exc()

def evaluate_models():
    """模型性能对比评估"""
    print("\n📊 模型性能对比评估")
    print("=" * 50)
    
    # 检查可用模型
    available_models = []
    model_paths = [
        "runs/detect/baseline/weights/best.pt",
        "runs/detect/lightweight/weights/best.pt",
        "runs/detect/full_training/weights/best.pt"
    ]
    model_names = ["基线模型", "轻量化模型", "全量训练模型"]
    
    for path, name in zip(model_paths, model_names):
        if os.path.exists(path):
            available_models.append((name, path))
    
    if len(available_models) < 2:
        print("❌ 需要至少2个训练好的模型进行对比")
        print("   请先运行训练命令生成模型")
        return
    
    print("📋 可用模型:")
    for i, (name, path) in enumerate(available_models):
        print(f"   {i+1}. {name}: {path}")
    
    try:
        from ultralytics import YOLO
        import time
        
        print("\n🔄 开始性能评估...")
        
        # 创建测试数据
        test_image_path = "test_image.jpg"
        if not os.path.exists(test_image_path):
            import cv2
            import numpy as np
            test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            cv2.imwrite(test_image_path, test_img)
        
        results = {}
        for name, path in available_models:
            print(f"\n测试 {name}...")
            model = YOLO(path)
            
            # 参数量统计
            total_params = sum(p.numel() for p in model.model.parameters())
            
            # 推理速度测试
            warmup_runs = 10
            test_runs = 50
            
            # 预热
            for _ in range(warmup_runs):
                _ = model(test_image_path, verbose=False)
            
            # 测试推理时间
            start_time = time.time()
            for _ in range(test_runs):
                _ = model(test_image_path, verbose=False)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / test_runs * 1000  # ms
            fps = 1000 / avg_time
            
            results[name] = {
                'params': total_params,
                'inference_time': avg_time,
                'fps': fps,
                'model_size': os.path.getsize(path) / (1024 * 1024)  # MB
            }
        
        # 打印对比结果
        print("\n📊 性能对比结果:")
        print("=" * 80)
        print(f"{'模型':<15} {'参数量':<12} {'推理时间(ms)':<12} {'FPS':<8} {'模型大小(MB)':<12}")
        print("-" * 80)
        
        for name, metrics in results.items():
            print(f"{name:<15} {metrics['params']:>10,} {metrics['inference_time']:>10.2f} "
                  f"{metrics['fps']:>6.1f} {metrics['model_size']:>10.1f}")
        
        # 清理测试文件
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            
    except Exception as e:
        print(f"❌ 模型评估失败: {e}")
        import traceback
        traceback.print_exc()

def real_time_detection():
    """实时检测功能"""
    print("\n📹 实时检测")
    print("=" * 50)
    print("1. 摄像头检测")
    print("2. 视频文件检测")
    print("3. 图像检测")
    
    choice = input("请选择检测模式 (1-3): ").strip()
    
    # 选择模型
    model_path = None
    if os.path.exists("runs/detect/lightweight/weights/best.pt"):
        model_path = "runs/detect/lightweight/weights/best.pt"
        print("🤖 使用轻量化模型")
    elif os.path.exists("runs/detect/baseline/weights/best.pt"):
        model_path = "runs/detect/baseline/weights/best.pt"
        print("🤖 使用基线模型")
    else:
        print("❌ 未找到训练好的模型，请先运行训练命令")
        return
    
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        
        if choice == "1":
            print("📷 启动摄像头检测...")
            print("按 'q' 键退出")
            model.predict(source=0, show=True, save=False, stream=True)
            
        elif choice == "2":
            video_path = input("请输入视频文件路径: ").strip()
            if not os.path.exists(video_path):
                print("❌ 视频文件不存在")
                return
            print(f"📹 检测视频: {video_path}")
            model.predict(source=video_path, show=True, save=True)
            
        elif choice == "3":
            image_path = input("请输入图像文件路径: ").strip()
            if not os.path.exists(image_path):
                print("❌ 图像文件不存在")
                return
            print(f"🖼️  检测图像: {image_path}")
            results = model.predict(source=image_path, show=True, save=True)
            
        else:
            print("❌ 无效选择")
            
    except Exception as e:
        print(f"❌ 实时检测失败: {e}")
        import traceback
        traceback.print_exc()

def benchmark_model():
    """模型推理基准测试"""
    print("\n⚡ 模型推理基准测试")
    print("=" * 50)
    
    # 选择模型
    available_models = []
    model_paths = [
        ("基线模型", "runs/detect/baseline/weights/best.pt"),
        ("轻量化模型", "runs/detect/lightweight/weights/best.pt"),
        ("全量训练模型", "runs/detect/full_training/weights/best.pt")
    ]
    
    for name, path in model_paths:
        if os.path.exists(path):
            available_models.append((name, path))
    
    if not available_models:
        print("❌ 未找到训练好的模型")
        return
    
    print("可用模型:")
    for i, (name, _) in enumerate(available_models):
        print(f"  {i+1}. {name}")
    
    try:
        choice = int(input("请选择模型 (1-{}): ".format(len(available_models)))) - 1
        if choice < 0 or choice >= len(available_models):
            print("❌ 无效选择")
            return
        
        model_name, model_path = available_models[choice]
        
        from ultralytics import YOLO
        import torch
        import time
        import numpy as np
        
        print(f"\n🔄 加载模型: {model_name}")
        model = YOLO(model_path)
        
        # 创建不同尺寸的测试数据
        test_sizes = [320, 416, 640, 1024]
        batch_sizes = [1, 4, 8] if torch.cuda.is_available() else [1, 2]
        
        print("\n📊 基准测试结果:")
        print("=" * 80)
        print(f"{'图像尺寸':<10} {'批次大小':<8} {'推理时间(ms)':<12} {'FPS':<8} {'内存使用(MB)':<12}")
        print("-" * 80)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for img_size in test_sizes:
            for batch_size in batch_sizes:
                try:
                    # 创建测试数据
                    test_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
                    
                    # 预热
                    for _ in range(10):
                        with torch.no_grad():
                            _ = model.model(test_input)
                    
                    # 测试推理时间
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start_time = time.time()
                    
                    test_runs = 100
                    for _ in range(test_runs):
                        with torch.no_grad():
                            _ = model.model(test_input)
                    
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.time()
                    
                    avg_time = (end_time - start_time) / test_runs * 1000  # ms
                    fps = 1000 / (avg_time / batch_size)
                    
                    # 内存使用
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                        torch.cuda.reset_peak_memory_stats()
                    else:
                        memory_used = 0
                    
                    print(f"{img_size:<10} {batch_size:<8} {avg_time:<12.2f} {fps:<8.1f} {memory_used:<12.1f}")
                    
                except Exception as e:
                    print(f"{img_size:<10} {batch_size:<8} {'失败':<12} {'N/A':<8} {'N/A':<12}")
        
    except Exception as e:
        print(f"❌ 基准测试失败: {e}")
        import traceback
        traceback.print_exc()

def run_full_training_test():
    print("\n🔬 运行完整微调测试...")
    try:
        from full_training_test import FullTrainingTester
        tester = FullTrainingTester()
        tester.run_full_test()
    except Exception as e:
        print(f"❌ 完整微调测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="轻量化安全帽检测系统")
    parser.add_argument("--mode", choices=[
        "cmd", "baseline", "lightweight", "full", "evaluate", "detect", 
        "test", "analyze", "full_test", "benchmark"
    ], help="直接运行模式")
    parser.add_argument("--dataset", choices=["subset", "medium", "full"], 
                       default="medium", help="数据集规模")
    parser.add_argument("--data", type=str, help="数据集配置文件路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批次大小")
    parser.add_argument("--source", type=str, help="检测源 (摄像头/视频/图像)")
    parser.add_argument("--model1", type=str, help="对比评估的第一个模型")
    parser.add_argument("--model2", type=str, help="对比评估的第二个模型")
    parser.add_argument("--model", type=str, help="基准测试的模型路径")
    parser.add_argument("--test-modules", action="store_true", help="测试所有模块")
    parser.add_argument("--analyze-architecture", action="store_true", help="分析模型架构")
    parser.add_argument("--train-baseline", action="store_true", help="训练基线模型")
    parser.add_argument("--train-lightweight", action="store_true", help="训练轻量化模型")
    parser.add_argument("--train-full", action="store_true", help="全量数据训练")
    
    args = parser.parse_args()
    
    # 命令行快捷操作
    if args.test_modules:
        test_modules()
        return
    elif args.analyze_architecture:
        analyze_models()
        return
    elif args.train_baseline:
        train_baseline()
        return
    elif args.train_lightweight:
        train_lightweight()
        return
    elif args.train_full:
        train_full_dataset()
        return
    
    # 模式选择
    if args.mode:
        if args.mode == "cmd":
            pass  # 继续到交互式菜单
        elif args.mode == "baseline":
            train_baseline()
        elif args.mode == "lightweight":
            train_lightweight()
        elif args.mode == "full":
            train_full_dataset()
        elif args.mode == "evaluate":
            evaluate_models()
        elif args.mode == "detect":
            real_time_detection()
        elif args.mode == "test":
            test_modules()
        elif args.mode == "analyze":
            analyze_models()
        elif args.mode == "full_test":
            run_full_training_test()
        elif args.mode == "benchmark":
            benchmark_model()
        
        if args.mode != "cmd":
            return
    
    # 交互式菜单
    while True:
        show_menu()
        
        try:
            choice = input("\n请选择操作 (0-9): ").strip()
            
            if choice == "0":
                print("感谢使用轻量化安全帽检测系统！")
                break
            elif choice == "1":
                train_baseline()
            elif choice == "2":
                train_lightweight()
            elif choice == "3":
                train_full_dataset()
            elif choice == "4":
                evaluate_models()
            elif choice == "5":
                real_time_detection()
            elif choice == "6":
                test_modules()
            elif choice == "7":
                analyze_models()
            elif choice == "8":
                run_full_training_test()
            elif choice == "9":
                benchmark_model()
            else:
                print("❌ 无效选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n\n用户中断，退出程序")
            break
        except Exception as e:
            print(f"❌ 执行错误: {e}")
            
        input("\n按回车键继续...")

if __name__ == "__main__":
    main() 
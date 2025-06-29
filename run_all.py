#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LW-YOLOv8 模块逐步测试脚本

这个脚本按照以下顺序执行训练和测试：
1. 训练 baseline YOLOv8s（基准模型）
2. 训练 仅CSP-CTFN 模块版本
3. 训练 仅PSC-Head 模块版本  
4. 训练 仅SIoU Loss 模块版本
5. 训练 完整LW-YOLOv8（三个模块都有）
6. 进行全面模型对比和评估

使用方法:
    # 完整流程
    python run_all.py --epochs 300 --device cuda --batch 16
    
    # 快速测试（少量epochs）
    python run_all.py --quick --epochs 10
    
    # 跳过某些步骤
    python run_all.py --skip-baseline --skip-compare
"""

import argparse
import os
import sys
import subprocess
import yaml
import shutil
from pathlib import Path

def create_model_config(config_name, use_csp_ctfn=False, use_psc_head=False):
    """
    创建不同模块组合的模型配置文件
    
    Args:
        config_name (str): 配置文件名
        use_csp_ctfn (bool): 是否使用CSP-CTFN模块
        use_psc_head (bool): 是否使用PSC-Head模块
    """
    config_path = Path("ultralytics") / "cfg" / "models" / "v8" / config_name
    
    # 基础配置内容
    config_content = f"""# LW-YOLOv8 模块测试配置文件 - {config_name}
# 模块组合: CSP-CTFN={use_csp_ctfn}, PSC-Head={use_psc_head}

nc: 2  # 类别数量 (head=0, helmet=1)
scales: # 模型复合缩放常数
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024] 
  m: [0.67, 0.75, 768]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.25, 512]

# 骨干网络配置
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]     # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]    # 1-P2/4  
  - [-1, 3, C2f, [128, True]]     # 2 - 保持原始C2f
  - [-1, 1, Conv, [256, 3, 2]]    # 3-P3/8"""

    # 添加骨干网络中层配置
    if use_csp_ctfn:
        config_content += """
  - [-1, 6, CSP_CTFN, [256, True]] # 4 - 使用CSP-CTFN"""
    else:
        config_content += """
  - [-1, 6, C2f, [256, True]]     # 4 - 使用原始C2f"""
    
    config_content += """
  - [-1, 1, Conv, [512, 3, 2]]    # 5-P4/16"""
    
    if use_csp_ctfn:
        config_content += """
  - [-1, 6, CSP_CTFN, [512, True]] # 6 - 使用CSP-CTFN"""
    else:
        config_content += """
  - [-1, 6, C2f, [512, True]]     # 6 - 使用原始C2f"""
    
    config_content += """
  - [-1, 1, Conv, [1024, 3, 2]]   # 7-P5/32"""
    
    if use_csp_ctfn:
        config_content += """
  - [-1, 3, CSP_CTFN, [1024, True]] # 8 - 使用CSP-CTFN"""
    else:
        config_content += """
  - [-1, 3, C2f, [1024, True]]    # 8 - 使用原始C2f"""
    
    config_content += """
  - [-1, 1, SPPF, [1024, 5]]      # 9 - 空间金字塔池化

# 检测头配置
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 10
  - [[-1, 6], 1, Concat, [1]]     # 11 - cat backbone P4
  - [-1, 3, C2f, [512]]           # 12

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 13  
  - [[-1, 4], 1, Concat, [1]]     # 14 - cat backbone P3
  - [-1, 3, C2f, [256]]           # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]    # 16
  - [[-1, 12], 1, Concat, [1]]    # 17 - cat head P4
  - [-1, 3, C2f, [512]]           # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]    # 19
  - [[-1, 9], 1, Concat, [1]]     # 20 - cat head P5  
  - [-1, 3, C2f, [1024]]          # 21 (P5/32-large)
"""
    
    # 添加检测头配置
    if use_psc_head:
        config_content += """
  - [[15, 18, 21], 1, PSCDetect, [nc]]  # 22 - 使用PSC-Head参数共享检测头"""
    else:
        config_content += """
  - [[15, 18, 21], 1, Detect, [nc]]     # 22 - 使用原始检测头"""
    
    # 保存配置文件
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    return str(config_path)

def create_training_script(script_name, model_config, use_siou=False, project_name="experiment", device='cuda', epochs=300, batch=24):
    """
    创建训练脚本
    
    Args:
        script_name (str): 脚本文件名
        model_config (str): 模型配置文件路径
        use_siou (bool): 是否使用SIoU损失
        project_name (str): 项目名称
        device (str): 训练设备
        epochs (int): 训练轮数
        batch (int): 批次大小
    """
    # 确保路径使用正斜杠，转换为字符串
    model_config_path = str(Path(model_config)).replace('\\', '/')
    
    script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
{project_name} 训练脚本
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

# 解决OpenMP问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main():
    # 设置模型
    model = YOLO('yolov8s.pt')  # 从预训练权重开始
    
    # 应用自定义配置
    model.model = YOLO('{model_config_path}').model
    
    # 如果使用SIoU损失，修改损失函数
    {f"""
    from ultralytics.utils.loss import v8DetectionSIoULoss
    
    # 创建自定义训练器类
    class SIoUTrainer(model.trainer_class):
        def get_loss(self, batch):
            return v8DetectionSIoULoss(self.model)
    
    # 使用自定义训练器
    model.trainer_class = SIoUTrainer
    """ if use_siou else "# 使用默认CIoU损失"}
    
    # 开始训练
    results = model.train(
        data='datasets/dataset.yaml',
        epochs={epochs},
        batch={batch},
        project='runs/train',
        name='{project_name}',
        amp=True,
        device='{device}',
        workers=8,
        save_period=10,
        patience=50,
        verbose=True
    )
    
    print(f"✅ {project_name} 训练完成!")
    return results

if __name__ == '__main__':
    main()
'''
    
    script_path = Path(script_name)
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 设置执行权限（Windows下忽略）
    try:
        script_path.chmod(0o755)
    except:
        pass
    return script_path

def run_command(cmd, description, timeout=None):
    """运行命令并显示进度"""
    print(f"\n🚀 {description}")
    print(f"执行命令: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, check=True, timeout=timeout)
        print(f"✅ {description} - 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - 失败: {e}")
        return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - 超时")
        return False

def main():
    parser = argparse.ArgumentParser(description='LW-YOLOv8 模块逐步测试脚本')
    parser.add_argument('--epochs', type=int, default=100, help='每个模型的训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='训练设备选择 (cpu 或 cuda)')
    parser.add_argument('--quick', action='store_true', help='快速测试模式（使用更少epochs）')
    parser.add_argument('--skip-baseline', action='store_true', help='跳过baseline训练')
    parser.add_argument('--skip-compare', action='store_true', help='跳过模型对比')
    parser.add_argument('--timeout', type=int, default=7200, help='每个训练的超时时间（秒）')
    parser.add_argument('--only', choices=['baseline', 'csp', 'psc', 'siou', 'full'], help='只训练指定的模型')
    parser.add_argument('--resume', action='store_true', help='尝试恢复之前中断的训练')
    parser.add_argument('--data', type=str, default='datasets/dataset.yaml', help='数据集配置文件路径')
    
    args = parser.parse_args()
    
    # 快速模式调整参数
    if args.quick:
        args.epochs = min(args.epochs, 20)
        print(f"🚀 快速测试模式：使用 {args.epochs} epochs")
    
    print("🎯 LW-YOLOv8 模块逐步测试脚本")
    print("=" * 50)
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch}")
    print(f"训练设备: {args.device}")
    print(f"数据集: {args.data}")
    if args.only:
        print(f"仅训练: {args.only}")
    print(f"跳过baseline: {args.skip_baseline}")
    print(f"跳过对比: {args.skip_compare}")
    print(f"恢复训练: {args.resume}")
    print("=" * 50)
    
    results = {}
    
    # 1. 训练 Baseline YOLOv8s
    if not args.skip_baseline:
        print("\n🔵 第1步: 训练 Baseline YOLOv8s")
        cmd_baseline = [
            sys.executable, '-c',
            f"""
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
results = model.train(
    data='datasets/dataset.yaml',
    epochs={args.epochs},
    batch={args.batch},
    project='runs/train',
    name='baseline-yolov8s',
    amp=True,
    device='{args.device}',
    workers=8,
    save_period=10,
    patience=50,
    verbose=True
)
print('✅ Baseline YOLOv8s 训练完成')
"""
        ]
        results['baseline'] = run_command(cmd_baseline, "Baseline YOLOv8s 训练", args.timeout)
    else:
        print("ℹ️ 跳过 Baseline YOLOv8s 训练")
        results['baseline'] = True
    
    # 2. 训练仅CSP-CTFN模块版本
    print("\n🟡 第2步: 训练仅CSP-CTFN模块版本")
    config_csp = create_model_config("csp-ctfn-only.yaml", use_csp_ctfn=True, use_psc_head=False)
    script_csp = create_training_script("train_csp_only.py", config_csp, use_siou=False, project_name="csp-ctfn-only", device=args.device, epochs=args.epochs, batch=args.batch)
    
    cmd_csp = [sys.executable, str(script_csp)]
    results['csp_only'] = run_command(cmd_csp, "仅CSP-CTFN模块 训练", args.timeout)
    
    # 3. 训练仅PSC-Head模块版本  
    print("\n🟠 第3步: 训练仅PSC-Head模块版本")
    config_psc = create_model_config("psc-head-only.yaml", use_csp_ctfn=False, use_psc_head=True)
    script_psc = create_training_script("train_psc_only.py", config_psc, use_siou=False, project_name="psc-head-only", device=args.device, epochs=args.epochs, batch=args.batch)
    
    cmd_psc = [sys.executable, str(script_psc)]
    results['psc_only'] = run_command(cmd_psc, "仅PSC-Head模块 训练", args.timeout)
    
    # 4. 训练仅SIoU Loss版本
    print("\n🔴 第4步: 训练仅SIoU Loss版本") 
    config_siou = create_model_config("siou-only.yaml", use_csp_ctfn=False, use_psc_head=False)
    script_siou = create_training_script("train_siou_only.py", config_siou, use_siou=True, project_name="siou-only", device=args.device, epochs=args.epochs, batch=args.batch)
    
    cmd_siou = [sys.executable, str(script_siou)]
    results['siou_only'] = run_command(cmd_siou, "仅SIoU Loss 训练", args.timeout)
    
    # 5. 训练完整LW-YOLOv8（三个模块都有）
    print("\n🟢 第5步: 训练完整LW-YOLOv8（所有模块）")
    config_full = create_model_config("lw-yolov8-full.yaml", use_csp_ctfn=True, use_psc_head=True)
    script_full = create_training_script("train_full.py", config_full, use_siou=True, project_name="lw-yolov8-full", device=args.device, epochs=args.epochs, batch=args.batch)
    
    cmd_full = [sys.executable, str(script_full)]
    results['full'] = run_command(cmd_full, "完整LW-YOLOv8 训练", args.timeout)
    
    # 6. 全面模型对比
    if not args.skip_compare:
        print("\n📊 第6步: 全面模型对比")
        
        # 收集所有训练好的模型权重
        model_weights = {}
        weight_paths = [
            ('baseline', Path('runs/train/baseline-yolov8s/weights/best.pt')),
            ('csp_only', Path('runs/train/csp-ctfn-only/weights/best.pt')),
            ('psc_only', Path('runs/train/psc-head-only/weights/best.pt')),
            ('siou_only', Path('runs/train/siou-only/weights/best.pt')),
            ('full', Path('runs/train/lw-yolov8-full/weights/best.pt'))
        ]
        
        for name, path in weight_paths:
            if path.exists():
                model_weights[name] = str(path)
                print(f"✅ 找到模型权重: {name} -> {path}")
            else:
                print(f"⚠️ 未找到模型权重: {name} -> {path}")
        
        # 创建对比脚本
        compare_script = '''
import sys
from pathlib import Path
from ultralytics import YOLO
import torch
import pandas as pd

def get_model_info(model_path, model_name):
    """获取模型信息"""
    try:
        model = YOLO(model_path)
        
        # 获取模型参数
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        # 获取模型大小
        model_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
        
        # 验证模型
        results = model.val(data='datasets/dataset.yaml', verbose=False)
        
        return {
            'Model': model_name,
            'Parameters': total_params,
            'Trainable_Params': trainable_params,
            'Model_Size_MB': model_size,
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'Precision': results.box.mp,
            'Recall': results.box.mr
        }
    except Exception as e:
        print(f"错误评估模型 {model_name}: {e}")
        return None

def main():
    model_weights = MODEL_WEIGHTS_PLACEHOLDER
    
    print("🔍 开始模型对比分析...")
    results = []
    
    for name, path in model_weights.items():
        print(f"\n评估模型: {name}")
        info = get_model_info(path, name)
        if info:
            results.append(info)
    
    # 创建对比表格
    if results:
        df = pd.DataFrame(results)
        
        # 保存详细结果
        output_dir = Path('runs/compare')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_dir / 'model_comparison.csv', index=False)
        
        print("\n📊 模型对比结果:")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)
        
        # 性能排名
        if 'mAP50-95' in df.columns:
            df_sorted = df.sort_values('mAP50-95', ascending=False)
            print("\n🏆 mAP50-95 排名:")
            for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
                print(f"{i}. {row['Model']}: {row['mAP50-95']:.4f}")
        
        # 参数量对比
        if 'Parameters' in df.columns:
            df_params = df.sort_values('Parameters')
            print("\n📏 参数量排名（从少到多）:")
            for i, (_, row) in enumerate(df_params.iterrows(), 1):
                print(f"{i}. {row['Model']}: {row['Parameters']:,}")
        
        print(f"\n📁 详细结果已保存到: {output_dir / 'model_comparison.csv'}")
    else:
        print("❌ 没有成功评估的模型")

if __name__ == '__main__':
    main()
'''.replace('MODEL_WEIGHTS_PLACEHOLDER', str(model_weights))
        
        compare_script_path = Path('compare_all_models.py')
        with open(compare_script_path, 'w', encoding='utf-8') as f:
            f.write(compare_script)
        
        cmd_compare = [sys.executable, str(compare_script_path)]
        results['compare'] = run_command(cmd_compare, "全面模型对比", args.timeout//2)
    else:
        print("ℹ️ 跳过模型对比")
        results['compare'] = True
    
    # 总结报告
    print("\n" + "=" * 80)
    print("🎉 模块逐步测试完成！")
    print("=" * 80)
    
    # 显示结果统计
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    print(f"成功步骤: {success_count}/{total_count}")
    print("\n📋 各步骤结果:")
    
    step_names = {
        'baseline': '1. Baseline YOLOv8s',
        'csp_only': '2. 仅CSP-CTFN模块',
        'psc_only': '3. 仅PSC-Head模块',
        'siou_only': '4. 仅SIoU Loss',
        'full': '5. 完整LW-YOLOv8',
        'compare': '6. 模型对比'
    }
    
    for key, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"   {step_names.get(key, key)}: {status}")
    
    print("\n📁 主要输出文件:")
    
    # 检查生成的文件
    weight_dirs = [
        ('Baseline YOLOv8s', Path('runs/train/baseline-yolov8s/weights/best.pt')),
        ('仅CSP-CTFN', Path('runs/train/csp-ctfn-only/weights/best.pt')),
        ('仅PSC-Head', Path('runs/train/psc-head-only/weights/best.pt')),
        ('仅SIoU Loss', Path('runs/train/siou-only/weights/best.pt')),
        ('完整LW-YOLOv8', Path('runs/train/lw-yolov8-full/weights/best.pt'))
    ]
    
    for name, path in weight_dirs:
        if path.exists():
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} (未找到)")
    
    compare_file = Path('runs/compare/model_comparison.csv')
    if compare_file.exists():
        print(f"✅ 对比报告: {compare_file}")
        
        # 显示最佳模型推荐
        try:
            import pandas as pd
            df = pd.read_csv(compare_file)
            if not df.empty and 'mAP50-95' in df.columns:
                best_model = df.loc[df['mAP50-95'].idxmax()]
                print(f"\n🏆 推荐最佳模型: {best_model['Model']}")
                print(f"   📊 mAP50-95: {best_model['mAP50-95']:.4f}")
                if 'Parameters' in df.columns:
                    print(f"   🔧 参数量: {best_model['Parameters']:,}")
                if 'Model_Size_MB' in df.columns:
                    print(f"   💾 模型大小: {best_model['Model_Size_MB']:.2f} MB")
                
                # 模块效果分析
                print(f"\n📈 模块效果分析:")
                baseline_map = df[df['Model'] == 'baseline']['mAP50-95'].iloc[0] if 'baseline' in df['Model'].values else None
                if baseline_map is not None:
                    for _, row in df.iterrows():
                        if row['Model'] != 'baseline':
                            improvement = row['mAP50-95'] - baseline_map
                            if improvement > 0:
                                print(f"   ✅ {row['Model']}: +{improvement:.4f} (+{improvement/baseline_map*100:.2f}%)")
                            else:
                                print(f"   ❌ {row['Model']}: {improvement:.4f} ({improvement/baseline_map*100:.2f}%)")
        except Exception as e:
            print(f"⚠️ 无法读取对比结果: {e}")
    
    print(f"\n🎯 训练完成！所有模型权重保存在 runs/train/ 目录下")
    
    # 清理临时文件
    temp_files = [
        Path('train_csp_only.py'),
        Path('train_psc_only.py'), 
        Path('train_siou_only.py'),
        Path('train_full.py'),
        Path('compare_all_models.py')
    ]
    
    for temp_file in temp_files:
        if temp_file.exists():
            temp_file.unlink()

if __name__ == '__main__':
    main() 
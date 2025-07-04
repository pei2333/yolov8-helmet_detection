import argparse
import cv2
from ultralytics import YOLO
from pathlib import Path

# 类别信息
CLASS_NAMES = {0: "head", 1: "helmet"}
CLASS_COLORS = {0: (0, 0, 255), 1: (0, 255, 0)}

def draw_boxes(image, boxes):
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0].cpu().numpy()
        class_id = int(box.cls[0].cpu().numpy())
        color = CLASS_COLORS[class_id]
        label = f"{CLASS_NAMES[class_id]}: {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='模型权重路径')
    parser.add_argument('--source', type=str, required=True, help='图片或文件夹')
    parser.add_argument('--output', type=str, default='runs/detect', help='输出目录')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--show', action='store_true', help='是否弹窗显示')
    args = parser.parse_args()

    model = YOLO(args.weights)
    src = Path(args.source)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if src.is_file():
        results = model(str(src), conf=args.conf)
        img = cv2.imread(str(src))
        img = draw_boxes(img, results[0].boxes)
        out_path = out_dir / f"result_{src.name}"
        cv2.imwrite(str(out_path), img)
        if args.show:
            cv2.imshow('result', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print(f"推理完成，结果保存在: {out_path}")
    else:
        images = list(src.glob('*.jpg')) + list(src.glob('*.png'))
        for img_path in images:
            results = model(str(img_path), conf=args.conf)
            img = cv2.imread(str(img_path))
            img = draw_boxes(img, results[0].boxes)
            out_path = out_dir / f"result_{img_path.name}"
            cv2.imwrite(str(out_path), img)
        print(f"批量推理完成，结果保存在: {out_dir}")

if __name__ == '__main__':
    main()
import os
import yaml
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError('ultralytics package is required to run this script')


def load_dataset_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def yolo_label_to_xyxy(lbl_line, img_w, img_h):
    # YOLO format: class x_center y_center w h (normalized)
    parts = [float(x) for x in lbl_line.strip().split()]
    if len(parts) < 5:
        return None
    cls = int(parts[0])
    xc, yc, w, h = parts[1:5]
    x1 = (xc - w/2.0) * img_w
    y1 = (yc - h/2.0) * img_h
    x2 = (xc + w/2.0) * img_w
    y2 = (yc + h/2.0) * img_h
    return cls, [x1, y1, x2, y2]


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    return interArea / (boxAArea + boxBArea - interArea)


def main():
    # Paths
    dataset_yaml = os.path.join('dataset', 'car_dataset_reg.yaml')
    model_path = os.path.join('training', 'outputs', 'car_dataset_reg_yolov8n_416', 'weights', 'best.pt')
    out_dir = os.path.join('training', 'outputs', 'evaluation_20260813_023618')
    os.makedirs(out_dir, exist_ok=True)

    from pathlib import Path
    cfg = load_dataset_yaml(dataset_yaml)
    root = cfg.get('path')
    test_rel = cfg.get('test')
    # Resolve absolute test images and labels directories robustly
    if test_rel:
        test_images_dir = os.path.normpath(os.path.join(root, *test_rel.split('/')))
    else:
        test_images_dir = os.path.normpath(os.path.join(root, 'test', 'images'))
    # labels directory is sibling 'labels' directory to 'images'
    labels_dir = str(Path(test_images_dir).parent / 'labels')

    if not os.path.isdir(test_images_dir):
        raise FileNotFoundError(f'Test images dir not found: {test_images_dir}')
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f'Test labels dir not found: {labels_dir}')

    model = YOLO(model_path)

    # Collect image files
    imgs = sorted([f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    class_stats = {}
    classes = set()
    total_images = len(imgs)

    # Per-image matching
    for img_name in tqdm(imgs, desc='Predicting'):
        img_path = os.path.join(test_images_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # Load GT boxes
        lbl_name = os.path.splitext(img_name)[0] + '.txt'
        lbl_path = os.path.join(labels_dir, lbl_name)
        gt_boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parsed = yolo_label_to_xyxy(line, w, h)
                    if parsed is None:
                        continue
                    cls, box = parsed
                    gt_boxes.append({'cls': cls, 'box': box, 'matched': False})
                    classes.add(cls)

        # Run prediction
        results = model.predict(source=img_path, imgsz=416, conf=0.25, iou=0.45, device='cpu')
        if len(results) == 0:
            preds = []
        else:
            r = results[0]
            preds = []
            if hasattr(r, 'boxes') and r.boxes is not None:
                for b in r.boxes:
                    xyxy_arr = b.xyxy.cpu().numpy()
                    # ensure a flat list of four floats [x1,y1,x2,y2]
                    xyxy = [float(x) for x in xyxy_arr.reshape(-1).tolist()]
                    # use .item() to extract scalar tensors safely
                    try:
                        conf = float(b.conf.cpu().item())
                    except Exception:
                        conf = float(b.conf)
                    try:
                        cls = int(b.cls.cpu().item())
                    except Exception:
                        cls = int(b.cls)
                    preds.append({'cls': cls, 'box': xyxy, 'conf': conf, 'matched': False})

        # sort preds by confidence
        preds = sorted(preds, key=lambda x: x['conf'], reverse=True)

        # Matching
        for p in preds:
            best_iou = 0.0
            best_gt = None
            for gt in gt_boxes:
                if gt['matched']:
                    continue
                i = iou(p['box'], gt['box'])
                if i > best_iou:
                    best_iou = i
                    best_gt = gt
            if best_gt is not None and best_iou >= 0.5:
                # matched
                p['matched'] = True
                best_gt['matched'] = True
                classes.add(p['cls'])
                class_stats.setdefault(p['cls'], {'TP': 0, 'FP': 0, 'FN': 0})
                class_stats[p['cls']]['TP'] += 1
            else:
                # false positive
                classes.add(p['cls'])
                class_stats.setdefault(p['cls'], {'TP': 0, 'FP': 0, 'FN': 0})
                class_stats[p['cls']]['FP'] += 1

        # Remaining unmatched GTs are false negatives
        for gt in gt_boxes:
            if not gt['matched']:
                classes.add(gt['cls'])
                class_stats.setdefault(gt['cls'], {'TP': 0, 'FP': 0, 'FN': 0})
                class_stats[gt['cls']]['FN'] += 1

    # Prepare per-class metrics
    rows = []
    cls_list = sorted(list(classes))
    for c in cls_list:
        stats = class_stats.get(c, {'TP': 0, 'FP': 0, 'FN': 0})
        TP = stats['TP']
        FP = stats['FP']
        FN = stats['FN']
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({'class': c, 'TP': TP, 'FP': FP, 'FN': FN, 'Precision': precision, 'Recall': recall, 'F1': f1})

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, 'per_class_metrics.csv')
    df.to_csv(out_csv, index=False)

    # Confusion matrix for single-label detection: map to [TP, FP; FN, TN] not directly meaningful
    # We'll build a simple 2x2 matrix: rows=ground-truth (pos, neg), cols=pred (pos, neg)
    # pos = license_plate present per-image
    # Compute per-image presence
    per_image_gt_pos = 0
    per_image_pred_pos = 0
    tp_images = 0
    fp_images = 0
    fn_images = 0
    for img_name in imgs:
        img_path = os.path.join(test_images_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl_name = os.path.splitext(img_name)[0] + '.txt'
        lbl_path = os.path.join(labels_dir, lbl_name)
        gt_cnt = 0
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parsed = yolo_label_to_xyxy(line, w, h)
                    if parsed:
                        gt_cnt += 1

        results = model.predict(source=img_path, imgsz=416, conf=0.25, iou=0.45, device='cpu')
        pred_cnt = 0
        if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            pred_cnt = len(results[0].boxes)

        gt_pos = 1 if gt_cnt > 0 else 0
        pred_pos = 1 if pred_cnt > 0 else 0
        per_image_gt_pos += gt_pos
        per_image_pred_pos += pred_pos
        if gt_pos == 1 and pred_pos == 1:
            tp_images += 1
        elif gt_pos == 0 and pred_pos == 1:
            fp_images += 1
        elif gt_pos == 1 and pred_pos == 0:
            fn_images += 1

    tn_images = total_images - (tp_images + fp_images + fn_images)

    cm = np.array([[tp_images, fn_images], [fp_images, tn_images]])

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred: Pos', 'Pred: Neg'])
    ax.set_yticklabels(['GT: Pos', 'GT: Neg'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i,j]), ha='center', va='center', color='black')
    plt.title('Per-image Confusion Matrix')
    plt.tight_layout()
    cm_path = os.path.join(out_dir, 'confusion_matrix.png')
    fig.savefig(cm_path)

    print('Saved per-class metrics to', out_csv)
    print('Saved confusion matrix to', cm_path)


if __name__ == '__main__':
    main()

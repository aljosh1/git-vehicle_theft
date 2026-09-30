import os
import yaml
import cv2
from ultralytics import YOLO

ds = yaml.safe_load(open('dataset/car_dataset_reg.yaml'))
root = ds['path']
test_images_dir = os.path.join(root, ds['test'])
labels_dir = test_images_dir.replace(os.sep + 'images', os.sep + 'labels')

imgs = sorted([f for f in os.listdir(test_images_dir) if f.lower().endswith('.jpg')])
img_name = imgs[0]
img_path = os.path.join(test_images_dir, img_name)
lbl_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')
print('Image:', img_path)
print('Label file:', lbl_path)
with open(lbl_path,'r') as f:
    lines = f.read().splitlines()
    print('Label lines:', lines)

img = cv2.imread(img_path)
print('Image shape:', img.shape)
def yolo_label_to_xyxy(lbl_line, img_w, img_h):
    parts = [float(x) for x in lbl_line.strip().split()]
    cls = int(parts[0])
    xc, yc, w, h = parts[1:5]
    x1 = (xc - w/2.0) * img_w
    y1 = (yc - h/2.0) * img_h
    x2 = (xc + w/2.0) * img_w
    y2 = (yc + h/2.0) * img_h
    return cls, [x1,y1,x2,y2]

for line in lines:
    print('Parsed GT:', yolo_label_to_xyxy(line, img.shape[1], img.shape[0]))

model = YOLO(os.path.join('training','outputs','car_dataset_reg_yolov8n_416','weights','best.pt'))
res = model.predict(source=img_path, imgsz=416, conf=0.25, iou=0.45, device='cpu')
print('Num results:', len(res))
if len(res) > 0 and hasattr(res[0],'boxes') and res[0].boxes is not None:
    for b in res[0].boxes:
        print('Pred xyxy:', b.xyxy.cpu().numpy().reshape(-1).tolist())
        try:
            print('conf,class:', b.conf.cpu().item(), b.cls.cpu().item())
        except Exception:
            print('conf,class raw:', b.conf, b.cls)

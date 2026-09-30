import cv2
from backend.detection.plate_detector import PlateDetector
from backend.recognition.plate_ocr import read_plate
from backend.config import settings

settings.ensure_directories()
img_path = settings.data_dir / "uploads" / "roboflow_sample.jpg"
print("Image:", img_path)
img = cv2.imread(str(img_path))
if img is None:
    raise SystemExit("failed to load image")

pd = PlateDetector()
dets = pd.detect(img)
print(f"Found {len(dets)} plate candidates")
for i, d in enumerate(dets[:3], 1):
    print(f"Candidate {i}: bbox=({d.x1},{d.y1},{d.x2},{d.y2}) conf={d.confidence} meta={d.meta}")
    crop = img[d.y1:d.y2, d.x1:d.x2]
    text, conf = read_plate(crop)
    print(f"  OCR -> {text} (conf={conf})")

# save first candidate annotated
if dets:
    d = dets[0]
    a = img.copy()
    cv2.rectangle(a, (d.x1, d.y1), (d.x2, d.y2), (0,255,0), 2)
    out = settings.data_dir / "analyses" / "smoke_plate_out.jpg"
    cv2.imwrite(str(out), a)
    print("Wrote annotated image to", out)

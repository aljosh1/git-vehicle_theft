from backend.core.pipeline import Pipeline
from backend.config import settings
import cv2

settings.ensure_directories()

img_path = settings.data_dir / "uploads" / "roboflow_sample.jpg"
print("Using image:", img_path)
img = cv2.imread(str(img_path))
if img is None:
    raise SystemExit("Failed to read image")

p = Pipeline("smoke", source=img_path)
annotated, result = p.analyse_image(img, dispatch_alerts=False)

print(f"Processing time: {result.processing_ms:.1f} ms, FPS approx {result.fps:.2f}")
print(f"Vehicles: {len(result.vehicles)}, Plates detected: {len(result.plates)}")
print(f"Plate text: {result.plate_text}, confidence: {result.plate_confidence}")

# Save annotated image for quick inspection
out = settings.data_dir / "analyses" / "smoke_annotated.jpg"
cv2.imwrite(str(out), annotated)
print("Annotated image written to:", out)

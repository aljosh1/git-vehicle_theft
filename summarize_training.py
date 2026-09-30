import pandas as pd
from pathlib import Path
import json

# Training runs to check
runs = {
    "plate_detector_2000_416_5epochs": "training/outputs/plate_detector_2000_416_5epochs/results.csv",
    "plate_detector_640_100e": "training/outputs/plate_detector_640_100e/results.csv",
}

summary = {}

for run_name, csv_path in runs.items():
    p = Path(csv_path)
    if p.exists():
        df = pd.read_csv(p)
        # Get the last row (best model)
        last = df.iloc[-1]
        summary[run_name] = {
            "Total Epochs": int(last['epoch']) + 1,
            "Precision": f"{float(last['metrics/precision(B)']):.4f}",
            "Recall": f"{float(last['metrics/recall(B)']):.4f}",
            "mAP50": f"{float(last['metrics/mAP50(B)']):.4f}",
            "mAP50-95": f"{float(last['metrics/mAP50-95(B)']):.4f}",
            "Val Box Loss": f"{float(last['val/box_loss']):.4f}",
            "Train Box Loss": f"{float(last['train/box_loss']):.4f}",
        }
        
        # Find best mAP50 epoch
        best_idx = df['metrics/mAP50(B)'].idxmax()
        best_row = df.iloc[best_idx]
        summary[run_name]["Best mAP50 Epoch"] = int(best_row['epoch']) + 1
        summary[run_name]["Best mAP50 Value"] = f"{float(best_row['metrics/mAP50(B)']):.4f}"
        summary[run_name]["Best mAP50 Precision"] = f"{float(best_row['metrics/precision(B)']):.4f}"
        summary[run_name]["Best mAP50 Recall"] = f"{float(best_row['metrics/recall(B)']):.4f}"

print("\n" + "="*80)
print("TRAINING SUMMARY - ALL METRICS")
print("="*80)

for run_name, metrics in summary.items():
    print(f"\n{run_name}:")
    print("-" * 80)
    for key, value in metrics.items():
        print(f"  {key:<30}: {value}")

# Also print all epochs for detailed comparison
print("\n" + "="*80)
print("DETAILED EPOCH-BY-EPOCH METRICS")
print("="*80)

for run_name, csv_path in runs.items():
    p = Path(csv_path)
    if p.exists():
        df = pd.read_csv(p)
        print(f"\n{run_name}:")
        print(df[['epoch', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']].to_string(index=False))

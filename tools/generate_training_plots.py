import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'training' / 'outputs'

runs = [d for d in OUT.iterdir() if d.is_dir()]

for run in runs:
    csv = run / 'results.csv'
    if csv.exists():
        try:
            df = pd.read_csv(csv)
        except Exception as e:
            print(f"Failed to read {csv}: {e}")
            continue

        epochs = df['epoch'] if 'epoch' in df.columns else range(1, len(df)+1)

        # Metrics plot
        fig, ax = plt.subplots(2, 1, figsize=(8, 8), tight_layout=True)
        for col, lbl in [
            ('metrics/precision(B)', 'Precision'),
            ('metrics/recall(B)', 'Recall'),
            ('metrics/mAP50(B)', 'mAP@0.5'),
            ('metrics/mAP50-95(B)', 'mAP@0.5:0.95'),
        ]:
            if col in df.columns:
                ax[0].plot(epochs, df[col], label=lbl)
        ax[0].set_title('Validation metrics')
        ax[0].set_xlabel('Epoch')
        ax[0].legend()
        ax[0].grid(True)

        # Losses plot
        loss_cols = [c for c in df.columns if 'loss' in c]
        for c in loss_cols:
            ax[1].plot(epochs, df[c], label=c)
        ax[1].set_title('Losses')
        ax[1].set_xlabel('Epoch')
        ax[1].legend(fontsize='small')
        ax[1].grid(True)

        out_file = run / 'training_metrics.png'
        fig.suptitle(run.name)
        fig.savefig(out_file)
        plt.close(fig)
        print('Wrote', out_file)

    # evaluation summary plots
    summary = run / 'summary.txt'
    if summary.exists():
        txt = summary.read_text()
        # parse simple key: value lines
        vals = {}
        for line in txt.splitlines():
            if ':' in line:
                k, v = line.split(':', 1)
                k = k.strip()
                v = v.strip()
                try:
                    # strip trailing text like 'ms' or 'FPS'
                    num = float(v.split()[0])
                    vals[k] = num
                except Exception:
                    continue
        if vals:
            fig, ax = plt.subplots(1, 1, figsize=(6,4))
            keys = ['Precision','Recall','mAP@0.5','mAP@0.5:0.95']
            bars = [vals.get(k, None) for k in keys]
            ax.bar([k for k,b in zip(keys,bars) if b is not None], [b for b in bars if b is not None])
            ax.set_ylim(0,1)
            ax.set_title(f'Evaluation metrics: {run.name}')
            out_file = run / 'evaluation_metrics.png'
            fig.savefig(out_file)
            plt.close(fig)
            print('Wrote', out_file)

print('Done')

from __future__ import annotations

from pathlib import Path

from common import write_csv
from performance import evaluate_model


def run_generalization(models, dataset_root, out_csv, batch_size=16, scales=(1.0,)):
    rows = []
    for scale in scales:
        for model_info in models:
            rows.extend(
                evaluate_model(
                    model_info,
                    dataset_root=dataset_root,
                    batch_size=batch_size,
                    downsample=1,
                    resize_scale=float(scale),
                    restore_size=False,
                )
            )
    write_csv(Path(out_csv), rows)
    return rows

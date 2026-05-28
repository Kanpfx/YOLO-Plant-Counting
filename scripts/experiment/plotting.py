"""Generate paper-style figures from experiment CSV files."""

import csv
from pathlib import Path
import matplotlib.pyplot as plt

import experiment_config as exp_cfg


MODEL_STYLES = {
    key: {
        "label": info["name"],
        "color": info["color"],
    }
    for key, info in exp_cfg.MODELS.items()
}


def read_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def model_label(model_key):
    return MODEL_STYLES.get(model_key, {}).get("label", model_key)


def model_color(model_key):
    return MODEL_STYLES.get(model_key, {}).get("color", "#666666")


def select_rows(rows, models):
    order = {model["dir_name"]: i for i, model in enumerate(models)}
    rows = [row for row in rows if row["model"] in order]
    return sorted(
        rows,
        key=lambda row: (
            order[row["model"]],
            row.get("class", ""),
            as_float(row.get("resize_scale", row.get("downsample", 0))),
        ),
    )


def ordered_model_keys(models):
    return [model["dir_name"] for model in models]


def save_fig(fig, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def set_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "savefig.facecolor": "white",
        }
    )


def add_panel_label(ax, label):
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def metric_axis_limits(metric):
    if metric in {"precision", "recall", "f1"}:
        return 0.0, 1.02
    if metric == "mre":
        return 0.0, None
    return None, None


def focused_axis_limits(values, metric, pad_ratio=0.12):
    values = [value for value in values if value is not None]
    if not values:
        return metric_axis_limits(metric)
    vmin, vmax = min(values), max(values)
    if abs(vmax - vmin) < 1e-9:
        pad = max(abs(vmax) * 0.02, 0.01)
    else:
        pad = (vmax - vmin) * pad_ratio
    ymin, ymax = vmin - pad, vmax + pad
    if metric in {"precision", "recall", "f1"}:
        ymin = 0.3
        ymax = min(1.0, ymax)
        if ymax - ymin < 0.05:
            ymax = min(1.0, ymin + 0.05)
    elif metric == "mre":
        ymin = max(0.0, ymin)
        ymax = min(1.0, ymax)
        if ymax - ymin < 0.05:
            center = (ymin + ymax) * 0.5
            ymin = max(0.0, center - 0.025)
            ymax = min(1.0, center + 0.025)
    return ymin, ymax


def plot_metric_bars(rows, models, metrics, titles, ylabels, out_path, figsize=(9.5, 6.0)):
    set_style()
    model_keys = ordered_model_keys(models)
    by_model = {row["model"]: row for row in rows}
    labels = [model_label(key) for key in model_keys]
    colors = [model_color(key) for key in model_keys]
    x = list(range(len(model_keys)))

    ncols = 2
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for idx, (ax, metric, title, ylabel) in enumerate(zip(axes, metrics, titles, ylabels)):
        values = [as_float(by_model.get(key, {}).get(metric, 0.0)) for key in model_keys]
        bars = ax.bar(x, values, color=colors, edgecolor="#333333", linewidth=0.45)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ymin, ymax = metric_axis_limits(metric)
        if ymin is not None or ymax is not None:
            ax.set_ylim(ymin, ymax)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        add_panel_label(ax, chr(ord("a") + idx))
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}" if value < 10 else f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    for ax in axes[len(metrics):]:
        ax.axis("off")

    save_fig(fig, out_path)


def plot_performance(csv_path, out_path, models):
    rows = [row for row in select_rows(read_rows(csv_path), models) if row["class"] == "all"]
    metrics = ("precision", "recall", "f1", "mre")
    titles = ("Precision", "Recall", "F1-score", "Mean Relative Error")
    ylabels = ("Precision", "Recall", "F1-score", "MRE")
    plot_metric_bars(rows, models, metrics, titles, ylabels, out_path)


def plot_generalization(csv_path, out_path, models):
    set_style()
    rows = [row for row in select_rows(read_rows(csv_path), models) if row["class"] == "all"]
    metrics = ("precision", "recall", "f1", "mre")
    titles = ("Precision", "Recall", "F1-score", "Mean Relative Error")
    ylabels = ("Precision", "Recall", "F1-score", "MRE")

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.0), sharex=True)
    axes = axes.ravel()
    model_keys = ordered_model_keys(models)

    for idx, (ax, metric, title, ylabel) in enumerate(zip(axes, metrics, titles, ylabels)):
        metric_values = []
        for key in model_keys:
            group = sorted(
                (row for row in rows if row["model"] == key),
                key=lambda row: as_float(row.get("resize_scale", row.get("downsample"))),
            )
            xs = [as_float(row.get("resize_scale", row.get("downsample"))) for row in group]
            ys = [as_float(row[metric]) for row in group]
            metric_values.extend(ys)
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=4,
                linewidth=1.6,
                color=model_color(key),
                label=model_label(key),
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*focused_axis_limits(metric_values, metric))
        ax.grid(axis="both")
        add_panel_label(ax, chr(ord("a") + idx))

    ticks = sorted({as_float(row.get("resize_scale", row.get("downsample"))) for row in rows})
    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{tick:g}x" for tick in ticks])
    axes[2].set_xlabel("Resize scale")
    axes[3].set_xlabel("Resize scale")
    axes[0].legend(loc="best", ncol=1)

    save_fig(fig, out_path)


def plot_physical(csv_path, out_path, models):
    rows = select_rows(read_rows(csv_path), models)
    metrics = ("params_m", "macs_g", "gflops", "model_size_mb")
    titles = ("Parameters", "MACs", "FLOPs", "Model Size")
    ylabels = ("Params / M", "MACs / G", "GFLOPs", "Model Size / MB")
    plot_metric_bars(rows, models, metrics, titles, ylabels, out_path)

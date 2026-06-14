import os
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

QUALITY_METRICS  = ['precision', 'recall', 'f1', 'correctness', 'faithfulness']
TIME_METRICS     = ['retrieve_time_sec', 'generate_time_sec']
ABSTENTION_FIELD = 'true_abstention'


def _slugify(name: str) -> str:
    return name.replace(" ", "_").replace(".", "_").replace("/", "_")


def _plot_quality_bar(experiment_name: str, mean_values: pd.Series, output_dir: str) -> str | None:
    cols = [m for m in QUALITY_METRICS if m in mean_values.index]
    if not cols:
        return None

    values = mean_values[cols].values
    fig, ax = plt.subplots(figsize=(8, 4))

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
    bars = ax.bar(cols, values, color=colors[:len(cols)], width=0.5, zorder=3)

    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.set_ylabel("Score")
    ax.set_title(f"Quality Metrics - {experiment_name}", fontsize=12, pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.4f}",
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{_slugify(experiment_name)}_quality.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_time_bar(experiment_name: str, mean_values: pd.Series, output_dir: str) -> str | None:
    cols = [m for m in TIME_METRICS if m in mean_values.index]
    if not cols:
        return None

    values = mean_values[cols].values
    fig, ax = plt.subplots(figsize=(7, 3))

    colors = ['#4C72B0', '#DD8452']
    bars = ax.barh(cols, values, color=colors[:len(cols)], height=0.4, zorder=3)

    ax.set_xlabel("Seconds")
    ax.set_title(f"Timing Metrics - {experiment_name}", fontsize=12, pad=10)
    ax.grid(axis='x', linestyle='--', alpha=0.6, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(
            val + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}s",
            va='center',
            fontsize=9
        )

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{_slugify(experiment_name)}_timing.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_comparison(all_data: dict[str, pd.Series], output_dir: str) -> str | None:
    available_metrics = [
        m for m in QUALITY_METRICS
        if any(m in v.index for v in all_data.values())
    ]

    if not available_metrics or len(all_data) < 2:
        return None

    exp_names = list(all_data.keys())
    n_exp = len(exp_names)
    n_metrics = len(available_metrics)
    x = np.arange(n_metrics)
    width = 0.8 / n_exp

    fig, ax = plt.subplots(figsize=(max(9, n_metrics * 1.5), 5))
    cmap = plt.get_cmap('tab10')

    for i, (exp, means) in enumerate(all_data.items()):
        vals = [means.get(m, 0) for m in available_metrics]
        offset = (i - n_exp / 2 + 0.5) * width

        bars = ax.bar(
            x + offset,
            vals,
            width=width * 0.9,
            label=exp,
            color=cmap(i),
            zorder=3
        )

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.015,
                f"{val:.2f}",
                ha='center',
                va='bottom',
                fontsize=7
            )

    ax.set_xticks(x)
    ax.set_xticklabels(available_metrics)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title("Quality Metrics - All Experiments Comparison", fontsize=13, pad=12)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "comparison_quality.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def calculate_average_metrics(directory_path: str, output_file: str = "experiment_summary.md"):
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {directory_path}")
        return

    all_metrics = {}

    report_lines = []
    report_lines.append("# Experiment Results Summary\n\n")

    print(f"Found {len(csv_files)} CSV file(s). Starting analysis...\n")

    for file_path in sorted(csv_files):
        file_name = os.path.basename(file_path)
        experiment_name = os.path.splitext(file_name)[0]

        report_lines.append(f"## {file_name}\n\n")

        try:
            df = pd.read_csv(file_path)

            all_numeric = QUALITY_METRICS + TIME_METRICS

            for m in all_numeric:
                if m in df.columns:
                    df[m] = pd.to_numeric(df[m], errors='coerce')

            cols_to_avg = [m for m in all_numeric if m in df.columns]
            mean_values = df[cols_to_avg].mean()

            all_metrics[experiment_name] = mean_values

            report_lines.append("| Metric | Value |\n")
            report_lines.append("|--------|-------|\n")

            for k, v in mean_values.items():
                report_lines.append(f"| {k} | {v:.4f} |\n")

            report_lines.append("\n")

            if ABSTENTION_FIELD in df.columns:
                df[ABSTENTION_FIELD] = pd.to_numeric(df[ABSTENTION_FIELD], errors='coerce')
                total = len(df)
                total_correct = df[ABSTENTION_FIELD].fillna(0).sum()
                score = total_correct / total

                report_lines.append(f"- true_abstention: {score:.4f}\n")
                report_lines.append(f"- correct / total: {total_correct:.0f} / {total}\n\n")

            quality_path = _plot_quality_bar(experiment_name, mean_values, directory_path)
            if quality_path:
                report_lines.append(f"![Quality]({os.path.basename(quality_path)})\n\n")
                print(f"Saved: {quality_path}")

            time_path = _plot_time_bar(experiment_name, mean_values, directory_path)
            if time_path:
                report_lines.append(f"![Timing]({os.path.basename(time_path)})\n\n")
                print(f"Saved: {time_path}")

            report_lines.append("---\n\n")

        except Exception as e:
            print(f"[ERROR] {file_name}: {e}")
            report_lines.append(f"Error processing {file_name}: {e}\n\n---\n\n")

    comparison_path = _plot_comparison(all_metrics, directory_path)

    if comparison_path:
        report_lines.insert(
            1,
            f"![Comparison]({os.path.basename(comparison_path)})\n\n"
        )
        print(f"Saved comparison: {comparison_path}")

    output_path = os.path.join(directory_path, output_file)

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    calculate_average_metrics("./calculate/v1")
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

QUALITY_METRICS  = ['precision', 'recall', 'f1', 'correctness', 'faithfulness']
TIME_METRICS     = ['retrieve_time_sec', 'generate_time_sec']
ABSTENTION_FIELD = 'true_abstention'


def _slugify(name: str) -> str:
    return name.replace(" ", "_").replace(".", "_").replace("/", "_")


def _plot_global_comparison(all_data: dict[str, pd.Series], output_dir: str) -> str | None:
    """
    One unified chart:
    - Quality metrics
    - True abstention
    """

    metrics = QUALITY_METRICS + [ABSTENTION_FIELD]

    available_metrics = [
        m for m in metrics
        if any(m in v.index for v in all_data.values())
    ]

    if len(all_data) < 2 or not available_metrics:
        return None

    exp_names = list(all_data.keys())
    n_exp = len(exp_names)
    n_metrics = len(available_metrics)

    x = np.arange(n_metrics)
    width = 0.8 / n_exp

    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 1.5), 5))
    cmap = plt.get_cmap("tab10")

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
                val + 0.01,
                f"{val:.2f}",
                ha='center',
                va='bottom',
                fontsize=7
            )

    ax.set_xticks(x)
    ax.set_xticklabels(available_metrics)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score")
    ax.set_title("RAG Evaluation Comparison (Quality + Abstention)")
    ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()

    out_path = os.path.join(output_dir, "comparison_all_metrics.png")
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

            all_cols = QUALITY_METRICS + TIME_METRICS + [ABSTENTION_FIELD]

            for m in all_cols:
                if m in df.columns:
                    df[m] = pd.to_numeric(df[m], errors='coerce')

            cols_to_avg = [m for m in all_cols if m in df.columns]
            mean_values = df[cols_to_avg].mean()

            all_metrics[experiment_name] = mean_values

            # TABLE ONLY (clean)
            report_lines.append("| Metric | Value |\n")
            report_lines.append("|--------|-------|\n")

            for k, v in mean_values.items():
                report_lines.append(f"| {k} | {v:.4f} |\n")

            report_lines.append("\n")

            # abstention summary (kept simple)
            if ABSTENTION_FIELD in df.columns:
                total = len(df)
                score = df[ABSTENTION_FIELD].fillna(0).sum() / total

                report_lines.append(f"true_abstention: {score:.4f}\n\n")

            report_lines.append("---\n\n")

        except Exception as e:
            print(f"[ERROR] {file_name}: {e}")
            report_lines.append(f"Error: {file_name} -> {e}\n\n---\n\n")

    # ONE GLOBAL COMPARISON CHART ONLY
    comparison_path = _plot_global_comparison(all_metrics, directory_path)

    if comparison_path:
        report_lines.insert(
            1,
            f"Comparison chart: {os.path.basename(comparison_path)}\n\n"
        )
        print(f"Saved comparison chart: {comparison_path}")

    output_path = os.path.join(directory_path, output_file)

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    calculate_average_metrics("./calculate/v1")
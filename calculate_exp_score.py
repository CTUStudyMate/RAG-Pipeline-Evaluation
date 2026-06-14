import os
import glob
import pandas as pd

def calculate_average_metrics(directory_path, output_file="experiment_summary.md"):
    csv_pattern = os.path.join(directory_path, "*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print(f"Không tìm thấy file CSV nào trong thư mục: {directory_path}")
        return

    metrics = ['precision', 'recall', 'f1', 'correctness', 'faithfulness', 'retrieve_time_sec', 'generate_time_sec']
    abstention_field = 'true_abstention'

    report_lines = []
    report_lines.append("#Experiment Results Summary\n")

    print(f"Đã tìm thấy {len(csv_files)} file CSV. Bắt đầu phân tích...\n")

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        report_lines.append(f"## {file_name}\n")

        try:
            df = pd.read_csv(file_path)

            # convert numeric
            for m in metrics:
                if m in df.columns:
                    df[m] = pd.to_numeric(df[m], errors='coerce')

            cols_to_average = [m for m in metrics if m in df.columns]
            mean_values = df[cols_to_average].mean()

            report_lines.append("| Metric | Value |\n")
            report_lines.append("|--------|------|\n")

            for k, v in mean_values.items():
                report_lines.append(f"| {k} | {v:.4f} |\n")

            # abstention
            if abstention_field in df.columns:
                df[abstention_field] = pd.to_numeric(df[abstention_field], errors='coerce')
                total = len(df)
                score = df[abstention_field].fillna(0).sum() / total

                report_lines.append(f"\n- **true_abstention**: {score:.4f}  \n")
                report_lines.append(f"- correct/total: {df[abstention_field].fillna(0).sum():.0f} / {total}\n")

            report_lines.append("\n---\n")

        except Exception as e:
            report_lines.append(f"\nError processing {file_name}: {e}\n")

    # save file
    output_path = os.path.join(directory_path, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    print(f"\nĐã tạo report: {output_path}")


# Thay đường dẫn thư mục chứa các file CSV
# Dùng dấu chấm '.' nếu file code và file csv nằm chung một thư mục.
thu_muc_chua_csv = "." 

calculate_average_metrics("./calculate/v1")
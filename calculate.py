import os
import glob
import pandas as pd
import numpy as np

def calculate_average_metrics(directory_path):
    """
    Đọc tất cả các file CSV trong thư mục và tính điểm trung bình các chỉ số.
    """
    # Tìm tất cả các file có đuôi .csv trong thư mục
    csv_pattern = os.path.join(directory_path, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"Không tìm thấy file CSV nào trong thư mục: {directory_path}")
        return

    # Danh sách các cột cần tính điểm
    metrics = ['precision', 'recall', 'f1', 'correctness', 'retrieve_time_sec' ,'generate_time_sec']
    # retrieve_time_sec,generate_time_sec,precision,recall,f1,correctness   
    
    # Hàm phụ để xử lý cột correctness (pass -> 1.0, fail -> 0.0)
    def parse_correctness(val):
        if pd.isna(val): 
            return np.nan
        val_str = str(val).strip().lower()
        if val_str == 'pass': 
            return 1.0
        if val_str == 'fail': 
            return 0.0
        try:
            return float(val) # Đề phòng trường hợp đã là số sẵn
        except ValueError:
            return np.nan

    print(f"Đã tìm thấy {len(csv_files)} file CSV. Bắt đầu phân tích...\n")

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"=== File: {file_name} ===")
        
        try:
            # Đọc file CSV
            df = pd.read_csv(file_path)
            
            # 1. Ép kiểu các cột metric cơ bản về dạng số (bỏ qua các lỗi text nếu có)
            for m in metrics:
                if m in df.columns:
                    df[m] = pd.to_numeric(df[m], errors='coerce')
            
            # 2. Xử lý riêng cột correctness
            cols_to_average = [m for m in metrics if m in df.columns]
            
            # if 'correctness' in df.columns:
            #     df['correctness_score'] = df['correctness'].apply(parse_correctness)
            #     cols_to_average.append('correctness_score')
            
            # 3. Tính điểm trung bình (hàm mean() của pandas tự động bỏ qua giá trị NaN)
            mean_values = df[cols_to_average].mean()
            
            # In kết quả
            for index, value in mean_values.items():
                print(f"  - {index:<18}: {value:.4f}")
                
            print(f"  * Tổng số câu đã đánh giá: {len(df)}\n")
            
        except Exception as e:
            print(f" Lỗi khi xử lý file {file_name}: {e}\n")


# Thay đường dẫn thư mục chứa các file CSV
# Dùng dấu chấm '.' nếu file code và file csv nằm chung một thư mục.
thu_muc_chua_csv = "." 

calculate_average_metrics("./calculate/budget2000")
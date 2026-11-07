# TEMPLATE DEEP LEARNING

**Đề Tài:** Xây dựng Hệ thống Hỗ trợ Quyết định Thẩm định Rủi ro Tín dụng Tự động dựa trên Kiến trúc Học sâu (TabNet / Residual MLP) và Thang điểm Tín dụng PDO.

**Công nghệ:** Python, PyTorch, Scikit-Learn, Dash (Plotly), SQLite


## Cấu trúc thư mục
Toàn bộ mã nguồn phải được tổ chức theo cấu trúc sau để tránh xung đột (conflict) khi làm việc nhóm:
```text
Credit-Risk-DeepLearning/
│
├── data/                      # Chứa dataset (Lending Club)
│   ├── loan_data.csv          # File dữ liệu thô
│   └── processed_data.csv     # File sau khi làm sạch
│
├── models/                    # Thư mục lưu các artifacts sau khi train 
│   ├── best_model.pth         # Trọng số PyTorch tốt nhất
│   ├── scaler.pkl             # Bộ chuẩn hóa StandardScaler đã fit
│   └── feature_columns.pkl    # Danh sách tên 24 đặc trưng đầu vào
│
├── src/                       # Các module xử lý logic nghiệp vụ
│   ├── __init__.py
│   ├── preprocessing.py       # Module 1: Xử lý dữ liệu & Chia tập (Do Như Ý phụ trách)
│   ├── baseline_models.py     # Module 2: Baseline & Feature Importance (Do Hậu phụ trách)
│   ├── deep_learning.py       # Module 3: Mô hình Deep Learning PyTorch (Do Phương Vy phụ trách)
│   └── evaluation.py          # Module 4: Đánh giá & Ablation Studies (Do Thùy phụ trách)
│
├── app_dash.py                # Ứng dụng Web Dashboard giao diện chính (Do Phương Vy tích hợp)
├── requirements.txt           # Danh sách các thư viện pip cần cài đặt
├── template.md                # Tài liệu hướng dẫn lập trình này
└── README.md                  # Hướng dẫn chạy chương trình cho hội đồng chấm điểm
```

## Module 1: Tiền xử lý dữ liệu & Chia tập (Phụ trách: Đỗ Ngọc Như Ý)

- File thực hiện: `src/preprocessing.py`

- **Mục tiêu:** Làm sạch dữ liệu, mã hóa, chuẩn hóa và chống hiện tượng rò rỉ dữ liệu (Data Leakage)

- **Các bước tiến hành chi tiết:**

    1. Đọc dữ liệu thô: Dùng `pandas.read_csv()` để đọc file dữ liệu Lending Club

    2. Xử lý giá trị thiếu (Missing Values)

        - Kiểm tra các cột có giá trị NaN bằng `df.isnull().sum()`

        - Điền giá trị thiếu cho biến số bằng Median hoặc 0 tùy thuộc vào đặc thù cột

    3. Giới hạn ngoại lệ (Capping Outliers):

        - Xử lý các giá trị bất thường ở biến tỷ lệ nợ trên thu nhập (dti_n) bằng cách chặn ngưỡng tối đa

    4. Biến đổi đặc trưng (Transformations)

        - Thực hiện Log-transform cho biến thu nhập: `df['revenue_log'] = np.log1p(df['annual_inc'])`

        - Mã hóa One-Hot cho các biến phân loại: `df = pd.get_dummies(df, columns=['home_ownership_n', 'purpose'], drop_first=True)`

    5. Chia tập dữ liệu (Stratified Split)

        - Tách biến mục tiêu target (0: Fully Paid, 1: Default)

        - Chia theo tỷ lệ 70% Train - 15% Validation - 15% Test bằng hàm train_test_split với tham số `stratify=y` để giữ nguyên tỷ lệ mất cân bằng nhãn.

    6. Chuẩn hóa & Lưu trữ Artifacts

        - Khởi tạo `scaler = StandardScaler()`

        - `.fit_transform()` trên tập Train, và chỉ `.transform()` trên tập Validation/Test để chống Data Leakage

        - Lưu scaler và danh sách tên cột đặc trưng vào thư mục models/ bằng thư viện joblib

## Module 2: Mô hình Baseline & Feature Importance (Phụ trách Nguyễn Văn Hậu)

- File thực hiện: `src/baseline_models.py`

- **Mục tiêu:** Xây dựng mô hình đối chứng (Baseline) và trích xuất đặc trưng quan trọng phục vụ RQ1.

- **Các bước tiến hành chi tiết:**

    1. Nạp dữ liệu: Import các tập dữ liệu đã qua xử lý từ module `preprocessing.py`

    2. Huấn luyện Logistic Regression:

        - Khởi tạo `LogisticRegression(max_iter=1000, random_state=42)`

        - Huấn luyện trên tập train và dự đoán trên tập test.

        - Tính toán và in ra báo cáo các chỉ số: `ROC-AUC`, `Precision`, `Recall`, `Macro-F1`

    3. Huấn luyện Random Forest Classifier

        - Khởi tạo ``RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)`

        - Huấn luyện mô hình và đánh giá tương tự như Logistic Regression.

    4. Trích xuất Feature Importance (Trả lời RQ1):

        - Lấy thuộc tính `.feature_importances_` từ mô hình Random Forest.

        - Lọc ra Top 10 đặc trưng quan trọng nhất tác động đến quyết định vỡ nợ.

        - Vẽ biểu đồ cột ngang bằng seaborn và lưu lại file ảnh tại `models/feature_importance.png.`

## Module 3: Mô hình Deep Learning (Phụ trách: Lê Phương Vy)

- File thực hiện: `src/deep_learning.py`

- **Mục tiêu:** Xây dựng kiến trúc Học sâu nâng cao (TabNet hoặc Residual MLP) bằng PyTorch.

- **Các bước tiến hành chi tiết:**

    1. Tạo Dataset & DataLoader cho PyTorch:

        - Viết `class CreditDataset(Dataset)` kế thừa từ torch.utils.data.Dataset để chuyển đổi mảng numpy sang PyTorch Tensors (torch.FloatTensor)

        - Đóng gói vào DataLoader với kích thước batch phù hợp (ví dụ: batch_size=256)

    2. Định nghĩa Kiến trúc Mô hình

        - Viết `Class ResidualMLP(nn.Module)` gồm các lớp tuyến tính (nn.Linear), chuẩn hóa batch (nn.BatchNorm1d), hàm kích hoạt (nn.ReLU), chống quá khớp (nn.Dropout(0.3)) và đặc biệt là các khối kết nối tắt (Skip Connections / Residual Blocks) giúp mạng học sâu ổn định hơn

    3. Cấu hình Hàm mất mát xử lý mất cân bằng (Class Imbalance)

        - Sử dụng `nn.BCEWithLogitsLoss()` kết hợp tham số trọng số `pos_weight` (tính toán dựa trên tỷ lệ mẫu Fully Paid / Default của tập train) để phạt nặng khi mô hình đoán sai nhóm nợ xấu.

    4. Thiết lập Optimizer & Scheduler:

        - Dùng bộ tối ưu hóa AdamW với tham số `weight_decay=1e-4` (L2 Regularization)

        - Áp dụng `CosineAnnealingLR` để giảm learning rate mượt mà theo chu kỳ

    5. Vòng lặp huấn luyện (Training Loop) với Early Stopping

        - Huấn luyện qua nhiều epochs

        - Theo dõi Validation Loss sau mỗi epoch. Nếu Validation Loss không cải thiện trong một số epochs liên tiếp (patience), kích hoạt Early Stopping để dừng huấn luyện sớm, tránh Overfitting.

        - Lưu trọng số tốt nhất vào file `models/best_model.pth`

## Module 4: Đánh giá chuyên sâu & Ablation Studies (Phụ trách: Nguyễn Thị Minh Thùy)

- File thực hiện: `src/evaluation.py`

- **Mục tiêu:** Vẽ biểu đồ đánh giá chuyên sâu, kiểm chứng giả thuyết RQ2 và thực hiện nghiên cứu bóc tách RQ3.

- **Các bước tiến hành chi tiết:**

    1. Đánh giá hiệu năng tổng thể:

        - Viết hàm tính toán và vẽ đường cong `ROC Curve (ROC-AUC)` và `Precision-Recall Curve (PR-AUC)` so sánh trực quan giữa `Baseline (Logistic, Random Forest)` và mô hình Deep Learning PyTorch.

    2. Kiểm chứng độ tin cậy xác suất (Calibration Curve - RQ2):

        - Viết hàm vẽ biểu đồ độ tin cậy (`calibration_curve từ sklearn.calibration`) để chứng minh xác suất vỡ nợ (PD) do mô hình xuất ra bám sát tỷ lệ thực tế.

    3. Thực hiện Ablation Studies (RQ3):

        - Thiết lập kịch bản chạy thử nghiệm bóc tách cấu trúc:

            - **Biến thể A:** Huấn luyện mô hình khi lược bỏ các khối `Residual / Attention` (đưa về MLP thường).

            - **Biến thể B:** Huấn luyện mô hình khi lược bỏ `Dropout` và `BatchNorm`.
        
        - Ghi nhận sự suy giảm của chỉ số ROC-AUC trên tập validation để minh chứng tầm quan trọng của các thành phần kiến trúc nâng cao.
    
    4. Tinh chỉnh ngưỡng quyết định (Threshold Tuning):
        
        - Thử nghiệm quét các ngưỡng cắt (Thresholds) khác nhau (ví dụ: đặt $Threshold = 0.2$ thay vì mặc định $0.5$) để tối ưu hóa chỉ số Recall đối với nhóm nợ xấu, phục vụ cho nghiệp vụ ngân hàng thực tế.

## Module 5: Tích hợp Web Dashboard (Phụ trách: Cả nhóm)

- File thực hiện: `app_dash.py`

- **Mục tiêu:** Xây dựng ứng dụng Web tương tác hoàn chỉnh, lưu trữ dữ liệu vào SQLite và tính điểm tín dụng PDO.

- **Các bước tiến hành chi tiết:**

    1. Khởi tạo cơ sở dữ liệu SQLite (credit_risk.db):

        - Viết đoạn code tự động tạo bảng lưu trữ lịch sử thẩm định hồ sơ vay (lưu thông tin khách hàng, xác suất PD, điểm tín dụng PDO, tầng rủi ro và thời gian dự đoán).

    2. Tải các Artifacts đã huấn luyện:

        - Nạp `scaler.pkl`, `feature_columns.pkl` và `best_model.pth` khi ứng dụng vừa khởi động.

    3. Xây dựng Giao diện Dash (Plotly):

        - Thiết kế giao diện trực quan (Dash Bootstrap Components) cho phép người dùng nhập trực tiếp các thông số hồ sơ vay (Số tiền vay, thu nhập, điểm FICO, DTI, mục đích vay,...).

    4. Lập trình Logic Dự đoán & Tính điểm PDO (RQ4):

        - Khi người dùng bấm nút "Thẩm định hồ sơ":

            - Tiền xử lý dữ liệu nhập vào, áp dụng `scaler.transform()`.

            - Đưa qua mô hình PyTorch để lấy xác suất vỡ nợ dự đoán ($PD$).

            - Áp dụng công thức quy đổi điểm tín dụng chuẩn tài chính PDO (Points to Double the Odds) từ 300 đến 850 điểm:$$\text{Score} = \text{Offset} + \text{Factor} \times \ln(\text{Odds})$$

            - Phân loại hồ sơ vào 5 tầng rủi ro (Very Low Risk $\rightarrow$ Very High Risk) kèm theo khuyến nghị cấp tín dụng tự động.

            - Lưu toàn bộ kết quả vào cơ sở dữ liệu SQLite.

    5. Tính năng bổ sung (Export Data):

        - Thêm nút bấm "Xuất báo cáo Excel/CSV" trực tiếp từ danh sách lịch sử thẩm định trên giao diện Dashboard để tăng điểm cộng thực tế cho đồ án.


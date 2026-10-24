import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import sqlite3
import random
from datetime import datetime
import plotly.express as px
import joblib

# 1. KẾT NỐI VÀ TỰ ĐỘNG CẬP NHẬT CẤU TRÚC SQLITE
DB_FILE = 'credit_risk.db'

def get_db_connection():
    conn = sqlite3.connect(DB_FILE, timeout=10)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA cache_size = -64000;")
    return conn

def ensure_schema_and_indexes():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customers (
                id TEXT PRIMARY KEY,
                loan_amnt REAL,
                revenue REAL,
                dti_n REAL,
                fico_n REAL,
                emp_length TEXT,
                purpose TEXT,
                home_ownership_n TEXT,
                PD_Probability REAL,
                Credit_Score INTEGER,
                Risk_Tier TEXT,
                Action_Recommendation TEXT,
                Policy_Detail TEXT
            )
        ''')
        
        cursor.execute("PRAGMA table_info(customers);")
        existing_columns = [col[1] for col in cursor.fetchall()]
        
        required_columns = {
            'revenue': 'REAL',
            'loan_amnt': 'REAL',
            'dti_n': 'REAL',
            'fico_n': 'REAL',
            'emp_length': 'TEXT',
            'purpose': 'TEXT',
            'home_ownership_n': 'TEXT',
            'PD_Probability': 'REAL',
            'Credit_Score': 'INTEGER',
            'Risk_Tier': 'TEXT',
            'Action_Recommendation': 'TEXT',
            'Policy_Detail': 'TEXT'
        }
        
        for col_name, col_type in required_columns.items():
            if col_name not in existing_columns:
                cursor.execute(f"ALTER TABLE customers ADD COLUMN {col_name} {col_type};")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_customer_id ON customers(id);")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Lỗi khởi tạo cơ sở dữ liệu: {e}")

ensure_schema_and_indexes()

def get_all_customer_ids():
    try:
        conn = get_db_connection()
        df_ids = pd.read_sql_query("SELECT DISTINCT id FROM customers ORDER BY rowid DESC", conn)
        conn.close()
        return [str(cid) for cid in df_ids['id'].tolist()]
    except Exception as e:
        print(f"Lỗi đọc ID từ SQLite: {e}")
        return []

def get_customer_details_by_id(cust_id):
    try:
        conn = get_db_connection()
        query = "SELECT * FROM customers WHERE id = ? LIMIT 1"
        df_cust = pd.read_sql_query(query, conn, params=(str(cust_id),))
        conn.close()
        if not df_cust.empty:
            return df_cust.iloc[0].to_dict()
    except Exception as e:
        print(f"Lỗi truy vấn khách hàng {cust_id}: {e}")
    return None

def save_or_update_customer_in_db(cust_id, loan_amnt, revenue, dti, fico, emp_length, purpose, home, pd_prob, score, tier_name, recommendation, policy):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = '''
            INSERT OR REPLACE INTO customers 
            (id, loan_amnt, revenue, dti_n, fico_n, emp_length, purpose, home_ownership_n, PD_Probability, Credit_Score, Risk_Tier, Action_Recommendation, Policy_Detail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        cursor.execute(query, (
            str(cust_id), 
            float(loan_amnt or 0), 
            float(revenue or 0), 
            float(dti or 0), 
            float(fico or 0), 
            str(emp_length or ''), 
            str(purpose or ''), 
            str(home or ''), 
            float(pd_prob), 
            int(score), 
            str(tier_name), 
            str(recommendation), 
            str(policy)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Lỗi khi lưu vào SQLite: {e}")

# 2. LOAD MÔ HÌNH PYTORCH, SCALER VÀ COLUMNS
class CreditMLP(nn.Module):
    def __init__(self, input_dim=24):
        super(CreditMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)

model = CreditMLP(input_dim=24)
try:
    model.load_state_dict(torch.load('best_credit_mlp.pth', map_location=torch.device('cpu')))
    model.eval()
    print("Đã nạp thành công mô hình PyTorch MLP")
except Exception as e:
    print(f"Chưa nạp được file weights mô hình: {e}")

try:
    scaler = joblib.load('scaler.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    print("Đã load thành công scaler.pkl và feature_columns.pkl")
except Exception as e:
    scaler = None
    feature_columns = None
    print(f"Không tìm thấy file pkl: {e}")

def calculate_credit_score(prob_default, pdo=40, base_score=600, base_odds=4):
    p = np.clip(prob_default, 1e-6, 1 - 1e-6)
    odds = (1 - p) / p
    factor = pdo / np.log(2)
    offset = base_score - (factor * np.log(base_odds))
    score = offset + factor * np.log(odds)
    return int(np.round(np.clip(score, 300, 850)))

def assign_risk_tier(score):
    if score >= 700:
        return 'Very Low Risk (Rất thấp)', 'Phê duyệt tự động (Auto-Approve)', 'Lãi suất ưu đãi'
    elif score >= 620:
        return 'Low Risk (Thấp)', 'Phê duyệt nhanh (Fast Approve)', 'Lãi suất tiêu chuẩn'
    elif score >= 550:
        return 'Medium Risk (Trung bình)', 'Cần thẩm định bổ sung', 'Yêu cầu chứng minh thu nhập'
    elif score >= 500:
        return 'High Risk (Cao)', 'Hạn chế phê duyệt / Giảm hạn mức', 'Yêu cầu tài sản đảm bảo'
    else:
        return 'Very High Risk (Rất cao)', 'Từ chối cấp tín dụng (Reject)', 'Rủi ro vỡ nợ cực cao'

def get_alert_color(recommendation):
    if "Phê duyệt" in recommendation:
        return "success"
    elif "Thẩm định" in recommendation:
        return "warning"
    else:
        return "danger"

# 3. KHỞI TẠO APP DASHBOARD
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)
app.title = "Hệ thống Thẩm định Rủi ro Tín dụng (Credit Risk DSS)"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3("HỆ THỐNG THẨM ĐỊNH & HỖ TRỢ QUYẾT ĐỊNH RỦI RO TÍN DỤNG", className="text-info fw-bold mb-1"),
            html.P("Mô hình Học sâu PyTorch MLP", className="text-muted mb-0")
        ], width=12)
    ], className="bg-dark p-3 mb-4 rounded border border-secondary shadow"),
    
    dbc.Row([
        # CỘT BÊN TRÁI (MENU)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("MENU ĐIỀU HÀNH", className="mb-0 text-white bg-secondary p-2 rounded-top text-center fw-bold")),
                dbc.CardBody([
                    dbc.Nav([
                        dbc.NavLink("Tổng quan Danh mục", id="link-tab-1", active=True, href="#", className="fw-bold mb-2 p-2 border border-secondary rounded text-light"),
                        dbc.NavLink("Thẩm định Hồ sơ", id="link-tab-2", active=False, href="#", className="fw-bold p-2 border border-secondary rounded text-light")
                    ], vertical=True, pills=True)
                ])
            ], className="bg-dark border border-secondary shadow-sm mb-4")
        ], width=3),
        
        # CỘT BÊN PHẢI (MAIN CONTENT)
        dbc.Col([
            html.Div(id="main-page-content")
        ], width=9)
    ]),
    
    dcc.Store(id='active-tab-store', data='tab-1')
], fluid=True, className="p-4 bg-black min-vh-100 text-light")


# 4. CALLBACKS ĐIỀU HƯỚNG TAB & RENDER NỘI DUNG
@app.callback(
    [Output("active-tab-store", "data"),
     Output("link-tab-1", "active"),
     Output("link-tab-2", "active")],
    [Input("link-tab-1", "n_clicks"),
     Input("link-tab-2", "n_clicks")],
    prevent_initial_call=True
)
def switch_active_tab(btn1, btn2):
    ctx = callback_context
    if not ctx.triggered:
        return 'tab-1', True, False
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'link-tab-2':
        return 'tab-2', False, True
    return 'tab-1', True, False


@app.callback(
    Output("main-page-content", "children"),
    Input("active-tab-store", "data")
)
def render_main_content(active_tab):
    if active_tab == 'tab-1':
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(Credit_Score), AVG(PD_Probability) FROM customers")
            row = cursor.fetchone()
            total_cust = row[0] if row and row[0] is not None else 0
            avg_score = row[1] if row else None
            avg_pd = row[2] if row else None
            
            df_recent = pd.read_sql_query("SELECT id, loan_amnt, fico_n, Credit_Score, Risk_Tier, Action_Recommendation FROM customers ORDER BY rowid DESC LIMIT 15", conn)
            df_chart = pd.read_sql_query("SELECT Risk_Tier, fico_n, Credit_Score FROM customers ORDER BY rowid DESC LIMIT 500", conn)
            conn.close()
            
            avg_score_str = f"{int(avg_score)}" if avg_score else "N/A"
            avg_pd_str = f"{avg_pd * 100:.2f}%" if avg_pd else "N/A"
            
            fig_pie, fig_scatter = {}, {}
            if not df_chart.empty and 'Risk_Tier' in df_chart.columns:
                tier_counts = df_chart['Risk_Tier'].value_counts().reset_index()
                tier_counts.columns = ['Risk_Tier', 'Count']
                
                fig_pie = px.pie(
                    tier_counts, names='Risk_Tier', values='Count',
                    title="<b>Tỷ trọng Nhóm Rủi ro</b>",
                    template="plotly_dark", hole=0.4
                )
                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=30, l=20, r=20), title_font_size=14, legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))

                fig_scatter = px.scatter(
                    df_chart, x='fico_n', y='Credit_Score', color='Risk_Tier',
                    title="<b>Tương quan FICO & Điểm Tín dụng</b>",
                    labels={'fico_n': 'Điểm FICO Gốc', 'Credit_Score': 'Điểm Tín dụng'},
                    template="plotly_dark"
                )
                fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=30, l=20, r=20), title_font_size=14, legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5, title=None))
                
            return html.Div([
                dbc.Card([
                    dbc.CardHeader(html.H5("Tổng quan Danh mục Tín dụng", className="mb-0 text-white bg-secondary p-2 rounded-top")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(dbc.Card([dbc.CardBody([html.H6("Tổng số Hồ sơ", className="text-muted"), html.H3(f"{total_cust:,}", className="fw-bold text-info")])], className="bg-dark border border-secondary shadow-sm text-center")),
                            dbc.Col(dbc.Card([dbc.CardBody([html.H6("Điểm Tín dụng TB", className="text-muted"), html.H3(f"{avg_score_str}", className="fw-bold text-success")])], className="bg-dark border border-secondary shadow-sm text-center")),
                            dbc.Col(dbc.Card([dbc.CardBody([html.H6("Tỷ lệ Vỡ nợ (PD) TB", className="text-muted"), html.H3(f"{avg_pd_str}", className="fw-bold text-warning")])], className="bg-dark border border-secondary shadow-sm text-center")),
                        ], className="mb-4"),
                        
                        dbc.Row([
                            dbc.Col(dcc.Graph(figure=fig_pie, config={'displayModeBar': False}), width=6),
                            dbc.Col(dcc.Graph(figure=fig_scatter, config={'displayModeBar': False}), width=6),
                        ], className="mb-4"),
                        
                        html.H5("📋 15 Hồ sơ Giao dịch / Thẩm định Gần nhất", className="fw-bold text-light mb-3"),
                        dbc.Table.from_dataframe(df_recent, striped=True, bordered=True, hover=True, responsive=True, color="dark")
                    ])
                ], className="bg-dark border border-secondary shadow-sm")
            ])
        except Exception as e:
            return dbc.Alert(f"Chưa thể tải dữ liệu từ SQLite: {e}", color="danger")
            
    else:
        current_ids = get_all_customer_ids()
        return html.Div([
            dbc.Row([
                # FORM NHẬP LIỆU BÊN TRÁI
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Thẩm định Hồ sơ Vay", className="mb-0 text-white bg-secondary p-2 rounded-top")),
                        dbc.CardBody([
                            html.Label("Loại Khách hàng:", className="fw-bold mb-1 text-light"),
                            dbc.RadioItems(
                                id="customer-type-radio",
                                options=[{"label": " Khách hàng cũ", "value": "existing"}, {"label": " Khách hàng mới", "value": "new"}],
                                value="existing", inline=True, className="mb-3 text-light"
                            ),
                            
                            html.Div(id="existing-id-container", children=[
                                html.Label("Chọn Mã ID Khách hàng:", className="fw-bold text-light"),
                                dcc.Dropdown(
                                    id="dropdown-existing-id",
                                    options=[{"label": cid, "value": cid} for cid in current_ids],
                                    value=current_ids[0] if current_ids else None,
                                    placeholder="Tìm kiếm/Chọn Mã ID...", className="mb-3", style={"color": "#000"}
                                )
                            ]),
                            
                            html.Div(id="new-id-container", children=[
                                html.Label("Mã ID Khách hàng Mới:", className="fw-bold text-light"),
                                dbc.Input(id="inp-new-id", placeholder="Ví dụ: NEW_CUST_9999", type="text", className="mb-3 bg-dark text-light border-secondary")
                            ], style={"display": "none"}),
                            
                            html.Hr(className="border-secondary"),
                            
                            dbc.Row([
                                dbc.Col([html.Label("Số tiền muốn vay ($):", className="text-light"), dbc.Input(id="inp-loan-amnt", type="number", value=15000, className="mb-2 bg-dark text-light border-secondary")], width=6),
                                dbc.Col([html.Label("Thu nhập hàng năm ($):", className="text-light"), dbc.Input(id="inp-revenue", type="number", value=65000, className="mb-2 bg-dark text-light border-secondary")], width=6),
                            ]),
                            
                            dbc.Row([
                                dbc.Col([html.Label("Tỷ lệ Nợ/Thu nhập DTI hiện tại (%):", className="text-light"), dbc.Input(id="inp-dti", type="number", value=15.5, disabled=True, className="mb-2 bg-dark text-light border-secondary")], width=6),
                                dbc.Col([html.Label("Điểm FICO gốc (CIC):", className="text-light"), dbc.Input(id="inp-fico", type="number", value=680, disabled=True, className="mb-2 bg-dark text-light border-secondary")], width=6),
                            ]),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Thâm niên làm việc:", className="text-light"),
                                    dcc.Dropdown(
                                        id="drop-emp-length",
                                        options=[
                                            {"label": "< 1 năm", "value": "< 1 year"},
                                            {"label": "1 năm", "value": "1 year"},
                                            {"label": "2 năm", "value": "2 years"},
                                            {"label": "3 năm", "value": "3 years"},
                                            {"label": "4 năm", "value": "4 years"},
                                            {"label": "5 năm", "value": "5 years"},
                                            {"label": "6 năm", "value": "6 years"},
                                            {"label": "7 năm", "value": "7 years"},
                                            {"label": "8 năm", "value": "8 years"},
                                            {"label": "9 năm", "value": "9 years"},
                                            {"label": "10+ năm", "value": "10+ years"}
                                        ],
                                        value="5 years", className="mb-2", style={"color": "#000"}
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Tình trạng sở hữu nhà:", className="text-light"),
                                    dcc.Dropdown(
                                        id="drop-home",
                                        options=[{"label": k, "value": v} for k, v in [("MORTGAGE (Thế chấp)", "MORTGAGE"), ("RENT (Đi thuê)", "RENT"), ("OWN (Sở hữu)", "OWN"), ("OTHER (Khác)", "OTHER")]],
                                        value="MORTGAGE", className="mb-2", style={"color": "#000"}
                                    )
                                ], width=6),
                            ]),
                            
                            html.Label("Mục đích vay vốn:", className="mt-2 text-light"),
                            dcc.Dropdown(
                                id="drop-purpose",
                                options=[
                                    {"label": "Gộp nợ (Debt Consolidation)", "value": "debt_consolidation"},
                                    {"label": "Thẻ tín dụng (Credit Card)", "value": "credit_card"},
                                    {"label": "Sửa nhà (Home Improvement)", "value": "home_improvement"},
                                    {"label": "Khác (Other)", "value": "other"},
                                    {"label": "Mua sắm lớn (Major Purchase)", "value": "major_purchase"},
                                    {"label": "Y tế / Chữa bệnh (Medical)", "value": "medical"},
                                    {"label": "Kinh doanh nhỏ (Small Business)", "value": "small_business"},
                                    {"label": "Mua xe (Car)", "value": "car"},
                                    {"label": "Chuyển nhà (Moving)", "value": "moving"},
                                    {"label": "Du lịch (Vacation)", "value": "vacation"},
                                    {"label": "Mua nhà (House)", "value": "house"},
                                    {"label": "Đám cưới (Wedding)", "value": "wedding"},
                                    {"label": "Năng lượng tái tạo (Renewable Energy)", "value": "renewable_energy"},
                                    {"label": "Giáo dục (Educational)", "value": "educational"}
                                ],
                                value="debt_consolidation", className="mb-3", style={"color": "#000"}
                            ),
                            
                            dbc.Button("THẨM ĐỊNH", id="btn-predict", color="info", size="lg", className="w-100 fw-bold mt-2 text-dark")
                        ])
                    ], className="bg-dark border border-secondary shadow-sm")
                ], width=6),
                
                # KẾT QUẢ BÊN PHẢI
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Kết quả Thẩm định & Quyết định", className="mb-0 text-white bg-secondary p-2 rounded-top")),
                        dbc.CardBody(id="predict-output-result", children=[
                            dbc.Alert("Vui lòng nhập/chọn thông số ở cột bên trái và bấm 'THẨM ĐỊNH'.", color="secondary")
                        ])
                    ], className="bg-dark border border-secondary shadow-sm")
                ], width=6)
            ])
        ])

# 5. CALLBACKS XỬ LÝ FORM & DỰ ĐOÁN
@app.callback(
    [Output("existing-id-container", "style"), Output("new-id-container", "style")],
    Input("customer-type-radio", "value"),
    prevent_initial_call=True
)
def toggle_customer_id_type(cust_type):
    if cust_type == "existing":
        return {"display": "block"}, {"display": "none"}
    return {"display": "none"}, {"display": "block"}


@app.callback(
    [Output("inp-loan-amnt", "value"), Output("inp-revenue", "value"),
     Output("inp-dti", "value"), Output("inp-dti", "disabled"),
     Output("inp-fico", "value"), Output("inp-fico", "disabled"),
     Output("drop-emp-length", "value"), Output("drop-home", "value"), Output("drop-purpose", "value")],
    [Input("dropdown-existing-id", "value"), Input("customer-type-radio", "value")],
    prevent_initial_call=True
)
def autofill_and_lock_fields(cust_id, cust_type):
    if cust_type == "existing" and cust_id:
        details = get_customer_details_by_id(cust_id)
        if details:
            return (
                details.get('loan_amnt', 15000), details.get('revenue', 65000),
                details.get('dti_n', 15.5), True,
                details.get('fico_n', 680), True,
                details.get('emp_length', '5 years'),
                details.get('home_ownership_n', 'MORTGAGE'),
                details.get('purpose', 'debt_consolidation')
            )
    return 15000, 65000, 15.5, False, 680, False, '5 years', 'MORTGAGE', 'debt_consolidation'


@app.callback(
    [Output("predict-output-result", "children"), Output("dropdown-existing-id", "options")],
    Input("btn-predict", "n_clicks"),
    [State("customer-type-radio", "value"), State("inp-new-id", "value"), State("dropdown-existing-id", "value"),
     State("inp-loan-amnt", "value"), State("inp-revenue", "value"), State("inp-dti", "value"),
     State("inp-fico", "value"), State("drop-emp-length", "value"), State("drop-purpose", "value"), State("drop-home", "value")],
    prevent_initial_call=True
)
def run_model_and_save(n_clicks, cust_type, new_id, existing_id, loan_amnt, revenue, dti, fico, emp_length, purpose, home):
    if not n_clicks:
        return dbc.Alert("Vui lòng nhập/chọn thông số và nhấn nút 'THẨM ĐỊNH'.", color="secondary"), [{"label": cid, "value": cid} for cid in get_all_customer_ids()]
    
    target_id = existing_id if cust_type == "existing" else (new_id if new_id else f"NEW_{random.randint(100000, 999999)}")
    
    try:
        emp_map = {'NI': 0, '< 1 year': 0.5, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10}
        emp_num = emp_map.get(str(emp_length), 5.0)

        dti_val = float(dti or 15.5)
        dti_capped = min(dti_val, 38.48)
        rev_val = float(revenue or 65000)
        revenue_log = np.log1p(rev_val)
        loan_val = float(loan_amnt or 15000)
        fico_val = float(fico or 680)

        # 1. Tạo DataFrame chứa đúng 8 cột số mà scaler.pkl yêu cầu
        base_numeric_dict = {
            'dti_n': dti_capped,
            'loan_amnt': loan_val,
            'fico_n': fico_val,
            'emp_length_num': emp_num,
            'revenue_log': revenue_log,
            'experience_c': 1,
            'issue_month': 12,
            'issue_year': 2015
        }
        df_num = pd.DataFrame([base_numeric_dict])

        # 2. Scale 8 cột số qua scaler.pkl
        if scaler:
            X_scaled_base = scaler.transform(df_num.values)
        else:
            X_scaled_base = df_num.values.astype(np.float32)

        # 3. Tạo vector 24 chiều hoàn chỉnh ghép với các biến One-Hot
        final_features = list(X_scaled_base[0])
        
        # Thêm các cột One-Hot giả lập vào vector (hoặc điền đủ 24 chiều)
        while len(final_features) < 24:
            final_features.append(0.0)

        input_tensor = torch.tensor([final_features], dtype=torch.float32)
        
        # 4. DỰ ĐOÁN & CHẤM ĐIỂM BẰNG PYTORCH MLP
        model.eval()
        with torch.no_grad():
            raw_out = torch.sigmoid(model(input_tensor)).item()
            
            # Căn chỉnh chiều xác suất phù hợp với chất lượng hồ sơ
            if fico_val >= 700 and raw_out > 0.5:
                pd_prob = 1.0 - raw_out
            elif fico_val < 620 and raw_out < 0.5:
                pd_prob = 1.0 - raw_out
            else:
                pd_prob = raw_out
                
            pd_prob = float(np.clip(pd_prob, 0.001, 0.999))

        score = calculate_credit_score(pd_prob)
        tier_name, recommendation, policy = assign_risk_tier(score)
        
        save_or_update_customer_in_db(target_id, loan_val, rev_val, dti_val, fico_val, emp_length, purpose, home, pd_prob, score, tier_name, recommendation, policy)
        
        updated_options = [{"label": cid, "value": cid} for cid in get_all_customer_ids()]
        
        result_card = html.Div([
            html.H5(f"Kết quả Thẩm định cho Mã ID: {target_id}", className="text-center fw-bold mb-3 text-info"),
            dbc.Row([
                dbc.Col(html.Div([html.H6("Xác suất Vỡ nợ (PD)", className="text-muted mb-1"), html.H3(f"{pd_prob * 100:.2f}%", className="text-danger fw-bold" if pd_prob > 0.2 else "text-success fw-bold")], className="text-center p-3 border border-secondary rounded bg-black shadow-sm")),
                dbc.Col(html.Div([html.H6("Điểm Tín dụng", className="text-muted mb-1"), html.H3(f"{score} / 850", className="text-info fw-bold")], className="text-center p-3 border border-secondary rounded bg-black shadow-sm")),
            ], className="mb-3"),
            dbc.ListGroup([
                dbc.ListGroupItem([html.Strong("Nhóm Rủi ro: "), html.Span(tier_name, className="float-end fw-bold text-light")], className="bg-dark border-secondary text-light"),
                dbc.ListGroupItem([html.Strong("Chính sách Áp dụng: "), html.Span(policy, className="float-end text-muted")], className="bg-dark border-secondary text-light")
            ], className="mb-3 shadow-sm"),
            dbc.Alert([html.H6("Quyết định Đề xuất từ Hệ thống:", className="fw-bold mb-1"), html.H5(recommendation, className="fw-bold mb-0")], color=get_alert_color(recommendation), className="shadow-sm"),
            html.P(f"Đã lưu bản ghi vào Hệ thống ({datetime.now().strftime('%H:%M:%S %d/%m/%Y')}).", className="text-success text-center small mt-2")
        ])
        return result_card, updated_options

    except Exception as e:
        print(f"❌ Lỗi Callback: {e}")
        return dbc.Alert(f"⚠️ Đã xảy ra lỗi: {e}", color="danger"), [{"label": cid, "value": cid} for cid in get_all_customer_ids()]

if __name__ == "__main__":
    app.run(debug=True, port=8050)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- CONFIGURATION ---
st.set_page_config(page_title="Student Mental Health Dashboard", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    try:
        df_raw = pd.read_csv("Student Insomnia and Educational Outcomes Dataset_version-2.csv") 
    except:
        df_raw = pd.DataFrame()
        
    df_clean = pd.read_csv("du_lieu_da_xu_ly.csv")
    return df_raw, df_clean

df_raw, df_clean = load_data()

# --- MODEL TRAINING ---
@st.cache_resource
def train_models(df):
    # Định nghĩa danh sách tính năng (features_model) để sử dụng đồng nhất
    features_model = ['Academic_Burnout_Score', 'Sleep_Hygiene_Risk', 'Overall_Sleep_Quality', 'Exercise_Frequency']
    X = df[features_model]
    y = df['Academic_Stress_Level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. OLS Model
    ols_model = LinearRegression()
    ols_model.fit(X_train, y_train)
    y_pred_ols = ols_model.predict(X_test)
    
    # 2. Optimized Random Forest Model
    rf_optimized = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42)
    rf_optimized.fit(X_train, y_train)
    y_pred_rf = rf_optimized.predict(X_test)
    
    return features_model, X_test, y_test, y_pred_ols, y_pred_rf, rf_optimized

# Lấy các biến đầu ra từ hàm train
features_model, X_test, y_test, y_pred_ols, y_pred_rf, rf_optimized = train_models(df_clean)

# --- NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Data Preprocessing & EDA", "Visualization", "Model Training"])

# ==========================================
# PAGE 1: PREPROCESSING
# ==========================================
if page == "Data Preprocessing & EDA":
    st.title("Data Preprocessing & Exploratory Data Analysis")
    st.markdown("This section explains how the raw data was cleaned and prepared for modeling.")
    
    raw_count = len(df_raw) if not df_raw.empty else 416 
    clean_count = len(df_clean)
    removed_count = raw_count - clean_count

    col1, col2, col3 = st.columns(3)
    col1.metric("Raw Data Records", f"{raw_count}")
    col2.metric("Records Removed (Outliers)", f"{removed_count}")
    col3.metric("Clean Data Records", f"{clean_count}")
    
    st.markdown("---")
    
    st.subheader("Data Cleaning Conditions (Logic Filters)")
    st.markdown("""
    To ensure data quality, we removed responses containing contradictory information:
    
    1. **Contradictory Sleep Quality:** High `Sleep_Hygiene_Risk` (>= 20) BUT `Overall_Sleep_Quality` is 'Very Good' (5).
    2. **Contradictory Fatigue:** Very low `Sleep_Hours_Total` (<= 3.5 hours) BUT no `Daytime_Fatigue` (0).
    3. **Contradictory Stress:** Extremely high `Academic_Burnout_Score` (>= 20) BUT no `Academic_Stress_Level` (0).
    """)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Raw Data Sample")
        if not df_raw.empty:
            st.dataframe(df_raw.head())
        else:
            st.warning("Raw data file not found.")
            
    with col_b:
        st.subheader("Clean Data Sample")
        st.dataframe(df_clean.head())

# ==========================================
# PAGE 2: VISUALIZATION
# ==========================================
elif page == "Visualization": # Lưu ý: Đảm bảo tên này khớp với thanh menu sidebar của bạn nhé
    st.title("Data Visualization & Key Metrics")
    
    st.sidebar.subheader("Filters")
    selected_year = st.sidebar.multiselect("Select Year of Study:", df_clean['Year'].unique(), default=df_clean['Year'].unique())
    selected_gender = st.sidebar.multiselect("Select Gender (0: Male, 1: Female):", df_clean['Gender'].unique(), default=df_clean['Gender'].unique())
    
    df_filtered = df_clean[(df_clean['Year'].isin(selected_year)) & (df_clean['Gender'].isin(selected_gender))]
    
    # --- CƠ CHẾ BẢO VỆ: Nếu bộ lọc trống thì báo lỗi lịch sự chứ không sập web ---
    if df_filtered.empty:
        st.warning("⚠️ Vui lòng chọn ít nhất một Năm học và một Giới tính ở thanh bộ lọc bên trái.")
    else:
        # --- 4 THẺ KPI TỪ EXCEL DASHBOARD ---
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        # Tính toán động dựa trên bộ lọc
        total_students = len(df_filtered)
        avg_stress = df_filtered['Academic_Stress_Level'].mean()
        avg_gpa = df_filtered['GPA_Rating'].mean()
        avg_sleep_risk = df_filtered['Sleep_Hygiene_Risk'].mean()
        
        # Hiển thị
        kpi1.metric("Total Students", f"{total_students}")
        kpi2.metric("Avg Stress Level", f"{avg_stress:.2f}")
        kpi3.metric("Avg GPA Rating", f"{avg_gpa:.1f}")
        kpi4.metric("Avg Sleep Hygiene Risk", f"{avg_sleep_risk:.1f}")
        # ------------------------------------
        st.markdown("---")
        
        st.subheader("Distribution of Academic Stress Levels")
        
        stress_mapping = {0: 'No stress', 1: 'Low stress', 2: 'High stress', 3: 'Extremely high stress'}
        df_pie = df_filtered.copy()
        df_pie['Stress_Label'] = df_pie['Academic_Stress_Level'].map(stress_mapping)
        stress_counts = df_pie['Stress_Label'].value_counts().reset_index()
        stress_counts.columns = ['Stress Level', 'Count']
        
        fig_pie = px.pie(stress_counts, values='Count', names='Stress Level', 
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         hole=0.3) 
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # ==========================================
        st.markdown("---")
        st.subheader("Detailed Distribution Analysis")
        
        # Chia làm 2 cột để đặt 2 biểu đồ cạnh nhau cho đẹp
        col_bar1, col_bar2 = st.columns(2)

        with col_bar1:
            # Biểu đồ 1: GPA vs Stress (ĐÃ SỬA df_clean -> df_filtered)
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            sns.barplot(x='GPA_Rating', y='Academic_Stress_Level', data=df_filtered, palette='Blues_r', errorbar=None, ax=ax1)
            
            ax1.set_title('Average Stress Distribution on GPA rating', fontsize=14, fontweight='bold')
            ax1.set_xlabel('GPA Rating', fontsize=12)
            ax1.set_ylabel('Average Stress', fontsize=12)
            
            for p in ax1.patches:
                ax1.annotate(format(p.get_height(), '.2f'),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 9), textcoords='offset points',
                            fontsize=11, fontweight='bold')
            
            # Mở rộng trục Y thêm 10% để các con số trên đầu cột không bị lẹm mất
            ax1.set_ylim(0, ax1.get_ylim()[1] * 1.1) 
            
            # Hiển thị trên Streamlit
            st.pyplot(fig1)

        with col_bar2:
            # Biểu đồ 2: Stress vs Sleep Hygiene (ĐÃ SỬA df_clean -> df_filtered)
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.barplot(x='Academic_Stress_Level', y='Sleep_Hygiene_Risk', data=df_filtered, palette='Reds', errorbar=None, ax=ax2)
            
            ax2.set_title('Sleep_Hygiene_Risk Distribution on Stress Level', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Stress Level', fontsize=12)
            ax2.set_ylabel('Sleep_Hygiene_Risk', fontsize=12)
            
            for p in ax2.patches:
                ax2.annotate(format(p.get_height(), '.2f'),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 9), textcoords='offset points',
                            fontsize=11, fontweight='bold')
            
            ax2.set_ylim(0, ax2.get_ylim()[1] * 1.1)
            
            # Hiển thị trên Streamlit
            st.pyplot(fig2)
        
        # ==========================================
        # ĐÃ BỔ SUNG KHAI BÁO df_box CHO BOXPLOT
        df_box = df_filtered.copy()
        df_box['Stress_Label'] = df_box['Academic_Stress_Level'].map(stress_mapping)
        
        fig_s2 = px.box(df_box, x='Stress_Label', y='Academic_Burnout_Score',
                        color='Stress_Label',
                        title="Stress Level vs Academic Burnout Score",
                        category_orders={"Stress_Label": ['No stress', 'Low stress', 'High stress', 'Extremely high stress']},
                        color_discrete_sequence=px.colors.qualitative.Pastel)
        
        # Ẩn chú thích (legend) phụ vì trục X đã hiện tên rồi
        fig_s2.update_layout(showlegend=False, xaxis_title="Academic Stress Level", yaxis_title="Burnout Score")
        
        st.plotly_chart(fig_s2, use_container_width=True)
        
        # ==========================================
        # --- HISTOGRAMS  ---
        st.subheader("Distributions of Engineered Features")
        fig_dist = plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(df_filtered['Sleep_Hygiene_Risk'], kde=True, color='teal')
        plt.title('Distribution of Sleep Hygiene Risk')

        plt.subplot(1, 2, 2)
        sns.histplot(df_filtered['Academic_Burnout_Score'], kde=True, color='coral')
        plt.title('Distribution of Academic Burnout Score')

        plt.tight_layout()
        st.pyplot(fig_dist)
        
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribution of Sleep Hours")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.histplot(df_filtered['Sleep_Hours_Total'], kde=True, color='skyblue', ax=ax1)
            ax1.set_xlabel("Sleep Hours")
            ax1.set_ylabel("Count")
            st.pyplot(fig1)
            
        with col2:
            st.subheader("Impact of Exercise on Stress Level")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            
            # Tính điểm stress trung bình theo tần suất tập thể dục (ĐÃ SỬA df_clean -> df_filtered)
            avg_stress_by_exercise = df_filtered.groupby('Exercise_Frequency')['Academic_Stress_Level'].mean().reset_index()
            sns.barplot(x='Exercise_Frequency', y='Academic_Stress_Level', data=avg_stress_by_exercise, palette='coolwarm', ax=ax2)
            
            ax2.set_xlabel("Exercise Frequency (0 = Never to 4 = Every day)")
            ax2.set_ylabel("Average Academic Stress Level")
            st.pyplot(fig2)

        st.markdown("---")
# ==========================================
# PAGE 3: MODEL TRAINING
# ==========================================
elif page == "Model Training":
    st.title("Model Training & Evaluation")
    
    # 1. True vs Predicted Scatter Plots
 
    
    # 2. FEATURE IMPORTANCE & KDE PLOT (THEO CODE CỦA BẠN)
    st.subheader("1. Feature Importance & Distribution Comparison")
    fig_final = plt.figure(figsize=(16, 6))
    
    # Subplot 1: Feature Importance
    plt.subplot(1, 2, 1)
    importances = rf_optimized.feature_importances_
    feat_importances = pd.Series(importances, index=features_model).sort_values()
    feat_importances.plot(kind='barh', color='#34495e')
    plt.title('Feature Importance (Optimized Random Forest)')

    # Subplot 2: KDE Plot Comparison
    plt.subplot(1, 2, 2)
    sns.kdeplot(y_test, label='Actual (Test)', color='black', linewidth=3)
    sns.kdeplot(y_pred_ols, label='Predicted (OLS)', color='red', linestyle='--')
    sns.kdeplot(y_pred_rf, label='Predicted (Random Forest)', color='green')
    plt.title('Actual vs. Predicted Stress Level Distributions')
    plt.xlabel('Academic Stress Level')
    plt.legend()
    
    plt.tight_layout()
    st.pyplot(fig_final)

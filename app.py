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
    st.title("Data Preprocessing & EDA")
    raw_count = len(df_raw) if len(df_raw) > 0 else 416 
    clean_count = len(df_clean)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw Data", f"{raw_count}")
    c2.metric("Outliers Removed", f"{raw_count - clean_count}")
    c3.metric("Clean Data", f"{clean_count}")
    
    st.subheader("Clean Data Sample")
    st.dataframe(df_clean.head())

# ==========================================
# PAGE 2: VISUALIZATION
# ==========================================
elif page == "Visualization":
    st.title("Data Visualization & Key Metrics")
    
    st.sidebar.subheader("Filters")
    selected_year = st.sidebar.multiselect("Select Year of Study:", df_clean['Year'].unique(), default=df_clean['Year'].unique())
    selected_gender = st.sidebar.multiselect("Select Gender (0: Male, 1: Female):", df_clean['Gender'].unique(), default=df_clean['Gender'].unique())
    
    df_filtered = df_clean[(df_clean['Year'].isin(selected_year)) & (df_clean['Gender'].isin(selected_gender))]
    
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
    
    st.markdown("---")
    
    # --- PIVOT BAR CHARTS TỪ EXCEL ---
    st.subheader("Average Analytics")

    col_pivot1, col_pivot2 = st.columns(2)

    

    with col_pivot2:
        st.markdown("**2. Average Academic Stress Level by GPA Rating**")
        pivot2 = df_filtered.groupby('GPA_Rating')['Academic_Stress_Level'].mean().reset_index()
        fig_p2 = px.bar(pivot2, x='GPA_Rating', y='Academic_Stress_Level', 
                        color_discrete_sequence=['#FF9800'], text_auto='.2f')
        fig_p2.update_layout(xaxis_title="GPA Rating (1 = Poor to 5 = Excellent)", yaxis_title="Avg Academic Stress Level")
        st.plotly_chart(fig_p2, use_container_width=True)
        
    st.markdown("---")
    
    # --- SCATTER CHARTS TRỰC TIẾP TỪ SHEET DASHBOARD ---
    st.subheader("Relationships Analysis (Derived from Excel DASHBOARD)")
    st.markdown("Exploring the correlations using scatter plots with OLS trendlines.")
    
    col_scat1, col_scat2 = st.columns(2)
    
    

    with col_scat2:
        fig_s2 = px.scatter(df_filtered, x='Daytime_Fatigue', y='Academic_Burnout_Score',
                            trendline='ols', opacity=0.5,
                            title="Daytime Fatigue vs Academic Burnout Score",
                            color_discrete_sequence=['#e74c3c'])
        st.plotly_chart(fig_s2, use_container_width=True)

    

    st.markdown("---")
    
    # --- HISTOGRAMS (CODE DO BẠN CUNG CẤP) ---
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
        st.subheader("Average Sleep Hours by Study Year")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        avg_sleep_by_year = df_filtered.groupby('Year')['Sleep_Hours_Total'].mean().reset_index()
        sns.barplot(x='Year', y='Sleep_Hours_Total', data=avg_sleep_by_year, palette='viridis', ax=ax2)
        ax2.set_xlabel("Year of Study")
        ax2.set_ylabel("Average Sleep Hours")
        st.pyplot(fig2)

    st.markdown("---")
    
    st.subheader("Academic Stress Level across GPA Ratings")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='GPA_Rating', y='Academic_Stress_Level', data=df_filtered, palette='viridis', ax=ax3)
    ax3.set_xlabel('GPA Rating (1 = Poor to 5 = Excellent)')
    ax3.set_ylabel('Academic Stress Level (0 = No Stress to 3 = Extremely High)')
    st.pyplot(fig3)

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

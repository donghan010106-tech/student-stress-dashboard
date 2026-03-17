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
    # Features selection based on your logic
    features_model = ['Academic_Burnout_Score', 'Sleep_Hygiene_Risk', 'Overall_Sleep_Quality', 'Exercise_Frequency']
    X = df[features_model]
    y = df['Academic_Stress_Level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # OLS Model
    ols_model = LinearRegression()
    ols_model.fit(X_train, y_train)
    ols_pred = ols_model.predict(X_test)
    
    # Random Forest Model
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    return X, X_test, y_test, ols_pred, rf_pred, rf_model

X_data, X_test, y_test, ols_pred, rf_pred, rf_model = train_models(df_clean)

# --- NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["🧹 Data Preprocessing & EDA", "📊 Visualization", "🤖 Model Training"])

# ==========================================
# PAGE 1: PREPROCESSING
# ==========================================
if page == "🧹 Data Preprocessing & EDA":
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
elif page == "📊 Visualization":
    st.title("Data Visualization")
    
    # 4 KPI Metrics
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Students", len(df_clean))
    k2.metric("Avg Stress Level", f"{df_clean['Academic_Stress_Level'].mean():.2f}")
    k3.metric("Avg GPA Rating", f"{df_clean['GPA_Rating'].mean():.1f}")
    k4.metric("Avg Sleep Risk", f"{df_clean['Sleep_Hygiene_Risk'].mean():.1f}")
    
    st.markdown("---")
    
    # Stress Level Pie Chart
    st.subheader("Academic Stress Level Distribution")
    stress_map = {0: 'No stress', 1: 'Low stress', 2: 'High stress', 3: 'Extremely high stress'}
    pie_data = df_clean['Academic_Stress_Level'].map(stress_map).value_counts().reset_index()
    pie_data.columns = ['Level', 'Count']
    fig_pie = px.pie(pie_data, values='Count', names='Level', hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    
    # Scatter Plots from Dashboard
    st.subheader("Relationship Analysis")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.plotly_chart(px.scatter(df_clean, x='Phone_Usage_Before_Sleep', y='Sleep_Hygiene_Risk', trendline='ols', title="Phone vs Sleep Risk"), use_container_width=True)
    with col_s2:
        st.plotly_chart(px.scatter(df_clean, x='Daytime_Fatigue', y='Academic_Burnout_Score', trendline='ols', title="Fatigue vs Burnout"), use_container_width=True)

    st.markdown("---")
    
    # Boxplot GPA
    st.subheader("Academic Stress Level across GPA Ratings")
    fig_box, ax_box = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='GPA_Rating', y='Academic_Stress_Level', data=df_clean, palette='viridis', ax=ax_box)
    st.pyplot(fig_box)

# ==========================================
# PAGE 3: MODEL TRAINING
# ==========================================
elif page == "🤖 Model Training":
    st.title("Model Training & Evaluation")
    
    # 1. Scatter Plots
    st.subheader("1. True vs Predicted Stress Levels")
    fig_scat, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(x=y_test, y=ols_pred, ax=axes[0], color="blue").set_title(f"OLS (R2: {r2_score(y_test, ols_pred):.3f})")
    sns.scatterplot(x=y_test, y=rf_pred, ax=axes[1], color="green").set_title(f"Random Forest (R2: {r2_score(y_test, rf_pred):.3f})")
    for ax in axes: ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    st.pyplot(fig_scat)
    
    st.markdown("---")
    
    # 2. KDE & Feature Importance (FIXED NAMES HERE)
    st.subheader("2. Distribution & Feature Importance")
    fig_final = plt.figure(figsize=(16, 6))
    
    # Subplot 1: Importance
    plt.subplot(1, 2, 1)
    pd.Series(rf_model.feature_importances_, index=X_data.columns).sort_values().plot(kind='barh', color='#34495e')
    plt.title('Feature Importance (Random Forest)')

    # Subplot 2: KDE Comparison (Sử dụng đúng tên biến ols_pred và rf_pred)
    plt.subplot(1, 2, 2)
    sns.kdeplot(y_test, label='Actual (Test)', color='black', linewidth=3)
    sns.kdeplot(ols_pred, label='Predicted (OLS)', color='red', linestyle='--')
    sns.kdeplot(rf_pred, label='Predicted (Random Forest)', color='green')
    plt.title('Actual vs. Predicted Stress Level Distributions')
    plt.legend()
    
    plt.tight_layout()
    st.pyplot(fig_final)

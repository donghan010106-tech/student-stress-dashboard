import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- PAGE CONFIGURATION ---
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

# --- MODEL TRAINING (CACHED) ---
@st.cache_resource
def train_models(df):
    cols_to_drop = ['Record_Time', 'Stress_Predicted'] 
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    y = df['Stress_Predicted']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    ols_model = LinearRegression()
    ols_model.fit(X_train, y_train)
    ols_pred = ols_model.predict(X_test)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    return X, X_test, y_test, ols_pred, rf_pred, ols_model, rf_model

X_data, X_test, y_test, ols_pred, rf_pred, ols_model, rf_model = train_models(df_clean)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["🧹 Data Preprocessing & EDA", 
     "📊 Visualization", 
     "🤖 Model Training"]
)

# ==========================================
# PAGE 1: DATA PREPROCESSING & EDA
# ==========================================
if page == "🧹 Data Preprocessing & EDA":
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
# PAGE 2: VISUALIZATION (UPDATED)
# ==========================================
elif page == "📊 Visualization":
    st.title("Data Visualization & Key Metrics")
    
    st.sidebar.subheader("Filters")
    selected_year = st.sidebar.multiselect("Select Year of Study:", df_clean['Year'].unique(), default=df_clean['Year'].unique())
    selected_gender = st.sidebar.multiselect("Select Gender (0: Male, 1: Female):", df_clean['Gender'].unique(), default=df_clean['Gender'].unique())
    
    df_filtered = df_clean[(df_clean['Year'].isin(selected_year)) & (df_clean['Gender'].isin(selected_gender))]
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Average Sleep Hours", f"{df_filtered['Sleep_Hours_Total'].mean():.1f} hrs")
    kpi2.metric("Avg Academic Burnout Score", f"{df_filtered['Academic_Burnout_Score'].mean():.1f}")
    kpi3.metric("Avg Predicted Stress", f"{df_filtered['Stress_Predicted'].mean():.2f}")
    
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
        
    # --- ĐOẠN CODE BỔ SUNG: DISTRIBUTION CỦA SLEEP HYGIENE & BURNOUT SCORE ---
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
    # -------------------------------------------------------------------------
    
    st.markdown("---")

    st.subheader("Academic Stress Level across GPA Ratings")
    st.markdown("This boxplot shows how academic stress levels vary across different GPA ratings (1 = Poor to 5 = Excellent).")
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='GPA_Rating', y='Academic_Stress_Level', data=df_filtered, palette='viridis', ax=ax3)
    ax3.set_xlabel('GPA Rating (1 = Poor to 5 = Excellent)')
    ax3.set_ylabel('Academic Stress Level (0 = No Stress to 3 = Extremely High)')
    st.pyplot(fig3)

# ==========================================
# PAGE 3: MODEL TRAINING
# ==========================================
elif page == "🤖 Model Training":
    st.title("Model Training & Evaluation")
    st.markdown("Evaluating the performance of **Ordinary Least Squares (OLS)** and **Random Forest Regressor**.")
    
    st.subheader("1. True vs Predicted Stress Levels")
    fig_pred, axes_pred = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.scatterplot(x=y_test, y=ols_pred, ax=axes_pred[0], color="dodgerblue", alpha=0.7)
    axes_pred[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes_pred[0].set_title(f'OLS (R² = {r2_score(y_test, ols_pred):.4f})', fontsize=14, fontweight='bold')
    axes_pred[0].set_xlabel('True Stress')
    axes_pred[0].set_ylabel('Predicted Stress')
    
    sns.scatterplot(x=y_test, y=rf_pred, ax=axes_pred[1], color="mediumseagreen", alpha=0.7)
    axes_pred[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes_pred[1].set_title(f'Random Forest (R² = {r2_score(y_test, rf_pred):.4f})', fontsize=14, fontweight='bold')
    axes_pred[1].set_xlabel('True Stress')
    axes_pred[1].set_ylabel('Predicted Stress')
    
    plt.tight_layout()
    st.pyplot(fig_pred)
    
    st.markdown("---")
    
    st.subheader("2. Feature Importance & Distribution Comparison")
    
    fig_combined = plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    importances = rf_model.feature_importances_
    feat_importances = pd.Series(importances, index=X_data.columns).sort_values()
    feat_importances.plot(kind='barh', color='#34495e')
    plt.title('Feature Importance (Optimized Random Forest)')

    plt.subplot(1, 2, 2)
    sns.kdeplot(y_test, label='Actual (Test)', color='black', linewidth=3)
    sns.kdeplot(ols_pred, label='Predicted (OLS)', color='red', linestyle='--')
    sns.kdeplot(rf_pred, label='Predicted (Random Forest)', color='green')
    plt.title('Actual vs. Predicted Stress Level Distributions')
    plt.xlabel('Academic Stress Level')
    plt.legend()

    plt.tight_layout()
    st.pyplot(fig_combined)
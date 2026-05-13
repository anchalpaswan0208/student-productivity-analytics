# ==================================================
# Student Productivity Analytics System
# Streamlit Application
# ==================================================

# ==================================================
# Import Libraries
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="Student Productivity Analytics",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# Title Section
# ==================================================

st.title("📊 Student Digital Behavior & Productivity Intelligence System")

st.markdown("""
Analyze student productivity, burnout risk,
wellness, and performance using behavioral analytics.
""")

# ==================================================
# Sidebar
# ==================================================

st.sidebar.header("Dashboard Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset CSV",
    type=["csv"]
)

# ==================================================
# Load Dataset
# ==================================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

else:

    st.warning(
        "Using sample dataset. Upload your own dataset for analysis."
    )

    df = pd.read_csv(
        "dashboard/dashboard_dataset.csv"
    )

# ==================================================
# Dataset Preview
# ==================================================

st.subheader("Dataset Preview")

st.dataframe(df.head())

# ==================================================
# KPI SECTION
# ==================================================

st.subheader("Executive KPIs")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Average Exam Score",
        round(df["Exam_Score"].mean(), 2)
    )

with col2:

    st.metric(
        "Average Productivity",
        round(df["Productivity_Score"].mean(), 2)
    )

with col3:

    st.metric(
        "Average Burnout Risk",
        round(df["Burnout_Risk_Score"].mean(), 2)
    )

with col4:

    high_risk_students = len(
        df[df["High_Risk_Flag"] == 1]
    )

    st.metric(
        "High Risk Students",
        high_risk_students
    )

# ==================================================
# FILTER SECTION
# ==================================================

st.sidebar.subheader("Filters")

persona_filter = st.sidebar.multiselect(
    "Select Persona",
    options=df["Persona"].unique(),
    default=df["Persona"].unique()
)

filtered_df = df[
    df["Persona"].isin(persona_filter)
]

# ==================================================
# PRODUCTIVITY ANALYSIS
# ==================================================

st.subheader("📈 Productivity Analysis")

fig, ax = plt.subplots(figsize=(10,6))

sns.scatterplot(
    data=filtered_df,
    x="Study_Hours",
    y="Exam_Score",
    hue="Student_Segment",
    ax=ax
)

plt.title("Study Hours vs Exam Score")

st.pyplot(fig)

# ==================================================
# BURNOUT ANALYSIS
# ==================================================

st.subheader("🔥 Burnout Analysis")

fig, ax = plt.subplots(figsize=(10,6))

sns.histplot(
    filtered_df["Burnout_Risk_Score"],
    bins=30,
    kde=True,
    ax=ax
)

plt.title("Burnout Risk Distribution")

st.pyplot(fig)

# ==================================================
# WELLNESS ANALYSIS
# ==================================================

st.subheader("💤 Sleep vs Productivity")

fig, ax = plt.subplots(figsize=(10,6))

sns.scatterplot(
    data=filtered_df,
    x="Sleep_Hours",
    y="Productivity_Score",
    hue="Burnout_Risk",
    ax=ax
)

plt.title("Sleep vs Productivity")

st.pyplot(fig)

# ==================================================
# SEGMENT ANALYSIS
# ==================================================

st.subheader("🧠 Student Segments")

segment_summary = filtered_df.groupby(
    "Student_Segment"
)[[
    "Exam_Score",
    "Productivity_Score",
    "Burnout_Risk_Score"
]].mean()

st.dataframe(segment_summary)

# ==================================================
# MACHINE LEARNING SECTION
# ==================================================

st.subheader("🤖 Burnout Prediction Model")

features = [
    "Study_Hours",
    "Sleep_Hours",
    "Screen_Time_Hours",
    "Stress_Level",
    "Mood_Score",
    "Productivity_Score",
    "Digital_Distraction_Score"
]

X = df[features]

y = df["High_Risk_Flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==================================================
# USER INPUTS
# ==================================================

st.sidebar.subheader("Student Input for Prediction")

study_hours = st.sidebar.slider(
    "Study Hours",
    0.0,
    12.0,
    5.0
)

sleep_hours = st.sidebar.slider(
    "Sleep Hours",
    0.0,
    10.0,
    6.0
)

screen_time = st.sidebar.slider(
    "Screen Time",
    0.0,
    15.0,
    5.0
)

stress_level = st.sidebar.slider(
    "Stress Level",
    0,
    10,
    5
)

mood_score = st.sidebar.slider(
    "Mood Score",
    0,
    10,
    5
)

productivity_score = st.sidebar.slider(
    "Productivity Score",
    0,
    10,
    5
)

distraction_score = st.sidebar.slider(
    "Distraction Score",
    0.0,
    15.0,
    5.0
)

# ==================================================
# PREDICTION
# ==================================================

input_data = pd.DataFrame({

    "Study_Hours": [study_hours],

    "Sleep_Hours": [sleep_hours],

    "Screen_Time_Hours": [screen_time],

    "Stress_Level": [stress_level],

    "Mood_Score": [mood_score],

    "Productivity_Score": [productivity_score],

    "Digital_Distraction_Score": [distraction_score]

})

prediction = model.predict(input_data)[0]

# ==================================================
# DISPLAY PREDICTION
# ==================================================

if prediction == 1:

    st.error("⚠️ High Burnout Risk Detected")

else:

    st.success("✅ Low Burnout Risk")

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.markdown("""
Created as part of the
**Student Digital Behavior & Productivity Intelligence System**
portfolio project.
""")
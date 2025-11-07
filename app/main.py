import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🪐 Asteroid Classification Project")
st.subheader("About the Project")
st.write("""
The **Data-Driven Classification of Hazardous Asteroids** project leverages data analytics and machine learning to
predict whether an asteroid is potentially hazardous to Earth.
Below are some visual insights from the processed dataset.
""")

# Load processed data
@st.cache_data
def load_data():
    df = pd.read_csv('../data/processed/X_train_scaled.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv')
    df['Hazardous'] = y_train
    return df

df = load_data()

st.markdown("### 📊 Data Overview")
st.write(df.head())

# 1. Hazardous vs Non-Hazardous Distribution
st.markdown("### ☄️ Hazardous vs Non-Hazardous Asteroids")
fig1, ax1 = plt.subplots()
df['Hazardous'].value_counts().plot(kind='pie', autopct='%1.1f%%', labels=['Non-Hazardous', 'Hazardous'], ax=ax1, colors=['#66b3ff','#ff9999'])
ax1.set_ylabel('')
st.pyplot(fig1)

# 2. Feature Distribution (example)
st.markdown("### 🚀 Feature Distribution (First Numeric Feature)")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
if len(numeric_cols) > 0:
    feature = st.selectbox("Select a feature to visualize:", numeric_cols)
    fig2, ax2 = plt.subplots()
    sns.histplot(df[feature], bins=20, kde=True, ax=ax2)
    st.pyplot(fig2)

# 3. Correlation Heatmap
st.markdown("### 🔗 Feature Correlation Heatmap")
fig3, ax3 = plt.subplots(figsize=(10,6))
sns.heatmap(df.corr(), cmap="coolwarm", ax=ax3)
st.pyplot(fig3)


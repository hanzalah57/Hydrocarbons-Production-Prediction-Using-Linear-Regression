import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Streamlit Page Config
st.set_page_config(layout="wide")

# App Title
st.markdown(
    """
    <h1 style='text-align: left; color: #A86631; font-size: 40px; font-family: Arial, sans-serif;'>
        Hydrocarbons Production Prediction Using Linear Regression
    </h1>
    """, 
    unsafe_allow_html=True
)

# Sidebar Navigation
page = st.sidebar.selectbox("Choose a section", ["Data Overview", "Prediction"])

# Load Data
df = pd.read_excel('Production_Data.xlsx')

if page == "Data Overview":
    st.write("### Data Preview")
    st.dataframe(df.drop(df.columns[0], axis=1)) 

    st.write("### Data Statistics")
    st.dataframe(df.describe())

elif page == "Prediction":
    st.write("### Predicted Production")

    # Define Features and Target
    X = df[['Por', 'Perm', 'AI', 'Brittle', 'TOC', 'VR']]
    y = df['Prod']

    # Standardizing Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Model
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Sidebar Input Features
    st.sidebar.header("🔍 Input Features")
    Porosity = st.sidebar.slider("Porosity (%)", float(df['Por'].min()), float(df['Por'].max()), key="porosity")
    Permeability = st.sidebar.slider("Permeability (mD)", float(df['Perm'].min()), float(df['Perm'].max()), key="permeability")
    AI = st.sidebar.slider("Acoustic Impedance", float(df['AI'].min()), float(df['AI'].max()), key="ai")
    Brittle = st.sidebar.slider("Brittleness", float(df['Brittle'].min()), float(df['Brittle'].max()), key="brittle")
    TOC = st.sidebar.slider("Total Organic Carbon", float(df['TOC'].min()), float(df['TOC'].max()), key="toc")
    VR = st.sidebar.slider("Vitrinite Reflectance", float(df['VR'].min()), float(df['VR'].max()), key="vr")

    # Prepare Input Data for Prediction
    input_data = np.array([[Porosity, Permeability, AI, Brittle, TOC, VR]])
    input_data_scaled = scaler.transform(input_data)  

    # Prediction
    prediction = model.predict(input_data_scaled)[0]

    # Display Prediction
    st.markdown(
    f"""
    <div style="
        padding: 20px; 
        border-radius: 12px; 
        background: linear-gradient(135deg, #A86631, #7a4f24);
        color: white; 
        text-align: center; 
        font-size: 27px; 
        font-weight: bold;
        box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.3);
        border: 2px solid #fff;
        max-width: 500px;
        margin: auto;
    ">
        <span style="font-size: 32px; font-weight: bold;">{prediction:.2f} barrels/day</span>
    </div>
    """,
    unsafe_allow_html=True
    )

    #Feature Contribution Bar Chart
    coefficients = model.coef_
    features = ['Porosity', 'Permeability', 'Acoustic Impedance', 'Brittleness', 'TOC', 'VR']
    contributions = [coeff * value for coeff, value in zip(coefficients, input_data[0])]

    contribution_df = pd.DataFrame({'Feature': features, 'Contribution': contributions})
    fig1 = px.bar(contribution_df, x='Feature', y='Contribution', 
                  title="Feature Contribution to Production", text_auto='.2f', color='Contribution')

    st.plotly_chart(fig1)

    #Prediction Trend vs. Porosity
    porosity_values = np.linspace(df['Por'].min(), df['Por'].max(), 50)
    trend_input = np.array([[p, Permeability, AI, Brittle, TOC, VR] for p in porosity_values])
    trend_input_scaled = scaler.transform(trend_input)
    predictions = model.predict(trend_input_scaled)

    trend_df = pd.DataFrame({'Porosity': porosity_values, 'Predicted Production': predictions})
    fig2 = px.line(trend_df, x='Porosity', y='Predicted Production', 
                   title="Predicted Production vs. Porosity", markers=True)

    st.plotly_chart(fig2)

    #Residual Plot (Actual vs. Predicted)
    y_pred = model.predict(X_scaled)
    residuals_df = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
    fig3 = px.scatter(residuals_df, x='Actual', y='Predicted', 
                      title="Actual vs. Predicted Production",
                      labels={'Actual': 'Actual Production', 'Predicted': 'Predicted Production'})
    fig3.add_shape(type='line', x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color='red', dash='dash'))

    st.plotly_chart(fig3)

# About Section
st.sidebar.info("""
This app predicts hydrocarbon production using a **Linear Regression Model**.  
The model is trained on an **open-source dataset** from the **GitHub repository** of Dr. Michael Pyrcz,  
a **petroleum engineering professor at the University of Texas at Austin**.
""")

st.sidebar.markdown("---")  
st.sidebar.markdown("👤 **Author: Mr. Hanzalah Bin Sohail**  \n🌏*Geophysicist*")  

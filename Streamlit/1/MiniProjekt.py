import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

st.header("School Shooting project by Ahmad & Hanni")

st.header("Step 1: Upload Your Data")
st.write("To begin, upload the CSV file containing school shooting data.")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show raw data
    st.subheader("Step 2: Data Preview")

    st.dataframe(df.head(10))
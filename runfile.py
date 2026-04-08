import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from ydata_profiling import ProfileReport
    from streamlit_pandas_profiling import st_profile_report
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False

st.set_page_config(page_title="CSV Evaluator", layout="wide")

st.title("Generic CSV Evaluation App")
st.write("Upload any CSV file for summary, exploration, and profiling.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.success("File uploaded successfully.")

        st.subheader("Dataset Preview")
        st.dataframe(df)

        st.subheader("Dataset Shape")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")

        st.subheader("Column Information")
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Missing %": ((df.isnull().sum() / len(df)) * 100).round(2).values,
            "Unique Values": df.nunique().values
        })
        st.dataframe(col_info)

        st.subheader("Duplicate Rows")
        st.write(f"Number of duplicate rows: {df.duplicated().sum()}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            st.subheader("Numeric Summary Statistics")
            st.dataframe(df[numeric_cols].describe().T)

            st.subheader("Basic Visualisation")
            selected_col = st.selectbox("Choose a numeric column to plot", numeric_cols)

            fig, ax = plt.subplots()
            ax.hist(df[selected_col].dropna(), bins=20)
            ax.set_title(f"Histogram of {selected_col}")
            ax.set_xlabel(selected_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        else:
            st.info("No numeric columns found for plotting.")

        st.subheader("Missing Values Summary")
        missing_df = pd.DataFrame({
            "Column": df.columns,
            "Missing Values": df.isnull().sum().values
        })
        st.dataframe(missing_df)

        st.subheader("Generate Profiling Report")
        if PROFILING_AVAILABLE:
            if st.button("Create Profiling Report"):
                with st.spinner("Generating report..."):
                    profile = ProfileReport(df, explorative=True)
                    st_profile_report(profile)
        else:
            st.warning(
                "ydata-profiling is not installed. Add 'ydata-profiling' and "
                "'streamlit-pandas-profiling' to requirements.txt."
            )

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Please upload a CSV file to begin.")

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Generic CSV Explorer", layout="wide")

ENCODINGS = ("utf-8", "latin1", "cp1252")


def read_csv_flexible(file_or_path, encoding_candidates=ENCODINGS):
    """Read a CSV from an uploaded file or local path with a few safe fallbacks."""
    last_error = None

    if hasattr(file_or_path, "getvalue"):
        raw_bytes = file_or_path.getvalue()
        for enc in encoding_candidates:
            try:
                return pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
            except Exception as exc:
                last_error = exc
        raise last_error

    path = Path(file_or_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    for enc in encoding_candidates:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last_error = exc
    raise last_error


@st.cache_data(show_spinner=False)
def load_uploaded_csv(uploaded_file):
    return read_csv_flexible(uploaded_file)


@st.cache_data(show_spinner=False)
def load_local_csv(local_path: str):
    return read_csv_flexible(local_path)


def build_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing_values": df.isnull().sum().values,
            "missing_pct": (df.isnull().mean() * 100).round(2).values,
            "non_null_values": df.notnull().sum().values,
            "unique_values": [df[c].nunique(dropna=True) for c in df.columns],
        }
    )


def make_filter_mask(df: pd.DataFrame, column: str):
    """Return a boolean mask based on Streamlit controls for a chosen column."""
    if pd.api.types.is_numeric_dtype(df[column]):
        non_null = df[column].dropna()
        if non_null.empty:
            st.info("This numeric column only has missing values.")
            return pd.Series(True, index=df.index)

        min_val = float(non_null.min())
        max_val = float(non_null.max())

        if min_val == max_val:
            st.info("This numeric column has a single constant value, so no range filter is needed.")
            return pd.Series(True, index=df.index)

        selected_range = st.slider(
            "Select numeric range",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
        )
        return df[column].between(selected_range[0], selected_range[1], inclusive="both")

    unique_values = df[column].dropna().astype(str).unique().tolist()
    if 0 < len(unique_values) <= 200:
        selected_values = st.multiselect("Select values", sorted(unique_values))
        if selected_values:
            return df[column].astype(str).isin(selected_values)
        return pd.Series(True, index=df.index)

    contains_text = st.text_input("Contains text")
    if contains_text:
        return df[column].astype(str).str.contains(contains_text, case=False, na=False)
    return pd.Series(True, index=df.index)


st.title("Generic CSV Explorer")
st.write(
    "Upload any CSV file and explore its structure, quality, summaries, and variable relationships interactively."
)

st.sidebar.header("Data source")
source_type = st.sidebar.radio("Choose input method", ["Upload CSV", "Local path"], index=0)

uploaded_file = None
local_path = ""

if source_type == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
else:
    st.sidebar.warning(
        "Local paths work only on the machine hosting the app. On GitHub/Streamlit Cloud, use Upload CSV instead."
    )
    local_path = st.sidebar.text_input("Enter local CSV path")

if source_type == "Upload CSV" and uploaded_file is None:
    st.info("Please upload a CSV file in the sidebar to begin.")
    st.stop()

if source_type == "Local path" and not local_path.strip():
    st.info("Please enter a valid local CSV path in the sidebar to begin.")
    st.stop()

try:
    if source_type == "Upload CSV":
        df = load_uploaded_csv(uploaded_file)
        data_name = uploaded_file.name
    else:
        df = load_local_csv(local_path.strip())
        data_name = Path(local_path.strip()).name
except Exception as exc:
    st.error(f"Could not load the CSV file: {exc}")
    st.stop()

numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_columns = [c for c in df.columns if c not in numeric_columns]
all_columns = df.columns.tolist()

with st.sidebar:
    st.header("Optional filter")
    filter_column = st.selectbox("Choose a column to filter", ["None"] + all_columns, index=0)

if filter_column != "None":
    filter_mask = make_filter_mask(df, filter_column)
    filtered_df = df[filter_mask].copy()
else:
    filtered_df = df.copy()

st.success(f"Loaded {data_name} with {df.shape[0]:,} rows and {df.shape[1]:,} columns.")
if len(filtered_df) != len(df):
    st.info(f"Current filtered view: {len(filtered_df):,} rows.")

rows = len(df)
cols = len(df.columns)
missing_total = int(df.isnull().sum().sum())
duplicate_total = int(df.duplicated().sum())

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Rows", f"{rows:,}")
metric_2.metric("Columns", f"{cols:,}")
metric_3.metric("Missing values", f"{missing_total:,}")
metric_4.metric("Duplicate rows", f"{duplicate_total:,}")

tab_overview, tab_quality, tab_relationships, tab_downloads = st.tabs(
    ["Overview", "Quality", "Relationships", "Downloads"]
)

with tab_overview:
    st.subheader("Dataset preview")
    preview_default = min(10, max(len(filtered_df), 5))
    preview_max = min(100, max(len(filtered_df), 5))
    preview_rows = st.slider("Rows to preview", min_value=5, max_value=preview_max, value=preview_default)
    st.dataframe(filtered_df.head(preview_rows), use_container_width=True)

    st.subheader("Column summary")
    st.dataframe(build_column_summary(filtered_df), use_container_width=True)

    if numeric_columns:
        st.subheader("Numeric summary statistics")
        st.dataframe(filtered_df[numeric_columns].describe().T, use_container_width=True)
    else:
        st.info("No numeric columns were found in this dataset.")

with tab_quality:
    st.subheader("Missing values overview")
    missing_df = build_column_summary(filtered_df)[["column", "missing_values", "missing_pct"]]
    st.dataframe(
        missing_df.sort_values(["missing_values", "column"], ascending=[False, True]),
        use_container_width=True,
    )

    quality_col1, quality_col2 = st.columns(2)
    with quality_col1:
        st.subheader("Duplicate rows")
        st.write(f"Duplicate rows in current view: {int(filtered_df.duplicated().sum()):,}")

    with quality_col2:
        st.subheader("Data types")
        dtype_df = pd.DataFrame({"column": filtered_df.columns, "dtype": filtered_df.dtypes.astype(str).values})
        st.dataframe(dtype_df, use_container_width=True)

    if numeric_columns:
        st.subheader("Single-variable charts")
        chart_col1, chart_col2, chart_col3 = st.columns(3)
        chart_type = chart_col1.selectbox("Chart type", ["Histogram", "Box", "Violin"])
        selected_column = chart_col2.selectbox("Numeric column", numeric_columns)
        bins = chart_col3.slider("Bins", min_value=5, max_value=100, value=30)

        if chart_type == "Histogram":
            fig = px.histogram(filtered_df, x=selected_column, nbins=bins)
        elif chart_type == "Box":
            fig = px.box(filtered_df, y=selected_column)
        else:
            fig = px.violin(filtered_df, y=selected_column, box=True)

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

with tab_relationships:
    st.subheader("Interactive relationships explorer")

    if len(numeric_columns) >= 2:
        rel_col1, rel_col2, rel_col3, rel_col4, rel_col5 = st.columns(5)
        x_col = rel_col1.selectbox("X-axis", numeric_columns, index=0)
        y_col = rel_col2.selectbox("Y-axis", numeric_columns, index=min(1, len(numeric_columns) - 1))
        color_options = ["None"] + all_columns
        size_options = ["None"] + numeric_columns
        color_col = rel_col3.selectbox("Color", color_options, index=0)
        size_col = rel_col4.selectbox("Size", size_options, index=0)
        chart_type = rel_col5.selectbox("Chart type", ["Scatter", "Line", "Density heatmap"])

        rel_opts1, rel_opts2 = st.columns(2)
        add_trendline = rel_opts1.checkbox("Add trendline to scatter", value=False)
        max_points = rel_opts2.slider(
            "Max plotted rows",
            min_value=100,
            max_value=min(max(len(filtered_df), 100), 10000),
            value=min(max(len(filtered_df), 1000), 5000),
            step=100,
        )

        plot_df = filtered_df.dropna(subset=[x_col, y_col]).copy()
        if len(plot_df) > max_points:
            plot_df = plot_df.sample(max_points, random_state=42)

        common_kwargs = {
            "data_frame": plot_df,
            "x": x_col,
            "y": y_col,
            "hover_data": all_columns,
        }

        if color_col != "None":
            common_kwargs["color"] = color_col
        if size_col != "None" and chart_type == "Scatter":
            common_kwargs["size"] = size_col

        if chart_type == "Scatter":
            if add_trendline:
                fig = px.scatter(**common_kwargs, trendline="ols")
            else:
                fig = px.scatter(**common_kwargs)
        elif chart_type == "Line":
            fig = px.line(**common_kwargs)
        else:
            fig = px.density_heatmap(**common_kwargs)

        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Switch X and Y axes, add color or size, and inspect variable relationships interactively."
        )
    else:
        st.info("At least two numeric columns are needed for the interactive relationships explorer.")

    st.subheader("Correlation matrix")
    if len(numeric_columns) >= 2:
        corr_df = filtered_df[numeric_columns].corr(numeric_only=True)
        fig = px.imshow(corr_df, text_auto=True, aspect="auto")
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("At least two numeric columns are needed for a correlation matrix.")

    if non_numeric_columns:
        st.subheader("Category exploration")
        cat_col1, cat_col2 = st.columns(2)
        category_column = cat_col1.selectbox("Categorical column", non_numeric_columns)
        top_n = cat_col2.slider("Top categories to show", min_value=5, max_value=30, value=10)
        counts = filtered_df[category_column].astype(str).value_counts(dropna=False).head(top_n).reset_index()
        counts.columns = [category_column, "count"]
        fig = px.bar(counts, x=category_column, y="count")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

with tab_downloads:
    st.subheader("Download current data")
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download current filtered data as CSV",
        data=csv_data,
        file_name="filtered_data.csv",
        mime="text/csv",
    )

    st.subheader("Quick notes")
    st.write(
        "- This app is generic and does not assume fixed column names.\n"
        "- Upload CSV is the best option when using GitHub + Streamlit Cloud.\n"
        "- The Relationships tab replaces the earlier profiling-style interaction."
    )

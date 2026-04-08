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

st.success(f"Loaded {data_name} with {df.shape[0]:,} rows and {df.shape[1]:,} columns.")

with st.expander("Preview and structure", expanded=True):
    preview_default = min(10, max(len(df), 5))
    preview_max = min(100, max(len(df), 5))
    preview_rows = st.slider("Rows to preview", min_value=5, max_value=preview_max, value=preview_default)
    st.dataframe(df.head(preview_rows), use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", f"{len(df.columns):,}")
    col3.metric("Missing values", f"{int(df.isnull().sum().sum()):,}")
    col4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    st.subheader("Column summary")
    st.dataframe(build_column_summary(df), use_container_width=True)

numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = [c for c in df.columns if c not in numeric_columns]
all_columns = df.columns.tolist()
filtered_df = df.copy()

if numeric_columns:
    with st.expander("Numeric summary", expanded=False):
        st.dataframe(df[numeric_columns].describe().T, use_container_width=True)

with st.expander("Missing values overview", expanded=False):
    missing_df = build_column_summary(df)[["column", "missing_values", "missing_pct"]]
    st.dataframe(missing_df.sort_values(["missing_values", "column"], ascending=[False, True]), use_container_width=True)

with st.expander("Filter data", expanded=False):
    filter_column = st.selectbox("Choose a column to filter", all_columns)

    if pd.api.types.is_numeric_dtype(df[filter_column]):
        non_null = df[filter_column].dropna()
        if not non_null.empty:
            min_val = float(non_null.min())
            max_val = float(non_null.max())
            if min_val == max_val:
                st.info("This numeric column has only one value, so no range filter is needed.")
            else:
                selected_range = st.slider(
                    "Select numeric range",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                )
                filtered_df = df[df[filter_column].between(selected_range[0], selected_range[1], inclusive="both")]
    else:
        unique_values = df[filter_column].dropna().astype(str).unique().tolist()
        if 0 < len(unique_values) <= 200:
            selected_values = st.multiselect("Select values", sorted(unique_values))
            if selected_values:
                filtered_df = df[df[filter_column].astype(str).isin(selected_values)]
        else:
            contains_text = st.text_input("Contains text")
            if contains_text:
                filtered_df = df[df[filter_column].astype(str).str.contains(contains_text, case=False, na=False)]

    st.write(f"Filtered rows: {len(filtered_df):,}")
    st.dataframe(filtered_df, use_container_width=True)

with st.expander("Interactive relationships explorer", expanded=True):
    if len(numeric_columns) >= 2:
        explorer_cols = st.columns(4)
        x_col = explorer_cols[0].selectbox("X-axis", numeric_columns, index=0)
        y_col = explorer_cols[1].selectbox("Y-axis", numeric_columns, index=min(1, len(numeric_columns) - 1))
        color_options = ["None"] + all_columns
        size_options = ["None"] + numeric_columns
        color_col = explorer_cols[2].selectbox("Color", color_options, index=0)
        size_col = explorer_cols[3].selectbox("Size", size_options, index=0)

        chart_cols = st.columns(3)
        chart_type = chart_cols[0].selectbox("Chart type", ["Scatter", "Line", "Hexbin-style density"])
        add_trendline = chart_cols[1].checkbox("Add trendline", value=False)
        max_points = chart_cols[2].slider("Max plotted rows", min_value=100, max_value=min(max(len(filtered_df), 100), 10000), value=min(max(len(filtered_df), 1000), 5000), step=100)

        plot_df = filtered_df[[c for c in set([x_col, y_col, color_col if color_col != "None" else None, size_col if size_col != "None" else None]) if c is not None]].copy()
        plot_df = filtered_df.copy() if plot_df.empty else filtered_df
        plot_df = plot_df.dropna(subset=[x_col, y_col])
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
            "Use the selectors above to switch variables and inspect relationships, similar to an exploratory profiling view."
        )
    else:
        st.info("At least two numeric columns are needed for the interactive relationships explorer.")

with st.expander("Single-variable charts", expanded=False):
    if numeric_columns:
        chart_type = st.selectbox("Choose chart type", ["Histogram", "Box", "Violin"])
        selected_column = st.selectbox("Choose a numeric column", numeric_columns)

        if chart_type == "Histogram":
            bins = st.slider("Number of bins", min_value=5, max_value=100, value=30)
            fig = px.histogram(filtered_df, x=selected_column, nbins=bins)
        elif chart_type == "Box":
            fig = px.box(filtered_df, y=selected_column)
        else:
            fig = px.violin(filtered_df, y=selected_column, box=True)

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns were found, so single-variable charts are not shown.")

with st.expander("Correlation matrix", expanded=False):
    if len(numeric_columns) >= 2:
        corr_df = filtered_df[numeric_columns].corr(numeric_only=True)
        fig = px.imshow(corr_df, text_auto=True, aspect="auto")
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("At least two numeric columns are needed for a correlation matrix.")

with st.expander("Download filtered data", expanded=False):
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download current data as CSV",
        data=csv_data,
        file_name="filtered_data.csv",
        mime="text/csv",
    )

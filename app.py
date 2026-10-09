from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error

DATA_PATH = Path(__file__).resolve().parent / "Rainfall_data.csv"


@st.cache_data
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
    df = df.sort_values("Date").set_index("Date")
    df = df.asfreq("MS")
    return df


@st.cache_data
def get_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    month_number = df.index.month.rename("Month")
    monthly = (
        df.groupby(month_number)
        .agg(
            Avg_Temperature=("Temperature", "mean"),
            Avg_Precipitation=("Precipitation", "mean"),
            Total_Precipitation=("Precipitation", "sum"),
        )
        .reset_index()
    )
    monthly["Month_Name"] = monthly["Month"].map(
        {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }
    )
    monthly = monthly[["Month_Name", "Avg_Temperature", "Avg_Precipitation", "Total_Precipitation"]]
    return monthly


@st.cache_data
def forecast_baseline(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df["Temperature"].copy()
    n_test = min(36, len(y) - 12)
    if n_test <= 0:
        empty = pd.DataFrame(index=y.index, columns=["Actual", "Seasonal Naive", "Climatology"])
        empty = empty.fillna(0)
        metrics = pd.DataFrame({"Metric": ["Seasonal Naive MAE", "Climatology MAE"], "Value": [0.0, 0.0]})
        return empty, metrics

    train, test = y[:-n_test], y[-n_test:]
    seasonal_naive = y.shift(12).iloc[-n_test:]
    climatology_map = train.groupby(train.index.month).mean()
    climatology = pd.Series(test.index.month.map(climatology_map), index=test.index)

    result = pd.DataFrame(
        {
            "Actual": test.values,
            "Seasonal Naive": seasonal_naive.values,
            "Climatology": climatology.values,
        },
        index=test.index,
    )

    metrics = pd.DataFrame(
        {
            "Metric": ["Seasonal Naive MAE", "Climatology MAE"],
            "Value": [
                mean_absolute_error(test, seasonal_naive),
                mean_absolute_error(test, climatology),
            ],
        }
    )
    return result, metrics


st.set_page_config(page_title="Weather Forecast Dashboard", page_icon="🌦️", layout="wide")


df = load_dataset()
month_summary = get_monthly_summary(df)
forecast_df, metrics_df = forecast_baseline(df)

start_year = int(df.index.year.min())
end_year = int(df.index.year.max())

st.title("Weather Forecast Dashboard")
st.caption("Monthly climate monitoring and forecast baseline for temperature and precipitation trends.")

with st.sidebar:
    st.header("Controls")
    year_start, year_end = st.slider(
        "Select year range",
        min_value=start_year,
        max_value=end_year,
        value=(start_year, end_year),
    )
    selected_metric = st.selectbox("Primary metric", ["Temperature", "Precipitation"])

filtered_df = df[
    (df.index.year >= year_start) & (df.index.year <= year_end)
].copy()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average temperature", f"{filtered_df['Temperature'].mean():.2f}°C")
col2.metric("Average precipitation", f"{filtered_df['Precipitation'].mean():.2f} mm")
col3.metric("Warmest month", filtered_df.loc[filtered_df["Temperature"].idxmax()].name.strftime("%b %Y"))
col4.metric("Wettest month", filtered_df.loc[filtered_df["Precipitation"].idxmax()].name.strftime("%b %Y"))

overview_tab, seasonality_tab, forecast_tab = st.tabs(["Overview", "Seasonality", "Forecast"])

with overview_tab:
    st.subheader("Historical trend")
    st.line_chart(filtered_df[["Temperature", "Precipitation"]])

    st.subheader("Climate summary")
    summary_table = filtered_df[["Temperature", "Precipitation"]].describe().T
    summary_table = summary_table.rename(columns={"count": "Count", "mean": "Mean", "std": "Std", "min": "Min", "50%": "Median", "max": "Max"})
    st.dataframe(summary_table, use_container_width=True)

with seasonality_tab:
    month_filtered = month_summary.copy()
    st.subheader("Average monthly temperature")
    st.bar_chart(month_filtered.set_index("Month_Name")["Avg_Temperature"])

    st.subheader("Average monthly precipitation")
    st.bar_chart(month_filtered.set_index("Month_Name")["Avg_Precipitation"])

    st.subheader("Seasonal table")
    st.dataframe(month_filtered, use_container_width=True)

with forecast_tab:
    st.subheader("Forecast baseline comparison")
    st.line_chart(forecast_df)

    st.subheader("Model accuracy")
    st.dataframe(metrics_df, use_container_width=True)

    st.info(
        "The forecast view uses a seasonal naive baseline and a climatology benchmark based on the historical monthly pattern, following the time-series methodology from the project notebook."
    )

st.markdown("---")
st.caption(f"Dataset range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")

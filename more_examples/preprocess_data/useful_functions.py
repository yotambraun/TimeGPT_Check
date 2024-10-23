import pandas as pd

def expand_data_with_zeros(df, end_date, freq="D"):
    new_df = pd.DataFrame()
    copy_df = df.copy()
    for unique_id_value in copy_df["unique_id"].unique():
        subset = copy_df[copy_df["unique_id"] == unique_id_value].copy()
        subset["ds"] = pd.to_datetime(subset["ds"],format="%d/%m/%Y")
        subset = subset.groupby("ds").sum().reset_index()  # Aggregate to remove duplicates
        max_date = subset["ds"].max()
        full_range = pd.date_range(start=subset["ds"].min(), end=max(max_date, end_date), freq=freq)
        subset = subset.set_index("ds").reindex(full_range).reset_index()
        subset = subset.rename(columns={"index": "ds"})
        subset = subset[subset["ds"] <= end_date]
        subset["unique_id"] = unique_id_value
        subset["y"] = subset["y"].fillna(0)
        new_df = pd.concat([new_df, subset])
    new_df = new_df.sort_values(by=["unique_id", "ds"])
    return new_df

def aggregate_by_frequency(df, freq="ME"):
    return df.groupby(["unique_id", pd.Grouper(key="ds", freq=freq)])["y"].sum().reset_index()
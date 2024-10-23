import pandas as pd
import matplotlib.pyplot as plt
from nixtla import NixtlaClient
from more_examples.config import DATASETTRAINPATH
from utils import load_and_prepare_data
from more_examples.utils import load_environment_variables


def detect_anomalies(client: NixtlaClient, df: pd.DataFrame) -> pd.DataFrame:
    """Detect anomalies in the time series data using the Nixtla TimeGPT API."""
    anomalies_df = client.detect_anomalies(
        df=df,
        time_col='ds',
        target_col='y',
        freq='ME',
    )
    return anomalies_df


def plot_anomalies(anomalies_df: pd.DataFrame, df_train: pd.DataFrame) -> None:
    """Plot the actual data and anomalies for each unique_id."""
    for unique_id in anomalies_df['unique_id'].unique():
        df_temp = anomalies_df[anomalies_df['unique_id'] == unique_id]
        df_temp_train = df_train[df_train['unique_id'] == unique_id]
        df_temp_train = pd.merge(
            df_temp_train, df_temp[['ds', 'unique_id', 'anomaly']],
            on=['ds', 'unique_id'], how='left'
        )
        df_temp_train["ds"] = pd.to_datetime(df_temp_train["ds"])
        df_temp_train['anomaly'] = df_temp_train['anomaly'].fillna(0)
        anomaly_filter = df_temp_train[df_temp_train['anomaly'] == 1]

        plt.figure(figsize=(10, 6))
        plt.plot(df_temp_train['ds'], df_temp_train['y'], label='Actual data')
        plt.scatter(
            anomaly_filter['ds'], anomaly_filter['y'],
            label='Anomaly', color='red', marker='*'
        )
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.title(f"Anomaly detection for unique_id: {unique_id}")
        plt.show()


def main() -> None:
    """Main function to load data, detect anomalies, and visualize results."""
    # Initialize NixtlaClient
    api_key = load_environment_variables()
    nixtla_client = NixtlaClient(api_key=api_key)

    # Load and prepare data
    df_train = load_and_prepare_data(
        DATASETTRAINPATH
    )

    # Detect anomalies
    anomalies_df = detect_anomalies(nixtla_client, df_train)

    # Plot anomalies (but not saving)
    plot_anomalies(anomalies_df, df_train)


if __name__ == "__main__":
    main()